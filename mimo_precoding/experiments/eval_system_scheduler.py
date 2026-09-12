"""
experiments/eval_system_scheduler.py — Priorité 6b (demande utilisateur) :
adaptation de `main_scheduler.py` (racine du projet, déc. 2025) au
pipeline actuel. Évaluation système complète -- Proportional Fair
Scheduler + Outer-Loop Link Adaptation (sélection MCS) + PHY Abstraction
(BLER cible) -- pour ajouter un niveau de réalisme (débit réellement
décodable via sélection MCS, pas juste la capacité de Shannon
théorique) en complément des figures principales du mémoire.

Corrections vs l'original :
- Canal UMi STANDARD verrouillé actuel (M=8,K=4, STANDARD_CONFIG,
  R=20m) au lieu de M=4/K=4 codé en dur.
- WMMSE corrigé (bisection sur µ, `precoders/classical.py` actuel) au lieu de
  l'ancien WMMSEPrecodedChannel non corrigé.
- Les 3 architectures signed_attn actuelles (SC/IB/TA-RB T=4, 83ep) au
  lieu de l'ancien checkpoint de décembre.
- Interface précodeur mise à jour : `system._call_precoder(h, no,
  training=False)` + `system.ch_helper.compute_effective_channel(h, g)`
  au lieu de `precoder((x, h))` + `precoder.compute_effective_channel`
  (l'ancien précodeur EST son propre PrecodedChannel ; l'actuel non).

Cadrage (rappel utilisateur) : sous-section courte de validation, pas
un pilier du chapitre -- une figure, deux paragraphes. Pas d'exploration
de la table MCS / du BLER cible au-delà du nécessaire.

CSI imparfait (ajouté après coup, demande utilisateur) : même principe
que partout ailleurs cette nuit (diag_csi_imperfect_*.py) -- le
précodeur voit un canal bruité (h_est, pilote --pilot_snr_db, bruit LS
gaussien), mais la propagation réelle / le calcul du SINR post-
égalisation utilisent le vrai canal (h_true). --pilot_snr_db absent =
CSI parfait (comportement original, h_est=h_true).

Usage:
    python3 experiments/eval_system_scheduler.py --smoke                    # 1 méthode x 1 SNR x 10 slots, CSI parfait
    python3 experiments/eval_system_scheduler.py                             # sweep complet, CSI parfait
    python3 experiments/eval_system_scheduler.py --pilot_snr_db 20           # sweep complet, CSI imparfait (pilote 20dB)
"""
import os, sys, json, time, argparse
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # results/, weights/ relatifs a la racine du paquet
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

from sionna.sys import PHYAbstraction, OuterLoopLinkAdaptation, PFSchedulerSUMIMO
from sionna.phy.utils import log2, insert_dims, ebnodb2no
from sionna.phy.ofdm import LMMSEPostEqualizationSINR
from sionna.phy.channel import cir_to_ofdm_channel

from system import MU_MIMO_System
from precoders.classical import rzf_precoder, wmmse_precoder
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from channel_config import STANDARD_CONFIG

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
CKPTS = {
    'SC': ('single_sc', 'results/diag_front_a_signed_attn.json'),
    'IB': ('intra_rb', 'results/diag_front_a_signed_attn_intra_rb.json'),
    'TA-RB': ('ta_rb_residual', 'results/diag_tarb_residual_signed_attn_T4_train.json'),
}
COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SC': '#1baf7a', 'IB': '#eda100', 'TA-RB': '#e87ba4'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SC': '^', 'IB': 'D', 'TA-RB': 'v'}


def noisy_channel(h_freq, pilot_snr_db, rng):
    """Même bruit LS gaussien que diag_csi_imperfect_*.py partout ailleurs
    cette nuit -- pilote de qualité pilot_snr_db, indépendant du SNR data."""
    sigma2 = 1.0 / (10.0 ** (pilot_snr_db / 10.0))
    shape = h_freq.shape
    noise = rng.normal(0, np.sqrt(sigma2 / 2), size=(2,) + tuple(shape)).astype(np.float32)
    return h_freq + tf.cast(tf.complex(noise[0], noise[1]), h_freq.dtype)


class SystemLevelSimulator(tf.keras.Model):
    """PF scheduler + OLLA (MCS selection) + PHY abstraction, batch_size=1
    (required for scheduler state). Same structure as main_scheduler.py's
    SystemLevelSimulator_WithScheduler, interface calls updated to the
    current MU_MIMO_System (see module docstring)."""

    def __init__(self, base_system, num_slots=100, bler_target=0.1):
        super().__init__()
        self.base_system = base_system
        self.num_slots = num_slots
        self.bler_target = bler_target
        self.rg = base_system.rg
        self.sm = base_system.sm
        self.num_users = base_system.num_users
        bs1 = [1, 1]
        self.phy_abs = PHYAbstraction()
        self.olla = OuterLoopLinkAdaptation(self.phy_abs, num_ut=self.num_users, batch_size=bs1)
        self.scheduler = PFSchedulerSUMIMO(
            num_ut=self.num_users, num_freq_res=self.rg.fft_size,
            num_ofdm_sym=self.rg.num_ofdm_symbols, batch_size=bs1,
            num_streams_per_ut=self.rg.num_streams_per_tx, beta=0.98)
        self.lmmse_sinr = LMMSEPostEqualizationSINR(resource_grid=self.rg, stream_management=self.sm)

    def _reset_state(self):
        self.olla.reset()
        self.olla.bler_target = self.bler_target
        self.olla.olla_delta_up = 0.2
        harq = -tf.ones([1, 1, self.num_users], dtype=tf.int32)
        sinr_fb = tf.ones([1, 1, self.num_users], dtype=tf.float32)
        bits = tf.zeros([1, 1, self.num_users], dtype=tf.int32)
        return harq, sinr_fb, bits

    def simulate_slot(self, h_true, h_est, no, harq_feedback, sinr_eff_feedback, num_decoded_bits, mcs_table_index=1):
        """h_est = canal vu par le précodeur (== h_true si CSI parfait) ;
        h_true = canal réel, utilisé pour la propagation / le SINR post-
        égalisation (le récepteur est toujours supposé avoir une
        connaissance parfaite du canal effectif pour la détection --
        seule l'estimée CSI côté précodeur/TX est dégradée, même
        convention que diag_csi_imperfect_*.py partout ailleurs)."""
        h_true = h_true[:1, ...]
        h_est = h_est[:1, ...]

        rate_est = log2(1.0 + tf.pow(10.0, self.olla.sinr_eff_db_last / 10.0))
        rate_est = insert_dims(rate_est, 2, axis=-2)
        rate_est = tf.tile(rate_est, [1, 1, self.rg.num_ofdm_symbols, self.rg.fft_size, 1])

        is_scheduled = self.scheduler(num_decoded_bits, rate_est)
        num_allocated_re = tf.reduce_sum(tf.cast(is_scheduled, tf.int32), axis=[-1, -3, -4])

        # -- précodage sur h_est (interface actuelle, cf. docstring) : RZF/WMMSE
        # n'ont pas de system.precoder (None, gérés inline dans MU_MIMO_System.call()) --
        if self.base_system.precoder_type == 'rzf':
            g = rzf_precoder(h_est, self.sm, no=no)
        elif self.base_system.precoder_type == 'wmmse':
            g = wmmse_precoder(h_est, no, self.sm)
        else:
            g = self.base_system._call_precoder(h_est, no, training=False)
        h_eff = self.base_system.ch_helper.compute_effective_channel(h_true, g)
        sinr = self.lmmse_sinr(h_eff, no=no, interference_whitening=True)[:1, ...]

        is_scheduled_mask = tf.squeeze(tf.cast(is_scheduled, tf.float32), axis=1)
        sinr_scheduled = sinr * is_scheduled_mask

        mcs_index = self.olla(num_allocated_re, harq_feedback=harq_feedback, sinr_eff=sinr_eff_feedback)

        sinr_for_phy_abs = tf.expand_dims(sinr_scheduled, axis=1)
        num_decoded_bits_new, harq_feedback_new, sinr_eff_new, _, _ = self.phy_abs(
            mcs_index, sinr=sinr_for_phy_abs, mcs_table_index=mcs_table_index, mcs_category=1)

        sinr_eff_feedback_new = tf.where(num_allocated_re > 0, sinr_eff_new, tf.zeros_like(sinr_eff_new))

        return {'num_decoded_bits': num_decoded_bits_new, 'harq_feedback': harq_feedback_new,
                'sinr_eff_feedback': sinr_eff_feedback_new, 'mcs_index': mcs_index,
                'num_allocated_re': num_allocated_re}

    def evaluate(self, ebno_db, pilot_snr_db=None, rng=None):
        harq_fb, sinr_fb, decoded_bits = self._reset_state()
        total_bits = np.zeros(self.num_users)
        mcs_history = []

        for slot in range(self.num_slots):
            self.base_system.new_topology(1)
            cir = self.base_system.channel_model(1, self.rg.num_ofdm_symbols, 1.0 / self.rg.ofdm_symbol_duration)
            h_true = cir_to_ofdm_channel(self.base_system.frequencies, *cir, normalize=True)
            h_est = h_true if pilot_snr_db is None else noisy_channel(h_true, pilot_snr_db, rng)
            no = ebnodb2no(tf.constant(ebno_db, tf.float32), self.base_system.num_bits_per_symbol, 0.5, self.rg)

            r = self.simulate_slot(h_true, h_est, no, harq_fb, sinr_fb, decoded_bits)
            harq_fb, sinr_fb, decoded_bits = r['harq_feedback'], r['sinr_eff_feedback'], r['num_decoded_bits']
            total_bits += r['num_decoded_bits'].numpy()[0, 0, :]
            mcs_history.append(r['mcs_index'].numpy()[0, 0, :])

        avg_tput = total_bits / self.num_slots
        sum_tput = float(np.sum(avg_tput))
        jain = float((np.sum(avg_tput) ** 2) / (self.num_users * np.sum(avg_tput ** 2) + 1e-12))
        slot_dur = self.rg.ofdm_symbol_duration * self.rg.num_ofdm_symbols
        bw = self.rg.fft_size * self.rg.subcarrier_spacing
        spectral_eff = sum_tput / slot_dur / bw

        return {'sum_throughput': sum_tput, 'spectral_efficiency': spectral_eff,
                'jain_fairness': jain, 'per_user_throughput': avg_tput.tolist(),
                'mcs_mean': float(np.mean(mcs_history)), 'mcs_history': np.array(mcs_history).tolist()}


def build_system(name):
    if name == 'RZF':
        return MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    if name == 'WMMSE':
        return MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='wmmse')
    arch, json_path = CKPTS[name]
    with open(json_path) as f:
        ckpt = json.load(f)['best_ckpt']
    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=arch, embed_dim=128, num_heads=4, num_layers=4,
                             tokens_per_rb=4)
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
                  embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
    if name == 'SC':
        system.precoder = SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
    elif name == 'IB':
        system.precoder = IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
    else:
        system.precoder = TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=4, **kwargs)
    dummy_h = tf.zeros([1, K, 1, 1, M, system.rg.num_ofdm_symbols, system.rg.fft_size], dtype=tf.complex64)
    _ = system.precoder(dummy_h, no=tf.constant(0.01), training=False)
    ok = system.load_weights_from(ckpt)
    assert ok, f"chargement échoué {name}"
    return system


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true', help='1 méthode x 1 SNR x 10 slots, pour chiffrer le runtime')
    p.add_argument('--num_slots', type=int, default=100)
    p.add_argument('--pilot_snr_db', type=float, default=None,
                    help='CSI imparfait : qualité du pilote (dB). Absent = CSI parfait (comportement original).')
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    os.makedirs('results', exist_ok=True)
    rng = np.random.RandomState(args.seed)
    csi_tag = '' if args.pilot_snr_db is None else f'_csi_imperfect_pilot{int(args.pilot_snr_db)}dB'
    csi_label = 'CSI parfait' if args.pilot_snr_db is None else f'CSI imparfait (pilote {args.pilot_snr_db:.0f}dB)'

    if args.smoke:
        methods, snr_range, num_slots = ['SC'], [15.0], 10
    else:
        methods, snr_range, num_slots = ['RZF', 'WMMSE', 'SC', 'IB', 'TA-RB'], [0.0, 5.0, 10.0, 15.0, 17.5, 20.0], args.num_slots

    all_results = {}
    for name in methods:
        print(f"\n{'='*70}\n{name} -- {csi_label}\n{'='*70}", flush=True)
        t_build = time.time()
        system = build_system(name)
        sim = SystemLevelSimulator(system, num_slots=num_slots)
        print(f"  (système construit en {time.time()-t_build:.1f}s)", flush=True)
        all_results[name] = {}
        for snr in snr_range:
            t0 = time.time()
            r = sim.evaluate(snr, pilot_snr_db=args.pilot_snr_db, rng=rng)
            dt = time.time() - t0
            all_results[name][snr] = r
            print(f"  SNR={snr:5.1f}dB | spectral_eff={r['spectral_efficiency']:6.2f} bps/Hz | "
                  f"sum_tput={r['sum_throughput']:9.1f} bits/slot | jain={r['jain_fairness']:.3f} | "
                  f"mcs_mean={r['mcs_mean']:.1f} | ({dt:.1f}s, {dt/num_slots*1000:.0f}ms/slot)", flush=True)

    if args.smoke:
        print(f"\n✅ Smoke test terminé -- extrapolation runtime sweep complet "
              f"(5 méthodes x 6 SNR x 100 slots) : ~{dt/num_slots*100*5*6/60:.1f}min")
        sys.exit(0)

    out = {name: {str(snr): {k: v for k, v in r.items() if k != 'mcs_history'} for snr, r in res.items()}
           for name, res in all_results.items()}
    out_json = f'results/diag_system_scheduler_eval{csi_tag}.json'
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Sauvé -> {out_json}")

    # -- figure a 3 panneaux --
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name in methods:
        res = all_results[name]
        se = [res[s]['spectral_efficiency'] for s in snr_range]
        st = [res[s]['sum_throughput'] for s in snr_range]
        ja = [res[s]['jain_fairness'] for s in snr_range]
        axes[0].plot(snr_range, se, marker=MARKERS[name], color=COLORS[name], label=name, linewidth=2, markersize=7)
        axes[1].plot(snr_range, st, marker=MARKERS[name], color=COLORS[name], label=name, linewidth=2, markersize=7)
        axes[2].plot(snr_range, ja, marker=MARKERS[name], color=COLORS[name], label=name, linewidth=2, markersize=7)
    axes[0].set(xlabel='SNR (dB)', ylabel='Efficacité spectrale (bps/Hz)', title='Avec scheduler + link adaptation')
    axes[1].set(xlabel='SNR (dB)', ylabel='Débit total (bits/slot)', title='Débit total')
    axes[2].set(xlabel='SNR (dB)', ylabel="Indice d'équité de Jain", title='Équité entre utilisateurs')
    for ax in axes:
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'results/diag_system_scheduler_eval{csi_tag}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'results/diag_system_scheduler_eval{csi_tag}.pdf', bbox_inches='tight')
    print(f"✅ Figure -> results/diag_system_scheduler_eval{csi_tag}.{{png,pdf}}")
