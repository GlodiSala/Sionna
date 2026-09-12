"""
diag_validate_datasets_fix.py — Validation du fix §0 (SESSION_NUIT_RESUME.md).

Mesure rho(lag) et le gap RZF-WMMSE sur un batch tiré du VRAI pipeline
d'entraînement (datasets.py: CachedSionnaDataset -> JointClusterGenerator
-> JointClusterDataset.get_batch), pas juste sur channel_config.py
directement -- pour valider le code path que verra réellement
SupervisedTrainer, pas seulement le mécanisme sous-jacent.

Référence à comparer (channel_config.py, REVISION (b), déjà validé par
diag_spatial_cluster_r20_k8.py / diag_m32_final_round.py) :
  STANDARD (M8,K4,R=20m) : rho@95sc~0.31, gap 0dB +2.79%, 5dB +0.97%, 10dB +0.35%
  MASSIVE  (M32,K8,R=5m) : rho@95sc~0.37, gap 0dB +0.97%, 5dB +0.39%, 10dB +0.25%

Utilise un pool restreint (dataset_size petit, cache jetable dans le
scratchpad) -- pas le cache de production (voir main_finall.py pour
celui-ci) : ce script sert uniquement à vérifier que le mécanisme de
génération conjointe + l'échantillonnage par minibatch avec remise
préservent la corrélation, pas à produire le dataset d'entraînement final.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_validate_datasets_fix.py
"""
import os, sys, json, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from datasets import CachedSionnaDataset
from channel_config import STANDARD_CONFIG, MASSIVE_CONFIG
from precoders_w import rzf_precoder, wmmse_precoder
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR
from sionna.phy.utils import ebnodb2no

SNR_POINTS   = [0.0, 5.0, 10.0, 15.0, 20.0]
NUM_BATCHES  = 10   # même rigueur que diag_spatial_cluster_r20_k8 / m32_final_round
EVAL_BATCH   = 16
VALIDATE_POOL = 1000   # petit pool jetable, juste pour valider le mécanisme
SCRATCH_DIR  = '/tmp/claude-4849/-users-sala-Documents-test-projet-Trans-freq-Sionna/7ea26a8d-5dda-4f6b-9915-78d4e8816314/scratchpad'

REFERENCE = {
    'STANDARD': {'rho95': 0.314, 'gaps': {0.0: 2.79, 5.0: 0.97, 10.0: 0.35}},
    'MASSIVE':  {'rho95': 0.370, 'gaps': {0.0: 0.97, 5.0: 0.39, 10.0: 0.25}},
}


class DummySystem:
    """Minimal stand-in -- CachedSionnaDataset only reads these attributes."""
    def __init__(self, num_tx, num_rx, fft_size=96, num_ofdm_symbols=14,
                 subcarrier_spacing=30e3):
        self.num_bs_antennas = num_tx
        self.num_users        = num_rx

        class _RG:
            pass
        rg = _RG()
        rg.fft_size            = fft_size
        rg.num_ofdm_symbols    = num_ofdm_symbols
        rg.subcarrier_spacing  = subcarrier_spacing
        self.rg = rg


def freq_correlation(h_freq_np, lags=(4, 8, 12, 24, 48, 95)):
    """Same definition as diag_spatial_cluster_channel.freq_correlation --
    operates on RAW h_freq (all 96 subcarriers): the physical channel
    response array doesn't depend on which subcarriers are logically
    nulled for data mapping, so this is directly comparable to the
    documented reference numbers (measured on a no-guard/no-dc-null RG)."""
    h = np.squeeze(h_freq_np, axis=(2, 3))[:, :, :, 0, :]   # [B,K,M,fft]
    N = h.shape[-1]
    den = np.mean(np.abs(h) ** 2)
    return {lag: float(np.abs(np.mean(h[..., :N - lag] * np.conj(h[..., lag:]))) / den)
            for lag in lags}


def validate(name, cfg):
    M, K = cfg['NUM_TX'], cfg['NUM_RX']
    R    = cfg['CLUSTER_RADIUS_M']
    print(f'\n{"="*75}\n{name}  M={M},K={K},R={R}m  --  validation du fix §0\n{"="*75}', flush=True)

    dummy = DummySystem(M, K)
    cache_file = os.path.join(SCRATCH_DIR, f'validate_joint_{name}.npz')
    if os.path.exists(cache_file):
        os.remove(cache_file)   # toujours régénérer pour ce test de validation

    t0 = time.time()
    ds = CachedSionnaDataset(
        dummy, dataset_size=VALIDATE_POOL, batch_size=128,
        cache_file=cache_file,
        cluster_radius_m=R,
        indoor_probability=cfg['INDOOR_PROBABILITY'],
        seed=123)
    print(f'  Pool généré en {time.time()-t0:.1f}s', flush=True)

    # ── rho(lag) sur un batch tiré du pipeline get_batch() réel ──────────
    h_batch = ds.get_batch(EVAL_BATCH).numpy()
    corr = freq_correlation(h_batch)
    print('  Corrélation :', ' '.join(f'{l}sc={v:.3f}' for l, v in corr.items()), flush=True)

    # ── gap RZF-WMMSE, mêmes ingrédients que le pipeline d'entraînement ──
    rg = ResourceGrid(num_ofdm_symbols=14, fft_size=96, subcarrier_spacing=30e3,
                       num_tx=1, num_streams_per_tx=K, cyclic_prefix_length=6,
                       pilot_pattern='kronecker', pilot_ofdm_symbol_indices=[2, 11])
    rx_tx_association = np.ones([K, 1])
    sm = StreamManagement(rx_tx_association, num_streams_per_tx=K)
    helper      = RZFPrecodedChannel(rg, sm)
    lmmse_sinr  = LMMSEPostEqualizationSINR(rg, sm)

    def sum_rate(h_freq, g, no):
        h_eff = helper.compute_effective_channel(h_freq, g)
        sinr  = lmmse_sinr(h_eff, no=no, interference_whitening=True)
        return tf.reduce_sum(tf.reduce_mean(
            tf.reduce_mean(tf.math.log(1.0 + sinr) / tf.math.log(2.0), axis=[1, 2, 4]),
            axis=0))

    gaps = {}
    for snr in SNR_POINTS:
        snr_t = tf.constant(snr, tf.float32)
        no = ebnodb2no(snr_t, 2, 0.5, rg)
        r_rzf, r_wmmse = [], []
        for _ in range(NUM_BATCHES):
            h_freq = ds.get_batch(EVAL_BATCH)   # tirage réel via le pipeline (with-replacement pool sampling)
            g_rzf = rzf_precoder(h_freq, stream_management=sm, no=no)
            r_rzf.append(float(sum_rate(h_freq, g_rzf, no)))
            g_wmmse = wmmse_precoder(h_freq, no=no, stream_management=sm, num_iterations=10)
            r_wmmse.append(float(sum_rate(h_freq, g_wmmse, no)))
        rzf_m, wmmse_m = np.mean(r_rzf), np.mean(r_wmmse)
        gap_pct = 100.0 * (wmmse_m - rzf_m) / max(rzf_m, 1e-6)
        gaps[snr] = (rzf_m, wmmse_m, gap_pct)
        flag = '  <-- gap' if gap_pct > 0.3 else ''
        print(f'    SNR={snr:5.1f}dB | RZF={rzf_m:7.2f} | WMMSE={wmmse_m:7.2f} | gap={gap_pct:+.2f}%{flag}', flush=True)

    ref = REFERENCE[name]
    print(f'\n  Référence ({name}, channel_config.py §1) : rho@95sc~{ref["rho95"]:.3f}, '
          f'gaps ~{ref["gaps"]}', flush=True)

    return {'corr': corr, 'gaps': {str(k): v for k, v in gaps.items()}}


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    results = {}
    for name, cfg in [('STANDARD', STANDARD_CONFIG), ('MASSIVE', MASSIVE_CONFIG)]:
        results[name] = validate(name, cfg)
        tf.keras.backend.clear_session()

    with open('results/diag_validate_datasets_fix.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSauvé -> results/diag_validate_datasets_fix.json')
