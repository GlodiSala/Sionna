"""
Joint-Cluster Dataset Generation for MIMO Precoding
====================================================
REVISION 2026-08-07 (§0 fix, see SESSION_NUIT_RESUME.md) : replaces the old
"SAGE-HB" mono-user + random-recombination pipeline.

Why this changed: channel_config.py's locked channel (REVISION (b)) creates
the RZF-WMMSE gap by drawing the K users of a sample JOINTLY, close to each
other around a shared random cluster center (gen_topology_clustered). The
old pipeline here drew 5 000 SINGLE-user channels independently (each its
own random cluster center), cached them, then formed "K-user" training
batches by picking K random indices out of the 5 000 (np.random.choice,
replace=False) -- i.e. recombining users from up to 5 000 different,
unrelated cluster draws. That silently destroys the spatial correlation the
new channel is built on: a batch of 4 users picked at random from 5 000
independent draws are essentially never from the same cluster, so the
model would train on a channel that is statistically "no gap", even though
channel_config.py itself is correct. This was invisible because the old
narrow-angle-window channel didn't depend on user-to-user correlation, so
recombination was harmless under it -- it only became wrong when the
channel mechanism changed under it.

Fix (option (a) from the resume doc): stop caching single-user draws and
recombining. Every stored sample is now a full K-user tuple produced by a
SINGLE call to channel_config.set_locked_topology(..., num_ut=K) -- so
every "user" in a sample is, by construction, drawn from the same cluster
center as its co-scheduled users. No recombination step exists anymore for
this to go wrong in.

Trade-off (documented, not hidden): this gives up the old ~98% storage
"saving" (250 000 effective samples reconstructed combinatorially from
5 000 base draws). There is no substitute for that trick once correctness
requires every sample to be an independent joint draw -- each stored
sample really does cost K times the single-user storage. In exchange, a
finite pool of directly-drawn joint samples is reused across training
steps via ordinary with-replacement minibatch sampling (standard practice
for any finite training set; not a re-introduction of the same bug, since
the correlation lives WITHIN each stored sample, not between samples).
"""

import os
import numpy as np
import tensorflow as tf
import time
from pathlib import Path
from tqdm import tqdm


import sionna
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm import ResourceGrid, RemoveNulledSubcarriers
from sionna.phy.channel.tr38901 import AntennaArray, UMi, UMa


# =============================================================================
# SHARED SIONNA SYSTEM SETUP -- factored out, was duplicated per-generator
# =============================================================================

def _build_sionna_system(num_tx_ant, carrier_freq, fft_size,
                          num_ofdm_symbols, subcarrier_spacing, scenario):
    """ResourceGrid + AntennaArrays + channel model, shared by whichever
    generator needs to draw raw CIRs. Single BS array of num_tx_ant
    elements (dual-pol cross), single-antenna UTs, direction=downlink,
    pathloss/shadow fading disabled (locked channel only cares about
    small-scale fading / multipath structure -- see channel_config.py)."""
    rg = ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        subcarrier_spacing=subcarrier_spacing,
        num_tx=1, num_streams_per_tx=1,
        cyclic_prefix_length=6,
        num_guard_carriers=[5, 6],
        dc_null=True,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 11]
    )

    ut_array = AntennaArray(
        num_rows=1, num_cols=1,
        polarization="single", polarization_type="V",
        antenna_pattern="omni", carrier_frequency=carrier_freq)

    bs_array = AntennaArray(
        num_rows=1, num_cols=int(num_tx_ant / 2),
        polarization="dual", polarization_type="cross",
        antenna_pattern="38.901", carrier_frequency=carrier_freq)

    # UMi/UMa partagent exactement la même signature de constructeur (Sionna
    # tr38901) -- ajout UMa (demande utilisateur, exploration canal, même
    # session) pour tester si une sélectivité fréquentielle plus marquée
    # (rho intra-RB plus bas, cf. diag_uma_selectivity.py) change le
    # classement des architectures. scenario reste un paramètre de bout en
    # bout (déjà le cas pour JointClusterGenerator/set_locked_topology/
    # gen_topology_clustered) -- seul ce dispatch était UMi-only.
    _model_cls = {"umi": UMi, "uma": UMa}.get(scenario.lower())
    if _model_cls is None:
        raise NotImplementedError(f"Scenario {scenario} not implemented")
    channel_model = _model_cls(
        carrier_frequency=carrier_freq,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction='downlink',
        enable_pathloss=False,
        enable_shadow_fading=False)

    frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)

    return rg, channel_model, frequencies


# =============================================================================
# JOINT-CLUSTER CHANNEL GENERATOR
# =============================================================================

class JointClusterGenerator:
    """
    Generates K-user channel tuples JOINTLY -- one call to
    channel_config.set_locked_topology(..., num_ut=num_users) per sample
    batch, so all K users in a sample share the same random cluster center
    (see module docstring for why this replaces SionnaSingleUserGenerator).
    """

    def __init__(self, num_tx_ant, num_users, cluster_radius_m,
                 indoor_probability=0.0, carrier_freq=2.6e9, fft_size=96,
                 num_ofdm_symbols=14, subcarrier_spacing=30e3,
                 scenario="umi", seed=42):
        self.num_tx_ant         = num_tx_ant
        self.num_users          = num_users
        self.cluster_radius_m   = cluster_radius_m
        self.indoor_probability = indoor_probability
        self.scenario           = scenario
        self.seed                = seed

        print(f"\n{'='*80}")
        print(f"JOINT-CLUSTER CHANNEL GENERATOR")
        print(f"{'='*80}")
        print(f"Scenario: {scenario.upper()}")
        print(f"TX Antennas: {num_tx_ant}  |  Users/sample (joint draw): {num_users}")
        print(f"Cluster radius: {cluster_radius_m} m")
        print(f"FFT Size: {fft_size}  |  OFDM Symbols: {num_ofdm_symbols}")
        print(f"Carrier Freq: {carrier_freq/1e9:.1f} GHz")
        print(f"{'='*80}\n")

        self.rg, self.channel_model, self.frequencies = _build_sionna_system(
            num_tx_ant, carrier_freq, fft_size, num_ofdm_symbols,
            subcarrier_spacing, scenario)

    def generate_joint_cluster_channels(self, num_samples: int,
                                         batch_size: int = 512) -> np.ndarray:
        """
        Returns [num_samples, num_users, 1, 1, num_tx_ant, num_ofdm, fft]
        -- already in the shape training consumes directly (no recombination
        needed downstream: num_users here IS the batch's rx axis, produced
        by a single joint topology draw per sample).
        """
        from channel_config import set_locked_topology

        num_batches  = (num_samples + batch_size - 1) // batch_size
        h_freq_list  = []
        start_time   = time.time()

        with tqdm(total=num_samples,
                  unit='samples',
                  desc='Generating joint-cluster channels',
                  bar_format='{l_bar}{bar:30}{r_bar}') as pbar:

            for i in range(num_batches):
                current_batch_size = min(batch_size,
                                          num_samples - i * batch_size)

                # Joint draw: num_ut=self.num_users users per topology call,
                # all around the SAME random cluster center (see
                # channel_config.gen_topology_clustered). This is the fix:
                # the old code called this with num_ut=1 in a loop and
                # recombined afterwards, which is what broke the correlation.
                set_locked_topology(self.channel_model, current_batch_size,
                                     num_ut=self.num_users,
                                     cluster_radius_m=self.cluster_radius_m,
                                     force_los=False,
                                     indoor_probability=self.indoor_probability,
                                     scenario=self.scenario)

                cir = self.channel_model(
                    current_batch_size,
                    self.rg.num_ofdm_symbols,
                    1.0 / self.rg.ofdm_symbol_duration)

                h_freq = cir_to_ofdm_channel(
                    self.frequencies, *cir, normalize=True)

                h_freq_list.append(h_freq.numpy())
                pbar.update(current_batch_size)

        h_freq_all    = np.concatenate(h_freq_list, axis=0)
        elapsed_total = time.time() - start_time

        print(f"✅ Generated {len(h_freq_all):,} joint {self.num_users}-user "
              f"samples in {elapsed_total:.1f}s")
        print(f"   Shape: {h_freq_all.shape}")
        print(f"   Mean power: {np.mean(np.abs(h_freq_all)**2):.6f}\n")

        return h_freq_all


# =============================================================================
# JOINT-CLUSTER DATASET -- replaces SAGEHBDataset (no recombination)
# =============================================================================

class JointClusterDataset:
    """
    tf.data pipeline with prefetch, drawing minibatches by uniform
    with-replacement sampling over the joint-cluster pool. No recombination
    of users across samples -- each pool entry is already a valid,
    independently-drawn K-user cluster tuple, so ordinary finite-dataset
    minibatch sampling (the same thing any dataset with a finite pool of
    examples does across epochs) cannot reintroduce the §0 bug.
    """

    def __init__(self, h_freq_joint: np.ndarray, seed: int = 42):
        self.h_freq_all       = h_freq_joint
        self.num_base_samples = len(h_freq_joint)
        self.num_users        = h_freq_joint.shape[1]
        self.num_tx_ant        = h_freq_joint.shape[4]
        self.num_ofdm          = h_freq_joint.shape[5]
        self.fft_size          = h_freq_joint.shape[6]
        self.seed               = seed

        print(f"\n{'='*80}")
        print(f"JOINT-CLUSTER DATASET")
        print(f"{'='*80}")
        print(f"Joint {self.num_users}-user samples in pool: {self.num_base_samples:,}")
        print(f"Channel shape: [M={self.num_tx_ant}, ofdm={self.num_ofdm}, fft={self.fft_size}]")
        print(f"Minibatches drawn with replacement over the pool (no recombination).")
        print(f"{'='*80}\n")

        self.rng = np.random.RandomState(seed)

        self._tf_dataset  = None   # built lazily on first get_batch
        self._tf_iterator = None

    def __len__(self):
        return self.num_base_samples

    def _infinite_generator(self, batch_size: int):
        """Yields batches forever -- tf.data handles the repeat."""
        while True:
            indices = self.rng.randint(0, self.num_base_samples, size=batch_size)
            batch = self.h_freq_all[indices]
            # [B, U, 1, 1, M, ofdm, fft]  -- already the right shape, each
            # of the U users in a row comes from the same joint draw.
            yield batch.astype(np.complex64)

    def _build_tf_pipeline(self, batch_size: int):
        output_shape = (batch_size,
                        self.num_users, 1, 1,
                        self.num_tx_ant,
                        self.num_ofdm,
                        self.fft_size)

        ds = tf.data.Dataset.from_generator(
            lambda: self._infinite_generator(batch_size),
            output_signature=tf.TensorSpec(
                shape=output_shape, dtype=tf.complex64))

        ds = ds.prefetch(buffer_size=2)

        self._tf_dataset  = ds
        self._tf_iterator = iter(ds)
        self._built_batch_size = batch_size

    def get_batch(self, batch_size: int) -> tf.Tensor:
        if self._tf_iterator is None or self._built_batch_size != batch_size:
            self._build_tf_pipeline(batch_size)
        return next(self._tf_iterator)


# =============================================================================
# CACHED SIONNA DATASET -- main entry point, unchanged interface
# =============================================================================

class CachedSionnaDataset:
    """
    Interface principale avec cache. `dataset_size` is now the number of
    directly-drawn joint K-user cluster samples in the pool (there is no
    augmentation multiplier anymore -- see module docstring for why). Pick
    it generously enough for training diversity, but note storage now
    scales with num_users (K), unlike the old single-user cache.

    `batch_size` here is only the CIR-GENERATION chunk size (not the
    training minibatch size, that's whatever get_batch(batch_size) is
    called with). Default lowered 512->128 (§0 fix): the old single-user
    generator's CIR intermediate tensor was per-1-user; joint generation
    multiplies it by num_users (K), so the same batch_size that was safe
    before can OOM now -- confirmed empirically (bs=512 OOM'd on a shared
    GPU at K=4/M=8, "Mul" op inside cir_to_ofdm_channel; bs<=256 safe even
    at K=8/M=32, the larger MASSIVE config). 128 keeps a margin for
    contention from other users' jobs on the shared GPU.
    """

    def __init__(self, system,
                 dataset_size: int = 20000,
                 batch_size: int = 128,
                 cache_file: str = 'cached_sionna_base.npz',
                 cluster_radius_m: float = 20.0,
                 indoor_probability: float = 0.0,
                 scenario: str = 'umi',
                 augmentation_multiplier=None,   # deprecated, kept for call compat
                 seed: int = 42):

        if augmentation_multiplier is not None:
            print(f"⚠️  augmentation_multiplier={augmentation_multiplier} passed but "
                  f"IGNORED: the SAGE-HB recombination it used to control was removed "
                  f"(§0 fix, SESSION_NUIT_RESUME.md) -- every one of the "
                  f"dataset_size={dataset_size} samples is now an independently, "
                  f"jointly-drawn K-user cluster tuple, nothing left to augment.")

        self.system             = system
        self.dataset_size       = dataset_size
        self.cache_file         = cache_file
        self.batch_size         = batch_size
        self.cluster_radius_m   = cluster_radius_m
        self.indoor_probability = indoor_probability
        self.scenario            = scenario
        self.seed               = seed

        print(f"\n{'='*80}")
        print(f"CACHED SIONNA DATASET (joint-cluster, §0 fix)")
        print(f"{'='*80}")
        print(f"Target pool size: {dataset_size:,} joint {system.num_users}-user samples")
        print(f"Cluster radius: {cluster_radius_m} m")
        print(f"Cache file: {cache_file}")
        print(f"Seed: {seed}")
        print(f"{'='*80}\n")

        if os.path.exists(cache_file):
            print(f"📂 Loading existing cached dataset...")
            self.h_freq_all = self._load_cached_channels()
        else:
            print(f"🔄 Generating new joint-cluster channel dataset...")
            self.h_freq_all = self._generate_and_cache_channels()

        self.cluster_dataset = JointClusterDataset(
            h_freq_joint=self.h_freq_all, seed=seed)

    def _generate_and_cache_channels(self) -> np.ndarray:
        generator = JointClusterGenerator(
            num_tx_ant=self.system.num_bs_antennas,
            num_users=self.system.num_users,
            cluster_radius_m=self.cluster_radius_m,
            indoor_probability=self.indoor_probability,
            carrier_freq=2.6e9,
            fft_size=self.system.rg.fft_size,
            num_ofdm_symbols=self.system.rg.num_ofdm_symbols,
            subcarrier_spacing=self.system.rg.subcarrier_spacing,
            scenario=self.scenario,
            seed=self.seed)

        h_freq = generator.generate_joint_cluster_channels(
            num_samples=self.dataset_size,
            batch_size=self.batch_size)

        os.makedirs(os.path.dirname(self.cache_file) or '.', exist_ok=True)
        np.savez_compressed(self.cache_file, h_freq=h_freq)

        file_size_mb = os.path.getsize(self.cache_file) / (1024**2)
        print(f"💾 Saved to: {self.cache_file} ({file_size_mb:.1f} MB)\n")
        return h_freq

    def _load_cached_channels(self) -> np.ndarray:
        start  = time.time()
        data   = np.load(self.cache_file)
        h_freq = data['h_freq']

        # Safety check (§0 fix): a pre-fix mono-user cache has axis-1
        # size 1 regardless of num_users, and would silently be
        # misinterpreted as a valid (but degenerate) joint pool. Fail loud
        # instead -- see SESSION_NUIT_RESUME.md §0.
        if h_freq.shape[1] != self.system.num_users:
            raise ValueError(
                f"Cache '{self.cache_file}' has axis-1 size {h_freq.shape[1]} "
                f"but system.num_users={self.system.num_users}. This looks like "
                f"a stale pre-§0-fix cache (mono-user, shape[1]==1) or a cache "
                f"for a different K. Delete/rename it and regenerate -- do NOT "
                f"bypass this check, loading it would silently retrain on "
                f"uncorrelated users again (see SESSION_NUIT_RESUME.md §0).")

        print(f"✅ Loaded {len(h_freq):,} joint {h_freq.shape[1]}-user samples in "
              f"{time.time()-start:.1f}s")
        print(f"   Shape: {h_freq.shape}")
        print(f"   Mean power: {np.mean(np.abs(h_freq)**2):.6f}\n")
        return h_freq

    def get_batch(self, batch_size: int) -> tf.Tensor:
        return self.cluster_dataset.get_batch(batch_size)

    @property
    def effective_dataset_size(self) -> int:
        return len(self.cluster_dataset)
