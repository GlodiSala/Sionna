"""
SAGE-HB Style Dataset Generation for MIMO Precoding
====================================================
Fixes vs version précédente :
- 5 000 base samples (était 1 000)
- tf.data pipeline avec prefetch (Fix 4) — GPU ne attend plus le CPU
- indices uniques par sample (np.random.choice replace=False)
"""

import os
import numpy as np
import tensorflow as tf
import time
from pathlib import Path
from tqdm import tqdm


import sionna
from sionna.phy.channel import (cir_to_ofdm_channel, subcarrier_frequencies,
                                  gen_single_sector_topology)
from sionna.phy.ofdm import ResourceGrid, RemoveNulledSubcarriers
from sionna.phy.channel.tr38901 import AntennaArray, UMi


# =============================================================================
# SIONNA SINGLE-USER CHANNEL GENERATOR  — inchangé
# =============================================================================

class SionnaSingleUserGenerator:
    def __init__(self, num_tx_ant=8, carrier_freq=2.6e9, fft_size=96,
                 num_ofdm_symbols=14, subcarrier_spacing=30e3,
                 scenario="umi", seed=42):
        self.num_tx_ant        = num_tx_ant
        self.carrier_freq      = carrier_freq
        self.fft_size          = fft_size
        self.num_ofdm_symbols  = num_ofdm_symbols
        self.subcarrier_spacing = subcarrier_spacing
        self.scenario          = scenario
        self.seed              = seed

        print(f"\n{'='*80}")
        print(f"SIONNA SINGLE-USER CHANNEL GENERATOR")
        print(f"{'='*80}")
        print(f"Scenario: {scenario.upper()}")
        print(f"TX Antennas: {num_tx_ant}")
        print(f"RX Antennas per user: 1 (single antenna)")
        print(f"FFT Size: {fft_size}")
        print(f"OFDM Symbols: {num_ofdm_symbols}")
        print(f"Carrier Freq: {carrier_freq/1e9:.1f} GHz")
        print(f"{'='*80}\n")

        self._setup_sionna_system(scenario)

    def _setup_sionna_system(self, scenario):
        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=1, num_streams_per_tx=1,
            cyclic_prefix_length=6,
            num_guard_carriers=[5, 6],
            dc_null=True,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11]
        )

        self.ut_array = AntennaArray(
            num_rows=1, num_cols=1,
            polarization="single", polarization_type="V",
            antenna_pattern="omni", carrier_frequency=self.carrier_freq)

        self.bs_array = AntennaArray(
            num_rows=1, num_cols=int(self.num_tx_ant / 2),
            polarization="dual", polarization_type="cross",
            antenna_pattern="38.901", carrier_frequency=self.carrier_freq)

        if scenario.lower() == "umi":
            self.channel_model = UMi(
                carrier_frequency=self.carrier_freq,
                o2i_model="low",
                ut_array=self.ut_array,
                bs_array=self.bs_array,
                direction='downlink',
                enable_pathloss=False,
                enable_shadow_fading=False)
        else:
            raise NotImplementedError(f"Scenario {scenario} not implemented")

        self.frequencies = subcarrier_frequencies(
            self.rg.fft_size, self.rg.subcarrier_spacing)

        print(f"✅ Sionna system configured")
        print(f"   Effective subcarriers: {self.rg.num_effective_subcarriers}")
        print(f"   Total subcarriers: {self.fft_size}")
        print(f"   Nulled: {self.fft_size - self.rg.num_effective_subcarriers}\n")

    def generate_single_user_channels(self, num_samples: int,
                                   batch_size: int = 512) -> np.ndarray:
        """
        Generate SINGLE USER channel realizations.

        Returns full FFT size (including nulled subcarriers) for compatibility
        with call_with_cached_channel().

        Shape: [num_samples, 1, 1, 1, num_tx_ant, num_ofdm_symbols, fft_size]
        """
        from tqdm import tqdm

        num_batches = (num_samples + batch_size - 1) // batch_size
        h_freq_list = []
        start_time  = time.time()

        with tqdm(total=num_samples,
                unit='samples',
                desc='Generating channels',
                bar_format='{l_bar}{bar:30}{r_bar}') as pbar:

            for i in range(num_batches):
                current_batch_size = min(batch_size,
                                        num_samples - i * batch_size)

                # Locked M64K36_ang15_los config (narrow azimuth window +
                # forced LOS + indoor_probability=0) -- see channel_config.py
                # for the validation trail. Each single-user draw here gets
                # placed inside the same window/LOS state; the SAGE-HB
                # augmentation step (SAGEHBDataset) later combines random
                # subsets of these into multi-user samples, which correctly
                # reproduces "K users randomly placed within the window,
                # forced LOS" as long as each constituent draw already is.
                # Falls back to plain 3GPP topology if channel_config isn't
                # importable, so this generator stays usable standalone.
                try:
                    from channel_config import set_locked_topology
                    set_locked_topology(self.channel_model,
                                         current_batch_size, num_ut=1,
                                         scenario=self.scenario)
                except ImportError:
                    topology = gen_single_sector_topology(
                        batch_size=current_batch_size,
                        num_ut=1,
                        scenario=self.scenario,
                        min_ut_velocity=0.0,
                        max_ut_velocity=0.0)
                    self.channel_model.set_topology(*topology)

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

        print(f"✅ Generated {len(h_freq_all):,} samples in {elapsed_total:.1f}s")
        print(f"   Shape: {h_freq_all.shape}")
        print(f"   Mean power: {np.mean(np.abs(h_freq_all)**2):.6f}\n")

        return h_freq_all


# =============================================================================
# SAGE-HB AUGMENTED DATASET — Fix 4 : tf.data prefetch
# =============================================================================

class SAGEHBDataset:
    """
    SAGE-HB dataset avec tf.data pipeline et prefetch.

    Changements vs version précédente :
    - replace=False dans np.random.choice → pas de collision user dans un sample
    - tf.data.Dataset.from_generator + prefetch → GPU ne attend pas le CPU
    - get_batch() délègue au pipeline prefetché
    """

    def __init__(self, h_freq_single_user: np.ndarray,
                 num_users: int = 4,
                 augmentation_multiplier: int = 50,
                 seed: int = 42):

        self.h_freq_base         = h_freq_single_user
        self.num_base_samples    = len(h_freq_single_user)
        self.num_users           = num_users
        self.augmentation_multiplier = augmentation_multiplier
        self.effective_size      = self.num_base_samples * augmentation_multiplier
        self.seed                = seed

        self.num_tx_ant = h_freq_single_user.shape[4]
        self.num_ofdm   = h_freq_single_user.shape[5]
        self.fft_size   = h_freq_single_user.shape[6]

        print(f"\n{'='*80}")
        print(f"SAGE-HB AUGMENTED DATASET")
        print(f"{'='*80}")
        print(f"Base single-user samples: {self.num_base_samples:,}")
        print(f"Users per multi-user sample: {self.num_users}")
        print(f"Augmentation multiplier: {augmentation_multiplier}×")
        print(f"Effective dataset size: {self.effective_size:,}")
        print(f"Single-user channel shape: "
              f"[M={self.num_tx_ant}, ofdm={self.num_ofdm}, fft={self.fft_size}]")
        without = self.effective_size * self.num_users
        reduction = (1 - self.num_base_samples / without) * 100
        print(f"Storage savings: {reduction:.1f}%")
        print(f"{'='*80}\n")

        self.rng = np.random.RandomState(seed)

        # ── Fix 4 : pipeline tf.data avec prefetch ───────────────────────
        # Le générateur Python tourne sur CPU en parallèle du GPU
        self._tf_dataset  = None   # construit lazily au premier get_batch
        self._tf_iterator = None

    def __len__(self):
        return self.effective_size

    # ── Générateur interne utilisé par tf.data ────────────────────────────
    def _infinite_generator(self, batch_size: int):
        """Génère des batches à l'infini — tf.data s'occupe du repeat."""
        while True:
            # replace=False : aucun user ne partage le même canal dans un sample
            # Pour batch_size=256 et num_base=5000, ça reste rapide
            indices = np.stack([
                self.rng.choice(self.num_base_samples,
                                size=self.num_users,
                                replace=False)
                for _ in range(batch_size)
            ], axis=0)
            # [batch_size, num_users]

            batch = self.h_freq_base[indices]
            # [B, U, 1, 1, 1, M, ofdm, fft]
            batch = np.squeeze(batch, axis=2)
            # [B, U, 1, 1, M, ofdm, fft]

            yield batch.astype(np.complex64)

    def _build_tf_pipeline(self, batch_size: int):
        """Construit le pipeline tf.data prefetché pour un batch_size donné."""
        output_shape = (batch_size,
                        self.num_users, 1, 1,
                        self.num_tx_ant,
                        self.num_ofdm,
                        self.fft_size)

        ds = tf.data.Dataset.from_generator(
            lambda: self._infinite_generator(batch_size),
            output_signature=tf.TensorSpec(
                shape=output_shape, dtype=tf.complex64))

        # prefetch(2) : pendant que le GPU traite batch N,
        # le CPU prépare déjà le batch N+1
        ds = ds.prefetch(buffer_size=2)

        self._tf_dataset  = ds
        self._tf_iterator = iter(ds)
        self._built_batch_size = batch_size

    def get_batch(self, batch_size: int) -> tf.Tensor:
        """Interface principale pour l'entraînement."""
        # Reconstruit le pipeline si batch_size change (rare)
        if self._tf_iterator is None or self._built_batch_size != batch_size:
            self._build_tf_pipeline(batch_size)
        return next(self._tf_iterator)


# =============================================================================
# CACHED SIONNA DATASET — inchangé sauf DATASET_SIZE par défaut
# =============================================================================

class CachedSionnaDataset:
    """
    Interface principale avec cache et augmentation SAGE-HB.

    Changement : dataset_size par défaut 5 000 (était 10 000 dans l'original,
    1 000 dans la version utilisée). Passer 5 000 dans main.py.
    """

    def __init__(self, system,
                 dataset_size: int = 5000,      # ← était 1 000
                 batch_size: int = 512,
                 cache_file: str = 'cached_sionna_base.npz',
                 augmentation_multiplier: int = 50,   # ← était 50, inchangé
                 seed: int = 42):

        self.system                  = system
        self.dataset_size            = dataset_size
        self.cache_file              = cache_file
        self.batch_size              = batch_size
        self.augmentation_multiplier = augmentation_multiplier
        self.seed                    = seed

        print(f"\n{'='*80}")
        print(f"CACHED SIONNA DATASET")
        print(f"{'='*80}")
        print(f"Target BASE size: {dataset_size:,} single-user samples")
        print(f"Augmentation: {augmentation_multiplier}×")
        print(f"Effective size: {dataset_size * augmentation_multiplier:,} "
              f"multi-user samples")
        print(f"Cache file: {cache_file}")
        print(f"Seed: {seed}")
        print(f"{'='*80}\n")

        if os.path.exists(cache_file):
            print(f"📂 Loading existing cached dataset...")
            self.h_freq_all = self._load_cached_channels()
        else:
            print(f"🔄 Generating new channel dataset...")
            self.h_freq_all = self._generate_and_cache_channels()

        self.sage_hb_dataset = SAGEHBDataset(
            h_freq_single_user=self.h_freq_all,
            num_users=system.num_users,
            augmentation_multiplier=augmentation_multiplier,
            seed=seed)

    def _generate_and_cache_channels(self) -> np.ndarray:
        generator = SionnaSingleUserGenerator(
            num_tx_ant=self.system.num_bs_antennas,
            carrier_freq=2.6e9,
            fft_size=self.system.rg.fft_size,
            num_ofdm_symbols=self.system.rg.num_ofdm_symbols,
            subcarrier_spacing=self.system.rg.subcarrier_spacing,
            scenario="umi",
            seed=self.seed)

        h_freq = generator.generate_single_user_channels(
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
        print(f"✅ Loaded {len(h_freq):,} base samples in "
              f"{time.time()-start:.1f}s")
        print(f"   Shape: {h_freq.shape}")
        print(f"   Mean power: {np.mean(np.abs(h_freq)**2):.6f}\n")
        return h_freq

    def get_batch(self, batch_size: int) -> tf.Tensor:
        return self.sage_hb_dataset.get_batch(batch_size)

    @property
    def effective_dataset_size(self) -> int:
        return len(self.sage_hb_dataset)