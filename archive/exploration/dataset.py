"""
Dataset Generation and Caching for MIMO Systems
"""
import os
import numpy as np
import tensorflow as tf
import sionna
from sionna.phy.channel import cir_to_ofdm_channel

class CachedSionnaDataset:
    """Pre-generate and cache Sionna channels for training"""
    
    def __init__(self, system, dataset_size=50000, batch_size=512, 
                 cache_file='cached_sionna.npy',
                 seed=42):  # ✅ ADD SEED PARAMETER
        self.system = system
        self.dataset_size = dataset_size
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.seed = seed  # ✅ STORE SEED
        
        print(f"\n{'='*80}")
        print(f"  CHANNEL DATASET")
        print(f"{'='*80}")
        print(f"  Target size: {dataset_size:,} samples")
        print(f"  Cache file: {cache_file}")
        print(f"  Seed: {seed}")  # ✅ SHOW SEED
        print(f"{'='*80}\n")
        
        if os.path.exists(cache_file):
            print(f"📂 Loading existing cached dataset...")
            self.load_cached_channels()
        else:
            print(f"🔄 Generating new channel dataset...")
            self.generate_and_cache_channels()
    
    def generate_and_cache_channels(self):
        """Generate channels using pre-allocated array"""
        import time
        
        # ✅ SET SEED FOR REPRODUCIBILITY
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
        batch_size = self.batch_size
        num_batches = self.dataset_size // batch_size
        
        print(f"Generating {num_batches} batches of {batch_size} samples each...\n")
        
        # Get shape from one batch
        self.system.new_topology(batch_size, seed=self.seed)  # ✅ USE SEED
        cir = self.system.channel_model(
            batch_size,
            self.system.rg.num_ofdm_symbols,
            1.0 / self.system.rg.ofdm_symbol_duration
        )
        h_freq_sample = cir_to_ofdm_channel(
            self.system.frequencies, *cir, normalize=True
        )
        
        # Pre-allocate array
        total_shape = (self.dataset_size,) + h_freq_sample.shape[1:]
        print(f"   Total shape: {total_shape}")
        print(f"   Memory: ~{self._estimate_memory(total_shape):.2f} GB\n")
        
        self.h_freq_all = np.zeros(total_shape, dtype=np.complex64)
        
        start_time = time.time()
        for i in range(num_batches):
            if i % 50 == 0:
                elapsed = time.time() - start_time
                progress = (i / num_batches) * 100
                if i > 0:
                    eta = elapsed / i * (num_batches - i)
                    print(f"   Progress: {i:4d}/{num_batches} ({progress:5.1f}%) | "
                          f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
            
            # ✅ USE DETERMINISTIC SEED FOR EACH BATCH
            batch_seed = self.seed + i if self.seed is not None else None
            self.system.new_topology(batch_size, seed=batch_seed)
            
            cir = self.system.channel_model(
                batch_size,
                self.system.rg.num_ofdm_symbols,
                1.0 / self.system.rg.ofdm_symbol_duration
            )
            h_freq = cir_to_ofdm_channel(self.system.frequencies, *cir, normalize=True)
            
            start_idx = i * batch_size
            self.h_freq_all[start_idx:start_idx + batch_size] = h_freq.numpy()
        
        print(f"\n✅ Generated {len(self.h_freq_all):,} samples in {(time.time()-start_time)/60:.1f}min")
        
        # Save
        print(f"\n💾 Saving to {self.cache_file}...")
        np.save(self.cache_file, self.h_freq_all)
        print(f"✅ Saved! Size: {os.path.getsize(self.cache_file)/(1024**3):.2f} GB\n")
    
    def load_cached_channels(self):
        """Load pre-generated channels"""
        import time
        start = time.time()
        
        if self.cache_file.endswith('.npy'):
            self.h_freq_all = np.load(self.cache_file)
        elif self.cache_file.endswith('.npz'):
            data = np.load(self.cache_file, allow_pickle=True)
            self.h_freq_all = data['h_freq']
        
        print(f"✅ Loaded {len(self.h_freq_all):,} samples in {time.time()-start:.1f}s")
        print(f"   Shape: {self.h_freq_all.shape}")
        print(f"   Mean power: {np.mean(np.abs(self.h_freq_all)**2):.6f}\n")
    
    def get_batch(self, batch_size):
        """Sample random batch with deterministic shuffling"""
        # ✅ USE NUMPY'S RANDOM STATE FOR DETERMINISTIC SAMPLING
        indices = np.random.choice(len(self.h_freq_all), size=batch_size, replace=True)
        return tf.constant(self.h_freq_all[indices], dtype=tf.complex64)
    
    @staticmethod
    def _estimate_memory(shape):
        """Estimate memory in GB"""
        return np.prod(shape) * 8 / (1024**3)  # complex64 = 8 bytes