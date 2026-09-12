"""
Redo the M64K32 classical comparison (RZF-Full, RZF-RB12, WMMSE) on CPU,
since GPU results at this scale were found to be numerically non-
reproducible (process-to-process variance up to 3x on the same config,
while CPU gives tight, reproducible agreement -- see diagnose_variance.py
investigation). This becomes the trustworthy source for the M64K32 final
numbers, superseding the GPU-computed classical_comparison_M64K32.*
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from classical_comparison import evaluate_config, BATCH_SIZE, NUM_BATCHES, RB_SIZE, WMMSE_ITERS
from channel_config import MASSIVE_CONFIG

if __name__ == '__main__':
    results = evaluate_config('M64K32_CPU', MASSIVE_CONFIG,
                               checkpoint_path='./results/classical_comparison_M64K32_CPU_checkpoint.npy')

    os.makedirs('./results', exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(results['snr'], results['rzf_full'], 'o-', label='RZF (Full CSI)')
    ax.plot(results['snr'], results['rzf_rb12'], 's--', label=f'RZF (RB={RB_SIZE})')
    ax.plot(results['snr'], results['wmmse'], '^-', label=f'WMMSE ({WMMSE_ITERS} iters)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title('M64K32 (CPU-verified): RZF vs WMMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('./results/classical_comparison_M64K32_CPU.png', dpi=150)
    fig.savefig('./results/classical_comparison_M64K32_CPU.pdf')
    plt.close(fig)
    np.save('./results/classical_comparison_M64K32_CPU.npy', results)

    print("\nSaved CPU-verified M64K32 comparison to ./results/classical_comparison_M64K32_CPU.*")
