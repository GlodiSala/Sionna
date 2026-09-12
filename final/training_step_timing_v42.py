"""
Get training-step timing for TransformerPrecoderV4 specifically as v4.2
(the version we actually fixed and validated for the K=36 parameter
blow-up), since the first timing pass used MU_MIMO_System's default
version='v4.0' by omission.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from training_step_timing import run, mf
from channel_config import STANDARD_CONFIG, MASSIVE_CONFIG

if __name__ == '__main__':
    all_results = []
    for name, cfg in [('STANDARD', STANDARD_CONFIG), ('MASSIVE', MASSIVE_CONFIG)]:
        r = run(name, cfg, 'transformer_rb', version='v4.2')
        all_results.append(r)

    os.makedirs('./results', exist_ok=True)
    np.save('./results/training_step_timing_v42.npy', all_results, allow_pickle=True)

    print(f"\n{'='*90}\nSUMMARY (v4.2)\n{'='*90}")
    wu, ft = mf.TRAINING_CONFIG['warmup_epochs'], mf.TRAINING_CONFIG['finetune_epochs']
    for r in all_results:
        if r.get('failed'):
            print(f"  {r['name']:9s} M={r['M']:3d} K={r['K']:3d}  FAILED (OOM at all batch sizes tried)")
            continue
        total_s = wu * r['warmup_epoch_s'] + ft * r['finetune_epoch_s']
        print(f"  {r['name']:9s} M={r['M']:3d} K={r['K']:3d}  batch={r['batch_size_used']:3d}  "
              f"params={r['n_params']:>12,}  "
              f"warmup={r['t_warmup_ms']:6.1f}ms/step  finetune={r['t_finetune_ms']:6.1f}ms/step  "
              f"~total ({wu}+{ft} epochs)={total_s/60:.1f} min")
