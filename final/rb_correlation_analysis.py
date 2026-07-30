# =============================================================================
# rb_correlation_analysis.py
# Measures intra-RB channel correlation for different rb_size values.
# Run this BEFORE deciding on rb_size for the transformer.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def analyze_rb_correlation(h_freq_path: str,
                            rb_sizes_to_test: list = [3, 6, 12],
                            num_samples: int = 2000,
                            save_path: str = './rb_correlation_analysis.png'):
    """
    Loads cached single-user channels and computes intra-RB correlation
    for multiple candidate rb_size values.

    Args:
        h_freq_path:      path to .npz cache file (contains key 'h_freq')
        rb_sizes_to_test: list of SC-per-token values to evaluate
        num_samples:      number of single-user samples to analyse
        save_path:        where to save the figure
    """

    # ─── Load data ────────────────────────────────────────────────────────────
    print(f"Loading channel data from: {h_freq_path}")
    data   = np.load(h_freq_path)
    h_all  = data['h_freq']          # [N, 1, 1, 1, M, ofdm, fft]
    N      = min(num_samples, len(h_all))
    h_all  = h_all[:N]

    print(f"  Loaded {N} samples")
    print(f"  Shape: {h_all.shape}")
    print(f"  Dtype: {h_all.dtype}")

    # Squeeze to [N, M, ofdm, fft]
    h = h_all[:, 0, 0, 0, :, :, :]   # [N, M, ofdm, fft]
    N, M, num_ofdm, fft_size = h.shape

    print(f"\n  Antennas:      {M}")
    print(f"  OFDM symbols:  {num_ofdm}")
    print(f"  Subcarriers:   {fft_size}")

    # ─── Helper: pearson correlation between two complex vectors ──────────────
    def complex_correlation(a, b):
        """
        Normalised complex correlation |<a, b>| / (||a|| ||b||)
        a, b: [..., M] complex
        Returns scalar in [0, 1].
        """
        dot   = np.sum(np.conj(a) * b, axis=-1)        # [..., ]
        norm  = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        return np.abs(dot) / (norm + 1e-12)

    # ─── Analysis 1: SC-to-SC correlation across the full band ────────────────
    # For each pair of subcarriers (sc_i, sc_j), compute mean correlation
    # across all samples, antennas, ofdm symbols.
    print("\n" + "="*60)
    print("  ANALYSIS 1: Full-band SC-to-SC correlation matrix")
    print("="*60)

    # Flatten to [N*ofdm, M, fft]
    h_flat = h.transpose(0, 2, 1, 3).reshape(-1, M, fft_size)  # [N*ofdm, M, fft]

    # Subsample for speed
    step     = max(1, len(h_flat) // 500)
    h_sub    = h_flat[::step]                                   # [~500, M, fft]

    # Correlation matrix [fft, fft]
    corr_matrix = np.zeros((fft_size, fft_size))
    for sc_i in range(fft_size):
        for sc_j in range(sc_i, fft_size):
            c = complex_correlation(h_sub[:, :, sc_i],
                                    h_sub[:, :, sc_j])          # [~500]
            val = float(np.mean(c))
            corr_matrix[sc_i, sc_j] = val
            corr_matrix[sc_j, sc_i] = val

    # Coherence bandwidth: SC offset where correlation drops below threshold
    thresholds    = [0.99, 0.95, 0.90]
    mean_corr_vs_offset = np.zeros(fft_size)
    for offset in range(fft_size):
        diag_vals = [corr_matrix[i, i + offset]
                     for i in range(fft_size - offset)]
        mean_corr_vs_offset[offset] = np.mean(diag_vals)

    print("\n  Mean correlation vs SC offset:")
    print(f"  {'Offset':>8} | {'Correlation':>12}")
    print(f"  {'-'*24}")
    for offset in [0, 1, 2, 3, 6, 12, 18, 24, 36]:
        if offset < fft_size:
            print(f"  {offset:>8} | {mean_corr_vs_offset[offset]:>12.4f}")

    print("\n  Coherence bandwidth (SC offset where corr drops below):")
    for thresh in thresholds:
        crossings = np.where(mean_corr_vs_offset < thresh)[0]
        bw = crossings[0] if len(crossings) > 0 else fft_size
        print(f"    Corr < {thresh:.2f} : offset = {bw} SCs  "
              f"({'> full band' if bw >= fft_size else f'{bw} SCs'})")

    # ─── Analysis 2: Intra-RB correlation for each candidate rb_size ──────────
    print("\n" + "="*60)
    print("  ANALYSIS 2: Intra-token correlation per rb_size candidate")
    print("="*60)
    print(f"\n  {'rb_size':>8} | {'tokens':>7} | "
          f"{'mean corr (adj)':>16} | {'mean corr (all)':>16} | "
          f"{'min corr':>10} | {'% > 0.99':>10} | {'% > 0.95':>10}")
    print(f"  {'-'*90}")

    rb_results = {}
    for rb_size in rb_sizes_to_test:
        if fft_size % rb_size != 0:
            print(f"  {rb_size:>8} | SKIPPED (72 % {rb_size} != 0)")
            continue

        num_tokens = fft_size // rb_size

        # For each token, compute all pairwise correlations among its SCs
        corr_adjacent = []   # correlation between adjacent SCs within token
        corr_all_pairs = []  # all pairs within token
        corr_min_per_token = []

        for tok in range(num_tokens):
            sc_start = tok * rb_size
            sc_end   = sc_start + rb_size
            # SCs in this token: indices [sc_start, ..., sc_end-1]

            token_corrs = []
            for i in range(rb_size):
                for j in range(i + 1, rb_size):
                    c = complex_correlation(h_sub[:, :, sc_start + i],
                                            h_sub[:, :, sc_start + j])
                    val = float(np.mean(c))
                    token_corrs.append(val)
                    if j == i + 1:               # adjacent pair
                        corr_adjacent.append(val)

            corr_all_pairs.extend(token_corrs)
            corr_min_per_token.append(min(token_corrs))

        corr_adjacent  = np.array(corr_adjacent)
        corr_all_pairs = np.array(corr_all_pairs)
        corr_min       = np.array(corr_min_per_token)

        pct_99 = 100 * np.mean(corr_all_pairs > 0.99)
        pct_95 = 100 * np.mean(corr_all_pairs > 0.95)

        rb_results[rb_size] = {
            'num_tokens':        num_tokens,
            'corr_adjacent':     corr_adjacent,
            'corr_all_pairs':    corr_all_pairs,
            'corr_min_per_token': corr_min,
            'mean_adj':          float(np.mean(corr_adjacent)),
            'mean_all':          float(np.mean(corr_all_pairs)),
            'min_all':           float(np.min(corr_all_pairs)),
            'pct_99':            pct_99,
            'pct_95':            pct_95,
        }

        print(f"  {rb_size:>8} | {num_tokens:>7} | "
              f"{np.mean(corr_adjacent):>16.4f} | "
              f"{np.mean(corr_all_pairs):>16.4f} | "
              f"{np.min(corr_all_pairs):>10.4f} | "
              f"{pct_99:>9.1f}% | "
              f"{pct_95:>9.1f}%")

    # ─── Analysis 3: Precoder variation within RB ─────────────────────────────
    # How much does the OPTIMAL (RZF) precoder actually vary within a token?
    # This tells us whether the coarse-then-refine approach is justified.
    print("\n" + "="*60)
    print("  ANALYSIS 3: Channel variation within token (normalised)")
    print("  (measures how different first vs last SC in each token are)")
    print("="*60)
    print(f"\n  {'rb_size':>8} | {'mean |h_last - h_first| / |h_mean|':>36} | "
          f"{'max':>8} | {'std':>8}")
    print(f"  {'-'*70}")

    for rb_size in rb_sizes_to_test:
        if fft_size % rb_size != 0:
            continue
        num_tokens = fft_size // rb_size

        # h_flat: [~500, M, fft]
        relative_variations = []
        for tok in range(num_tokens):
            sc_start = tok * rb_size
            h_first  = h_sub[:, :, sc_start]
            h_last   = h_sub[:, :, sc_start + rb_size - 1]
            h_mean_t = np.mean(h_sub[:, :, sc_start:sc_start + rb_size], axis=2)

            diff      = np.linalg.norm(h_last  - h_first, axis=1)  # [~500]
            ref       = np.linalg.norm(h_mean_t, axis=1)            # [~500]
            rel_var   = diff / (ref + 1e-12)                         # [~500]
            relative_variations.append(float(np.mean(rel_var)))

        rv = np.array(relative_variations)
        print(f"  {rb_size:>8} | {np.mean(rv):>36.4f} | "
              f"{np.max(rv):>8.4f} | {np.std(rv):>8.4f}")

    # ─── Plotting ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = plt.cm.tab10(np.linspace(0, 1, len(rb_sizes_to_test)))

    # Plot 1: correlation vs SC offset (coherence bandwidth)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(range(fft_size), mean_corr_vs_offset, 'k-', linewidth=2)
    for thresh in thresholds:
        ax1.axhline(thresh, linestyle='--', alpha=0.6,
                    label=f'corr = {thresh}')
    for rb_size in rb_sizes_to_test:
        if rb_size in rb_results:
            ax1.axvspan(0, rb_size, alpha=0.08,
                        color=colors[rb_sizes_to_test.index(rb_size)],
                        label=f'rb_size={rb_size}')
    ax1.set_xlabel('SC offset', fontsize=12)
    ax1.set_ylabel('Mean correlation', fontsize=12)
    ax1.set_title('Channel Correlation vs Subcarrier Offset\n'
                  '(shaded = intra-token span for each rb_size)', fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(36, fft_size - 1))
    ax1.set_ylim(0.5, 1.01)

    # Plot 2: correlation matrix heatmap
    ax2 = fig.add_subplot(gs[0, 2])
    im  = ax2.imshow(corr_matrix, cmap='hot', vmin=0.8, vmax=1.0,
                     aspect='auto')
    plt.colorbar(im, ax=ax2)
    for rb_size in rb_sizes_to_test:
        if rb_size in rb_results:
            for tok in range(fft_size // rb_size):
                sc = tok * rb_size
                ax2.axhline(sc - 0.5, color='cyan', linewidth=0.5, alpha=0.5)
                ax2.axvline(sc - 0.5, color='cyan', linewidth=0.5, alpha=0.5)
    ax2.set_title(f'SC Correlation Matrix\n(cyan grid = rb_size={rb_sizes_to_test[-1]})',
                  fontsize=11)
    ax2.set_xlabel('Subcarrier index')
    ax2.set_ylabel('Subcarrier index')

    # Plot 3: distribution of intra-token correlations per rb_size
    ax3 = fig.add_subplot(gs[1, 0])
    for i, rb_size in enumerate(rb_sizes_to_test):
        if rb_size not in rb_results:
            continue
        vals = rb_results[rb_size]['corr_all_pairs']
        ax3.hist(vals, bins=50, alpha=0.6, color=colors[i],
                 label=f'rb={rb_size} (μ={np.mean(vals):.3f})',
                 density=True)
    ax3.axvline(0.99, color='red',  linestyle='--', label='0.99')
    ax3.axvline(0.95, color='orange', linestyle='--', label='0.95')
    ax3.set_xlabel('Intra-token pairwise correlation')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Intra-Token\nPairwise Correlations')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: per-token minimum correlation
    ax4 = fig.add_subplot(gs[1, 1])
    for i, rb_size in enumerate(rb_sizes_to_test):
        if rb_size not in rb_results:
            continue
        num_tokens = fft_size // rb_size
        vals       = rb_results[rb_size]['corr_min_per_token']
        ax4.plot(range(num_tokens), vals, 'o-', color=colors[i],
                 label=f'rb={rb_size}', linewidth=2, markersize=5)
    ax4.axhline(0.99, color='red',    linestyle='--', alpha=0.7, label='0.99')
    ax4.axhline(0.95, color='orange', linestyle='--', alpha=0.7, label='0.95')
    ax4.set_xlabel('Token index')
    ax4.set_ylabel('Min pairwise correlation within token')
    ax4.set_title('Worst-Case Intra-Token Correlation\nper Token (across frequency)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: relative channel variation (first vs last SC in token)
    ax5 = fig.add_subplot(gs[1, 2])
    for i, rb_size in enumerate(rb_sizes_to_test):
        if fft_size % rb_size != 0:
            continue
        num_tokens = fft_size // rb_size
        rel_vars   = []
        for tok in range(num_tokens):
            sc_start = tok * rb_size
            h_first  = h_sub[:, :, sc_start]
            h_last   = h_sub[:, :, sc_start + rb_size - 1]
            h_mean_t = np.mean(h_sub[:, :, sc_start:sc_start + rb_size], axis=2)
            diff     = np.linalg.norm(h_last  - h_first, axis=1)
            ref      = np.linalg.norm(h_mean_t, axis=1)
            rel_vars.append(float(np.mean(diff / (ref + 1e-12))))
        ax5.plot(range(num_tokens), rel_vars, 'o-', color=colors[i],
                 label=f'rb={rb_size}', linewidth=2, markersize=5)
    ax5.set_xlabel('Token index')
    ax5.set_ylabel('|h_last - h_first| / |h_mean|')
    ax5.set_title('Relative Channel Variation\nWithin Token (first vs last SC)')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.suptitle('RB/Token Correlation Analysis — Channel Dataset',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Figure saved: {save_path}")

    # ─── Decision guidance ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  DECISION GUIDANCE")
    print("="*60)
    for rb_size in rb_sizes_to_test:
        if rb_size not in rb_results:
            continue
        r = rb_results[rb_size]
        print(f"\n  rb_size = {rb_size}  ({r['num_tokens']} tokens):")
        print(f"    Mean intra-token corr (all pairs): {r['mean_all']:.4f}")
        print(f"    % pairs with corr > 0.99:          {r['pct_99']:.1f}%")
        print(f"    % pairs with corr > 0.95:          {r['pct_95']:.1f}%")

        if r['pct_99'] > 95:
            verdict = ("✅ Very high correlation. "
                       "Mean-only summary may be sufficient. "
                       "Using mean+first+last is safe overkill.")
        elif r['pct_95'] > 90:
            verdict = ("⚠️  Moderate correlation. "
                       "Mean alone will lose info. "
                       "Use mean+first+last or reduce rb_size further.")
        else:
            verdict = ("❌ Low correlation. "
                       "This rb_size groups too many SCs. "
                       "Reduce rb_size or abandon RB grouping.")
        print(f"    Verdict: {verdict}")

    return rb_results


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    import sys

    # ── Edit these to match your setup ────────────────────────────────────────
    CACHE_FILE   = '/export/tmp/sala/sionna_base_5k_8x4.npz'
    RB_SIZES     = [3, 6, 12]      # candidates to compare
    NUM_SAMPLES  = 2000
    SAVE_PATH    = './rb_correlation_analysis.png'
    # ──────────────────────────────────────────────────────────────────────────

    results = analyze_rb_correlation(
        h_freq_path      = CACHE_FILE,
        rb_sizes_to_test = RB_SIZES,
        num_samples      = NUM_SAMPLES,
        save_path        = SAVE_PATH,
    )