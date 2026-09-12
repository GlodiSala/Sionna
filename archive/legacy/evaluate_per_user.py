import numpy as np
import tensorflow as tf
from sionna.phy.utils import ebnodb2no
import matplotlib.pyplot as plt

def compute_ber_per_user_np(b, b_hat):
    """Calcule le BER pour chaque utilisateur/stream"""
    b = b.numpy() if hasattr(b, 'numpy') else b
    b_hat = b_hat.numpy() if hasattr(b_hat, 'numpy') else b_hat
    
    b = np.squeeze(b, axis=1)
    b_hat = np.squeeze(b_hat, axis=1)
    
    batch_size = b.shape[0]
    num_streams = b.shape[1]
    k = b.shape[2]
    
    ber_per_user = []
    errors_per_user = []
    
    for stream_idx in range(num_streams):
        b_user = b[:, stream_idx, :]
        b_hat_user = b_hat[:, stream_idx, :]
        
        errors = np.sum(b_user != b_hat_user)
        total_bits = batch_size * k
        ber = errors / total_bits if total_bits > 0 else 0.0
        
        ber_per_user.append(ber)
        errors_per_user.append(errors)
    
    total_bits_per_user = batch_size * k
    
    return np.array(ber_per_user), np.array(errors_per_user), total_bits_per_user


def compute_block_errors_per_user(b, b_hat):
    """Compte les blocs erronés par utilisateur"""
    b = b.numpy() if hasattr(b, 'numpy') else b
    b_hat = b_hat.numpy() if hasattr(b_hat, 'numpy') else b_hat
    
    b = np.squeeze(b, axis=1)
    b_hat = np.squeeze(b_hat, axis=1)
    
    batch_size = b.shape[0]
    num_streams = b.shape[1]
    
    block_errors_per_user = []
    
    for stream_idx in range(num_streams):
        b_user = b[:, stream_idx, :]
        b_hat_user = b_hat[:, stream_idx, :]
        
        errors_per_block = np.sum(b_user != b_hat_user, axis=1)
        block_errors = np.sum(errors_per_block > 0)
        
        block_errors_per_user.append(block_errors)
    
    return np.array(block_errors_per_user), batch_size


def sim_ber_per_user(simulator, ebno_db, batch_size=256, 
                     num_target_block_errors=1000, max_mc_iter=1000):
    """Simulation Monte Carlo du BER par utilisateur"""
    num_users = simulator._num_streams_per_tx
    
    total_errors = np.zeros(num_users)
    total_bits = np.zeros(num_users)
    block_errors = np.zeros(num_users)
    total_blocks = np.zeros(num_users)
    
    ebno_tf = tf.constant(ebno_db, dtype=tf.float32)
    batch_size_tf = tf.constant(batch_size, dtype=tf.int32)
    
    iteration = 0
    
    while iteration < max_mc_iter:
        if np.all(block_errors >= num_target_block_errors):
            break
        
        b, b_hat, h_freq, g, precoding = simulator(
            batch_size_tf, ebno_tf, training=True  # ✅ training=False for eval
        )
        
        ber_batch, errors_batch, bits_batch = compute_ber_per_user_np(b, b_hat)
        block_err_batch, blocks_batch = compute_block_errors_per_user(b, b_hat)
        
        total_errors += errors_batch
        total_bits += bits_batch
        block_errors += block_err_batch
        total_blocks += blocks_batch
        
        iteration += 1
    
    ber_per_user = total_errors / total_bits
    bler_per_user = block_errors / total_blocks
    
    stats = {
        'ber_per_user': ber_per_user,
        'bler_per_user': bler_per_user,
        'total_errors': total_errors,
        'total_bits': total_bits,
        'block_errors': block_errors,
        'total_blocks': total_blocks,
        'iterations': iteration,
        'total_transmissions': iteration * batch_size
    }
    
    return ber_per_user, stats


def evaluate_ber_vs_ebno_per_user(simulator, ebno_range, batch_size=256,
                                   num_target_block_errors=1000, max_mc_iter=1000):
    """Évalue le BER par utilisateur sur une plage d'Eb/N0"""
    num_users = simulator._num_streams_per_tx
    num_points = len(ebno_range)
    
    ber_matrix = np.zeros((num_users, num_points))
    bler_matrix = np.zeros((num_users, num_points))
    ber_total_list = []
    iterations_list = []
    
    print("\n" + "="*80)
    print("SIMULATION MONTE CARLO - BER PAR UTILISATEUR")
    print("="*80)
    print(f"Batch size: {batch_size}")
    print(f"Target block errors per user: {num_target_block_errors}")
    print(f"Max iterations: {max_mc_iter}")
    print("="*80 + "\n")
    
    for idx, ebno_db in enumerate(ebno_range):
        print(f"Eb/N0 = {ebno_db:5.1f} dB ... ", end='', flush=True)
        
        ber_per_user, stats = sim_ber_per_user(
            simulator, ebno_db, batch_size,
            num_target_block_errors, max_mc_iter
        )
        
        ber_matrix[:, idx] = ber_per_user
        bler_matrix[:, idx] = stats['bler_per_user']
        iterations_list.append(stats['iterations'])
        
        ber_total = np.mean(ber_per_user)
        ber_total_list.append(ber_total)
        
        print(f"{stats['iterations']:4d} iter ({stats['total_transmissions']:6d} tx)")
        print(f"    BER Total: {ber_total:.2e}")
        
        for u in range(num_users):
            print(f"    User {u+1}: BER={ber_per_user[u]:.2e}, "
                  f"BLER={stats['bler_per_user'][u]:.2e}, "
                  f"Errors={int(stats['block_errors'][u])}")
    
    print("="*80 + "\n")
    
    results = {
        'ebno_range': ebno_range,
        'ber_per_user': ber_matrix,
        'bler_per_user': bler_matrix,
        'ber_total': np.array(ber_total_list),
        'iterations': iterations_list,
        'num_users': num_users
    }
    
    return results


def calculate_sum_rate_per_user_fixed(self, precoding_matrix, h_freq, noise_power=None):
    """
    Calculate sum rate for MULTI-USER MIMO
    
    Args:
        precoding_matrix: [B, K, num_rb, rb_size, M]
        h_freq: [B, K, N, 1, M, ofdm, fft] (Sionna format)
        noise_power: scalar
    
    Returns:
        sum_rate: scalar (total sum rate)
        metrics: dict with per-user rates
    """
    rb_size = self.simulator._rb_size
    
    if noise_power is None:
        noise_power = tf.constant(1e-13, dtype=tf.float32)
    
    # 1) Clean Sionna format
    # Expected: [B, K, N, 1, M, ofdm, fft]
    # For single antenna per user: [B, K, 1, 1, M, ofdm, fft]
    h_clean = tf.squeeze(h_freq, axis=[3])  # [B, K, N, M, ofdm, fft]
    
    # If N=1, squeeze that too
    if len(h_clean.shape) == 6 and tf.shape(h_clean)[2] == 1:
        h_clean = tf.squeeze(h_clean, axis=2)  # [B, K, M, ofdm, fft]
    
    # 2) Average over OFDM symbols
    h_avg = tf.reduce_mean(h_clean, axis=3)  # [B, K, M, fft]
    
    B = tf.shape(h_avg)[0]
    K = tf.shape(h_avg)[1]  # num_users
    M = tf.shape(h_avg)[2]  # num_tx_ant
    fft_size = tf.shape(h_avg)[3]
    
    num_rb = fft_size // rb_size
    
    # 3) Reshape into RB format
    h_rb = tf.reshape(h_avg, [B, K, M, num_rb, rb_size])
    h_rb = tf.transpose(h_rb, perm=[0, 1, 3, 4, 2])  # [B, K, num_rb, rb_size, M]
    
    # 4) Get precoding matrix (should already be in RB format)
    W_rb = precoding_matrix  # [B, K, num_rb, rb_size, M]
    
    # 5) Normalize power per RB per user (if not already done)
    power_per_rb = tf.reduce_sum(tf.abs(W_rb)**2, axis=[3, 4], keepdims=True)
    W_rb_normalized = W_rb / (tf.sqrt(power_per_rb) + 1e-12)
    
    # 6) Calculate effective channel per user: h_k * w_k
    # For user k: h_k [B, K, num_rb, rb_size, M], w_k [B, K, num_rb, rb_size, M]
    # Effective channel: sum over M dimension
    h_eff = tf.reduce_sum(
        h_rb * tf.math.conj(W_rb_normalized),
        axis=4  # Sum over TX antennas
    )  # [B, K, num_rb, rb_size]
    
    # Signal power per user: |h_k^H w_k|^2
    signal_power = tf.abs(h_eff) ** 2  # [B, K, num_rb, rb_size]
    
    # 7) Calculate interference from other users
    # For each user k, interference from user j≠k
    # Expand dimensions for broadcasting
    h_expanded = tf.expand_dims(h_rb, axis=1)  # [B, 1, K, num_rb, rb_size, M]
    W_expanded = tf.expand_dims(W_rb_normalized, axis=2)  # [B, K, 1, num_rb, rb_size, M]
    
    # Cross channel: h_k * w_j for all k,j
    cross_channel = tf.reduce_sum(
        h_expanded * tf.math.conj(W_expanded),
        axis=5  # Sum over M
    )  # [B, K, K, num_rb, rb_size]
    
    # Total interference power per user
    interference_power = tf.reduce_sum(
        tf.abs(cross_channel) ** 2,
        axis=2  # Sum over all j
    ) - signal_power  # [B, K, num_rb, rb_size] (exclude own signal)
    
    # 8) Calculate SINR
    SINR = signal_power / (interference_power + noise_power + 1e-12)
    
    # 9) Calculate rate per subcarrier
    rate_per_subcarrier = tf.math.log(1.0 + SINR) / tf.math.log(2.0)
    
    # 10) Sum rate per user (sum over RBs and subcarriers)
    rate_per_user_per_batch = tf.reduce_sum(
        rate_per_subcarrier, 
        axis=[2, 3]  # Sum over num_rb and rb_size
    )  # [B, K]
    
    # Average over batch
    rate_per_user = tf.reduce_mean(rate_per_user_per_batch, axis=0)  # [K]
    
    # Total sum rate
    total_sum_rate = tf.reduce_sum(rate_per_user)
    
    # Average SINR
    avg_sinr = tf.reduce_mean(SINR)
    avg_sinr_db = 10.0 * tf.math.log(avg_sinr + 1e-12) / tf.math.log(10.0)
    
    metrics = {
        'sinr_db': avg_sinr_db,
        'signal_power': tf.reduce_mean(signal_power),
        'interference_power': tf.reduce_mean(interference_power),
        'rate_per_user': rate_per_user,  # [K] - rate for each user
        'rate_per_subcarrier': tf.reduce_mean(rate_per_subcarrier)
    }
    
    return total_sum_rate, metrics
def compute_sum_rate_per_user(simulator, precoding_matrix, h_freq, noise_power):
    """
    ✅ FULLY FIXED: Calcule le sum rate PAR utilisateur
    Gère automatiquement le format RB ou fréquentiel du precoding matrix
    
    Returns:
        rate_per_user: [num_users] - débit par utilisateur en bps/Hz
        total_sum_rate: scalar - somme totale
    """
    rb_size = simulator._rb_size
    
    # ========== ÉTAPE 1: Nettoyer les dimensions Sionna du canal ==========
    if len(h_freq.shape) == 7:
        h_clean = tf.squeeze(h_freq, axis=[1, 3])  # [B, R, T, ofdm, fft]
    else:
        h_clean = h_freq
    
    # Moyenne sur symboles OFDM
    h_avg = tf.reduce_mean(h_clean, axis=3)  # [B, R, T, fft]
    h_avg = tf.transpose(h_avg, perm=[0, 1, 3, 2])  # [B, R, fft, T]
    
    B = tf.shape(h_avg)[0]
    R = tf.shape(h_avg)[1]
    fft_size = tf.shape(h_avg)[2]
    T = tf.shape(h_avg)[3]
    num_rb = fft_size // rb_size
    
    # Reshape canal en format RB
    h_rb = tf.reshape(h_avg, [B, R, num_rb, rb_size, T])
    
    # ========== ÉTAPE 2: Gérer le Precoding Matrix ==========
    W_shape = tf.shape(precoding_matrix)
    
    # ✅ CHECK: Le precoding matrix est-il déjà au format RB?
    # Format RB: [B, U, RB, rb_size, M] → 5 dimensions avec dim[3] == rb_size
    if len(precoding_matrix.shape) == 5:
        # Vérifier si c'est vraiment le format RB
        if precoding_matrix.shape[3] == rb_size:
            # ✅ DÉJÀ AU FORMAT RB! Pas besoin de reshape
            print(f"[DEBUG] Precoding matrix already in RB format: {precoding_matrix.shape}")
            W_rb = precoding_matrix
            
            # Normalisation par RB
            precoder_norm = tf.norm(W_rb, ord='euclidean', axis=[3, 4], keepdims=True)
            W_rb_normalized = W_rb / (precoder_norm + 1e-12)
        else:
            # Format non-standard, traiter comme fréquentiel
            print(f"[DEBUG] Non-standard 5D format, treating as frequency domain")
            W = precoding_matrix
            if len(W.shape) == 7:
                W = tf.squeeze(W, axis=[1, 3])
            W_avg = tf.reduce_mean(W, axis=3)
            W_avg = tf.transpose(W_avg, perm=[0, 1, 3, 2])
            
            S = tf.shape(W_avg)[1]
            W_rb = tf.reshape(W_avg, [B, S, num_rb, rb_size, T])
            
            precoder_norm = tf.norm(W_rb, axis=[3, 4], keepdims=True)
            W_rb_normalized = W_rb / (precoder_norm + 1e-12)
    
    elif len(precoding_matrix.shape) == 7:
        # ❌ Format Sionna complet: [B, 1, S, 1, T, ofdm, fft]
        print(f"[DEBUG] Sionna format detected: {precoding_matrix.shape}")
        W = tf.squeeze(precoding_matrix, axis=[1, 3])  # [B, S, T, ofdm, fft]
        W_avg = tf.reduce_mean(W, axis=3)  # [B, S, T, fft]
        W_avg = tf.transpose(W_avg, perm=[0, 1, 3, 2])  # [B, S, fft, T]
        
        S = tf.shape(W_avg)[1]
        W_rb = tf.reshape(W_avg, [B, S, num_rb, rb_size, T])
        
        precoder_norm = tf.norm(W_rb, axis=[3, 4], keepdims=True)
        W_rb_normalized = W_rb / (precoder_norm + 1e-12)
    
    else:
        # ❌ Format inattendu
        raise ValueError(f"Unexpected precoding matrix shape: {precoding_matrix.shape}")
    
    # ========== ÉTAPE 3: Calcul du SINR et du Rate ==========
    # Flatten pour calcul
    F = num_rb * rb_size
    h_flat = tf.reshape(h_rb, [B, R, F, T])
    W_flat = tf.reshape(W_rb_normalized, [B, R, F, T])
    
    # Canal effectif: H * W^H
    W_eff = tf.einsum('bufi,bujf->buij', h_flat, tf.math.conj(
        tf.transpose(W_flat, perm=[0, 1, 3, 2])))
    
    # SINR calculation
    diag_W = tf.linalg.diag_part(tf.abs(W_eff) ** 2)  # Signal power
    total_power = tf.reduce_sum(tf.abs(W_eff) ** 2, axis=3)
    interference = total_power - diag_W
    
    SINR = diag_W / (interference + noise_power + 1e-12)
    
    # Rate par sous-porteuse: log2(1 + SINR)
    rate_per_subcarrier = tf.math.log(1.0 + SINR) / tf.math.log(2.0)  # [B, U, U]
    
    # Rate par user (somme sur fréquences)
    rate_per_user_batch = tf.reduce_sum(rate_per_subcarrier, axis=2)  # [B, U]
    rate_per_user = tf.reduce_mean(rate_per_user_batch, axis=0)  # [U]
    
    # Sum rate total
    total_sum_rate = tf.reduce_sum(rate_per_user)
    
    return rate_per_user, total_sum_rate


def evaluate_sum_rate_per_user(simulator, ebno_range, batch_size=256, num_batches=100):
    """Évalue le sum rate par utilisateur sur une plage d'Eb/N0"""
    num_users = simulator._num_streams_per_tx
    num_points = len(ebno_range)
    
    rate_matrix = np.zeros((num_users, num_points))
    sum_rate_list = []
    
    print("\n" + "="*80)
    print("ÉVALUATION SUM RATE PAR UTILISATEUR")
    print("="*80)
    print(f"Batch size: {batch_size}")
    print(f"Batches per point: {num_batches}")
    print("="*80 + "\n")
    
    batch_size_tf = tf.constant(batch_size, dtype=tf.int32)
    
    for idx, ebno_db in enumerate(ebno_range):
        print(f"Eb/N0 = {ebno_db:5.1f} dB ... ", end='', flush=True)
        
        ebno_tf = tf.constant(ebno_db, dtype=tf.float32)
        noise_power = ebnodb2no(ebno_tf, 2, 0.5, simulator._rg)
        
        rates_accumulator = np.zeros(num_users)
        
        for _ in range(num_batches):
            b, b_hat, h_freq, g, precoding = simulator(
                batch_size_tf, ebno_tf, training=True  # ✅ training=False
            )
            
            if precoding is not None:
                rate_per_user, total_rate = compute_sum_rate_per_user(
                    simulator, precoding, h_freq, noise_power
                )
                
                rates_accumulator += rate_per_user.numpy()
        
        # Moyenne sur les batches
        avg_rates = rates_accumulator / num_batches
        rate_matrix[:, idx] = avg_rates
        sum_rate_list.append(np.sum(avg_rates))
        
        print(f"Sum Rate = {np.sum(avg_rates):6.2f} bps/Hz")
        for u in range(num_users):
            print(f"    User {u+1}: {avg_rates[u]:6.2f} bps/Hz")
    
    print("="*80 + "\n")
    
    results = {
        'ebno_range': ebno_range,
        'rate_per_user': rate_matrix,
        'sum_rate_total': np.array(sum_rate_list),
        'num_users': num_users
    }
    
    return results


def plot_results_per_user(ber_results, rate_results, ber_zf_results=None, 
                          training_history=None):
    """Graphiques des résultats par utilisateur"""
    # Ajuster layout
    if training_history is not None and 'bers_per_user' in training_history:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = np.array([[axes[0, 0], axes[0, 1]], 
                        [axes[1, 0], axes[1, 1]]])
    
    num_users = ber_results['num_users']
    ebno_range = ber_results['ebno_range']
    
    colors = ['b', 'r', 'g', 'm']
    markers = ['o', 's', '^', 'd']
    
    # 1. BER par utilisateur (évaluation)
    ax = axes[0, 0]
    for u in range(num_users):
        ax.semilogy(ebno_range, ber_results['ber_per_user'][u, :],
                   color=colors[u], marker=markers[u], 
                   label=f'User {u+1} (Transformer)', linewidth=2, markersize=6)
    
    if ber_zf_results is not None:
        for u in range(num_users):
            ax.semilogy(ebno_range, ber_zf_results['ber_per_user'][u, :],
                       color=colors[u], marker=markers[u], linestyle='--',
                       label=f'User {u+1} (ZF)', linewidth=1.5, markersize=5, alpha=0.6)
    
    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('BER par Utilisateur (Évaluation)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    # 2. Débit par utilisateur
    ax = axes[0, 1]
    for u in range(num_users):
        ax.plot(rate_results['ebno_range'], rate_results['rate_per_user'][u, :],
               color=colors[u], marker=markers[u],
               label=f'User {u+1}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('Débit (bps/Hz)', fontsize=12)
    ax.set_title('Débit par Utilisateur', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. BER par utilisateur pendant training
    if training_history is not None and 'bers_per_user' in training_history:
        ax = axes[0, 2]
        epochs = range(len(training_history['bers_per_user'][0]))
        for u in range(num_users):
            ax.semilogy(epochs, training_history['bers_per_user'][u],
                       color=colors[u], label=f'User {u+1}', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('BER', fontsize=12)
        ax.set_title('BER par Utilisateur (Training)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
    
    # 4. BER moyen
    ax = axes[1, 0]
    
    if 'ber_total' in ber_results:
        ber_avg = ber_results['ber_total']
    else:
        ber_avg = np.mean(ber_results['ber_per_user'], axis=0)
    
    ax.semilogy(ebno_range, ber_avg, 'b-o', label='Transformer (avg)', 
                linewidth=2.5, markersize=8)
    
    if ber_zf_results is not None:
        if 'ber_total' in ber_zf_results:
            ber_zf_avg = ber_zf_results['ber_total']
        else:
            ber_zf_avg = np.mean(ber_zf_results['ber_per_user'], axis=0)
        ax.semilogy(ebno_range, ber_zf_avg, 'r--s', label='ZF (avg)', 
                    linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('BER Total (Moyenne)', fontsize=12)
    ax.set_title('BER Total - Évaluation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    # 5. Sum Rate Total
    ax = axes[1, 1]
    ax.plot(rate_results['ebno_range'], rate_results['sum_rate_total'],
           'g-o', label='Sum Rate Total', linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=12)
    ax.set_title('Sum Rate Total', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 6. BER total pendant training
    if training_history is not None and 'bers' in training_history:
        if training_history.get('bers_per_user') is not None and axes.shape[1] > 2:
            ax = axes[1, 2]
            epochs = range(len(training_history['bers']))
            ax.semilogy(epochs, training_history['bers'], 'purple', 
                       linewidth=2.5, label='BER Total')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('BER Total', fontsize=12)
            ax.set_title('BER Total (Training)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig

def plot_ber_comparison_with_wmmse(ber_transformer, ber_zf, ber_wmmse, ebno_range, params_dict):
    """Plot BER comparison including WMMSE"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Total BER
    ax1 = axes[0]
    ax1.semilogy(ebno_range, ber_zf['ber_total'], 
                'r-o', label='Zero-Forcing', linewidth=2.5, markersize=8, alpha=0.8)
    ax1.semilogy(ebno_range, ber_wmmse['ber_total'], 
                'g-^', label='WMMSE', linewidth=2.5, markersize=8, alpha=0.8)  # ✅ NEW
    ax1.semilogy(ebno_range, ber_transformer['ber_total'], 
                'b-s', label='5D Transformer', linewidth=2.5, markersize=8, alpha=0.8)
    
    ax1.set_xlabel('Eb/N0 (dB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax1.set_title('BER Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='best')
    
    # BER per User @ 10 dB
    ax2 = axes[1]
    idx_10db = np.argmin(np.abs(ebno_range - 10))
    num_users = ber_transformer['ber_per_user'].shape[0]
    users = np.arange(1, num_users + 1)
    width = 0.25  # ✅ Adjusted for 3 bars
    
    ber_tf_users = ber_transformer['ber_per_user'][:, idx_10db]
    ber_zf_users = ber_zf['ber_per_user'][:, idx_10db]
    ber_wmmse_users = ber_wmmse['ber_per_user'][:, idx_10db]  # ✅ NEW
    
    ax2.bar(users - width, ber_zf_users, width, 
           label='ZF', color='red', alpha=0.7, edgecolor='black')
    ax2.bar(users, ber_wmmse_users, width, 
           label='WMMSE', color='green', alpha=0.7, edgecolor='black')  # ✅ NEW
    ax2.bar(users + width, ber_tf_users, width, 
           label='Transformer', color='blue', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('User', fontsize=12, fontweight='bold')
    ax2.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax2.set_title(f'BER per User @ Eb/N0 = 10 dB', fontsize=14, fontweight='bold')
    ax2.set_xticks(users)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both', axis='y')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    return fig