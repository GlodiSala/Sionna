import tensorflow as tf
import numpy as np
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.channel import cir_to_ofdm_channel


class SumRateTrainer:
    """
    Correct trainer that uses standard sum rate calculation
    RB grouping is ONLY in transformer architecture, not in sum rate calculation
    """
    
    def __init__(self, simulator, learning_rate=1e-3):
        self.simulator = simulator
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        print(f"\n{'='*80}")
        print("SUM RATE TRAINER - CORRECT IMPLEMENTATION")
        print(f"{'='*80}")
        print(f"Learning rate: {learning_rate}")
        print(f"Loss: -SumRate (pure spectral efficiency optimization)")
        
        # Verify trainable variables
        if hasattr(simulator, '_precoder'):
            n_vars = len(simulator._precoder.trainable_variables)
            if n_vars > 0:
                total_params = sum([tf.size(v).numpy() for v in simulator._precoder.trainable_variables])
                print(f"Trainable variables: {n_vars}")
                print(f"Total parameters: {total_params:,}")
            else:
                print("⚠️ WARNING: No trainable variables found!")
        print(f"{'='*80}\n")
    
    def calculate_sum_rate_with_precoding_matrix(self, h_freq, W, noise_power):
        """
        ✅ PERFECT sum rate calculation using precoding matrix

        Args:
            h_freq: [B, R, 1, 1, M, ofdm, fft] - channel
            W: Precoding matrix - format depends on precoder type
            noise_power: scalar
        """
        # Clean dimensions
        h = tf.squeeze(h_freq, axis=[2, 3])  # [B, R, M, ofdm, fft]

        # For transformer: W is in Sionna format [B, 1, M, R, ofdm, fft]
        # Extract: [B, M, R, ofdm, fft]
        if len(W.shape) == 6:
            W_clean = tf.squeeze(W, axis=1)  # [B, M, R, ofdm, fft]
        else:
            W_clean = W

        B = tf.shape(h)[0]
        R = tf.shape(h)[1]
        M = tf.shape(h)[2]
        num_ofdm = tf.shape(h)[3]
        fft_size = tf.shape(h)[4]

        # ===== Calculate effective channel: H * W =====
        # h_eff[b,r,u,o,f] = sum_m h[b,r,m,o,f] * W[b,m,u,o,f]
        # This gives us the channel gain from stream u to receiver r
        h_eff = tf.einsum('brmof,bmuof->bruof', h, W_clean)
        # Shape: [B, R, R, ofdm, fft] where h_eff[b,r,u] = channel from user u to receiver r

        # ===== Calculate SINR per subcarrier =====
        h_eff_power = tf.abs(h_eff) ** 2  # [B, R, R, ofdm, fft]

        # Signal power: diagonal elements (r == u)
        signal_power = tf.linalg.diag_part(h_eff_power)  # [B, R, ofdm, fft]

        # Interference: sum of all off-diagonal elements
        total_power = tf.reduce_sum(h_eff_power, axis=2)  # [B, R, ofdm, fft]
        interference_power = total_power - signal_power

        # SINR
        SINR = signal_power / (interference_power + noise_power + 1e-12)

        # Rate per subcarrier
        rate_per_subcarrier = tf.math.log(1.0 + SINR) / tf.math.log(2.0)
        # [B, R, ofdm, fft] - bits/s/Hz per subcarrier

        # Sum over all subcarriers and OFDM symbols
        rate_per_user = tf.reduce_sum(rate_per_subcarrier, axis=[2, 3])  # [B, R]
        avg_rate_per_user = tf.reduce_mean(rate_per_user, axis=0)  # [R]

        sum_rate = tf.reduce_sum(avg_rate_per_user)

        # Metrics
        avg_sinr = tf.reduce_mean(SINR)
        avg_sinr_db = 10.0 * tf.math.log(avg_sinr + 1e-12) / tf.math.log(10.0)

        metrics = {
            'sinr_db': avg_sinr_db,
            'signal_power': tf.reduce_mean(signal_power),
            'interference_power': tf.reduce_mean(interference_power),
            'rate_per_user': avg_rate_per_user
        }

        return sum_rate, metrics
    @tf.function
    def train_step(self, batch_size, ebno_db):
        """
        Single training step - pure sum rate optimization
        """
        batch_size_tf = tf.constant(batch_size, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            # Set topology
            self.simulator.new_topology(batch_size_tf)
            
            # Get channel
            cir = self.simulator._channel_model(
                batch_size_tf,
                self.simulator._rg.num_ofdm_symbols,
                1.0 / self.simulator._rg.ofdm_symbol_duration
            )
            h_freq = cir_to_ofdm_channel(
                self.simulator._frequencies, *cir, normalize=True
            )
            
            # Generate symbols
            b = self.simulator._binary_source([batch_size, 1, 4, self.simulator._k])
            c = self.simulator._encoder(b)
            x = self.simulator._mapper(c)
            x_rg = self.simulator._rg_mapper(x)
            
            # ✅ PRECODING (transformer uses RB grouping internally - this is fine!)
            # Precoding
            noise_power = ebnodb2no(
                ebno_db, 
                self.simulator._num_bits_per_symbol,
                self.simulator._coderate,
                self.simulator._rg
            )
            if self.simulator._precoder_type == "transformer_5d":
                x_precoded, h_eff, W = self.simulator._precoder(
                    x_rg, h_freq, training=True
                )
                
                # ✅ Use precoding matrix for accurate sum rate
                sum_rate, metrics = self.calculate_sum_rate_with_precoding_matrix(
                    h_freq, W, noise_power
                )
            else:
                x_precoded, h_eff = self.simulator._precoder(x_rg, h_freq)
                
                # For RZF, reconstruct W from effective channel or use simpler method
                sum_rate, metrics = self.calculate_sum_rate_correct(
                    h_freq, x_precoded, noise_power
                )
            
            loss = -sum_rate
        
        # Backpropagation
        trainable_vars = self.simulator._precoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Clip gradients
        gradients = [
            tf.clip_by_norm(g, 10.0) if g is not None else g 
            for g in gradients
        ]
        
        # Filter out None gradients
        valid_grads_vars = [
            (g, v) for g, v in zip(gradients, trainable_vars) 
            if g is not None
        ]
        
        if len(valid_grads_vars) > 0:
            self.optimizer.apply_gradients(valid_grads_vars)
        
        # Calculate BER for monitoring (not used in loss)
        y = self.simulator._apply_channel(x_precoded, h_freq, noise_power)
        
        if self.simulator._perfect_csi:
            h_hat = h_eff
            err_var = 0.0
        else:
            h_hat, err_var = self.simulator._ls_est(y, noise_power)
        
        x_hat, no_eff = self.simulator._lmmse_equ(y, h_hat, err_var, noise_power)
        llr = self.simulator._demapper(x_hat, no_eff)
        b_hat = self.simulator._decoder(llr)
        
        ber = compute_ber(b, b_hat)
        
        return loss, sum_rate, metrics, ber

    def train(self, num_epochs=200, batch_size=64, ebno_db_range=(5, 15)):
        """
        Training loop
        """
        print(f"{'='*80}")
        print(f"TRAINING: Pure Sum Rate Optimization")
        print(f"Epochs: {num_epochs}, Batch: {batch_size}, SNR: {ebno_db_range} dB")
        print(f"{'='*80}\n")
        
        history = {
            'losses': [],
            'sum_rates': [],
            'sinr_db': [],
            'bers': []
        }
        
        best_sum_rate = 0.0
        
        for epoch in range(num_epochs):
            # Random SNR
            ebno_db = tf.random.uniform(
                [], ebno_db_range[0], ebno_db_range[1], dtype=tf.float32
            )
            
            # Training step
            loss, sum_rate, metrics, ber = self.train_step(batch_size, ebno_db)
            
            # Store
            history['losses'].append(float(loss.numpy()))
            history['sum_rates'].append(float(sum_rate.numpy()))
            history['sinr_db'].append(float(metrics['sinr_db'].numpy()))
            history['bers'].append(float(ber.numpy()))
            
            # Track best
            if sum_rate > best_sum_rate:
                best_sum_rate = sum_rate
            
            # Log
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"SNR: {ebno_db.numpy():5.1f} dB | "
                      f"Loss: {loss.numpy():7.3f} | "
                      f"Rate: {sum_rate.numpy():6.2f} bps/Hz | "
                      f"SINR: {metrics['sinr_db'].numpy():5.1f} dB | "
                      f"BER: {ber.numpy():.2e}")
                
                if (epoch + 1) % 50 == 0:
                    print(f"  Rate/user: ", end="")
                    rates = metrics['rate_per_user'].numpy()
                    for i, r in enumerate(rates):
                        print(f"U{i+1}={r:5.2f} ", end="")
                    print()
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE!")
        print(f"Best sum rate: {best_sum_rate:.2f} bps/Hz")
        print(f"Final sum rate: {history['sum_rates'][-1]:.2f} bps/Hz")
        print(f"Final BER: {history['bers'][-1]:.2e}")
        print(f"{'='*80}\n")
        
        return history


def evaluate_and_compare(model_transformer, model_rzf, ebno_db=10.0, num_batches=20):
    """
    ✅ CORRECT evaluation using THE SAME sum rate calculation
    """
    print(f"\n{'='*80}")
    print(f"SUM RATE COMPARISON @ Eb/N0 = {ebno_db:.1f} dB")
    print(f"Using STANDARD sum rate calculation for both methods")
    print(f"{'='*80}\n")
    
    ebno_tf = tf.constant(ebno_db, dtype=tf.float32)
    batch_size = 128
    
    # Create trainer to access sum rate calculation
    trainer = SumRateTrainer(model_transformer)
    
    rates_tf = []
    rates_rzf = []
    sinr_tf = []
    sinr_rzf = []
    ber_tf = []
    ber_rzf = []
    
    for i in range(num_batches):
        # ===== SHARED CHANNEL =====
        model_transformer.new_topology(batch_size)
        
        cir = model_transformer._channel_model(
            batch_size,
            model_transformer._rg.num_ofdm_symbols,
            1.0 / model_transformer._rg.ofdm_symbol_duration
        )
        h_freq = cir_to_ofdm_channel(
            model_transformer._frequencies, *cir, normalize=True
        )
        
        # Symbols
        b = model_transformer._binary_source([batch_size, 1, 4, model_transformer._k])
        c = model_transformer._encoder(b)
        x = model_transformer._mapper(c)
        x_rg = model_transformer._rg_mapper(x)
        
        noise_power = ebnodb2no(ebno_tf, 2, 0.5, model_transformer._rg)
        
        # ===== TRANSFORMER =====
        x_prec_tf, h_eff_tf, _ = model_transformer._precoder(
            x_rg, h_freq, training=False
        )
        
        rate_tf, metrics_tf = trainer.calculate_sum_rate_standard(
            h_freq, x_prec_tf, noise_power
        )
        
        # Full chain for BER
        y_tf = model_transformer._apply_channel(x_prec_tf, h_freq, noise_power)
        x_hat_tf, no_eff_tf = model_transformer._lmmse_equ(
            y_tf, h_eff_tf, 0.0, noise_power
        )
        llr_tf = model_transformer._demapper(x_hat_tf, no_eff_tf)
        b_hat_tf = model_transformer._decoder(llr_tf)
        ber_tf_batch = compute_ber(b, b_hat_tf)
        
        rates_tf.append(float(rate_tf.numpy()))
        sinr_tf.append(float(metrics_tf['sinr_db'].numpy()))
        ber_tf.append(float(ber_tf_batch.numpy()))
        
        # ===== RZF =====
        x_prec_rzf, h_eff_rzf = model_rzf._precoder(x_rg, h_freq)
        
        rate_rzf, metrics_rzf = trainer.calculate_sum_rate_standard(
            h_freq, x_prec_rzf, noise_power
        )
        
        # Full chain for BER
        y_rzf = model_rzf._apply_channel(x_prec_rzf, h_freq, noise_power)
        x_hat_rzf, no_eff_rzf = model_rzf._lmmse_equ(
            y_rzf, h_eff_rzf, 0.0, noise_power
        )
        llr_rzf = model_rzf._demapper(x_hat_rzf, no_eff_rzf)
        b_hat_rzf = model_rzf._decoder(llr_rzf)
        ber_rzf_batch = compute_ber(b, b_hat_rzf)
        
        rates_rzf.append(float(rate_rzf.numpy()))
        sinr_rzf.append(float(metrics_rzf['sinr_db'].numpy()))
        ber_rzf.append(float(ber_rzf_batch.numpy()))
        
        if i == 0:
            print(f"Batch 1:")
            print(f"  Transformer: Rate={rate_tf:.2f} bps/Hz, "
                  f"SINR={metrics_tf['sinr_db']:.1f} dB, BER={ber_tf_batch:.2e}")
            print(f"  RZF:         Rate={rate_rzf:.2f} bps/Hz, "
                  f"SINR={metrics_rzf['sinr_db']:.1f} dB, BER={ber_rzf_batch:.2e}")
    
    # Results
    print(f"\n{'='*80}")
    print(f"RESULTS (averaged over {num_batches} batches):")
    print(f"{'='*80}")
    
    print(f"\nTransformer:")
    print(f"  Sum Rate: {np.mean(rates_tf):6.2f} ± {np.std(rates_tf):.2f} bps/Hz")
    print(f"  SINR:     {np.mean(sinr_tf):6.1f} ± {np.std(sinr_tf):.1f} dB")
    print(f"  BER:      {np.mean(ber_tf):.2e}")
    
    print(f"\nRZF:")
    print(f"  Sum Rate: {np.mean(rates_rzf):6.2f} ± {np.std(rates_rzf):.2f} bps/Hz")
    print(f"  SINR:     {np.mean(sinr_rzf):6.1f} ± {np.std(sinr_rzf):.1f} dB")
    print(f"  BER:      {np.mean(ber_rzf):.2e}")
    
    gain = np.mean(rates_tf) - np.mean(rates_rzf)
    gain_pct = 100 * gain / np.mean(rates_rzf)
    
    print(f"\nGain:")
    print(f"  Absolute: {gain:+6.2f} bps/Hz")
    print(f"  Relative: {gain_pct:+6.1f}%")
    print(f"{'='*80}\n")
    
    return {
        'transformer': np.mean(rates_tf),
        'rzf': np.mean(rates_rzf),
        'gain': gain,
        'gain_pct': gain_pct
    }