import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from main_sionna_simple import MU_MIMO_System, cir_to_ofdm_channel

def sanity_check_xrg():
    print("==================================================")
    print("      RUNNING SIONNA TENSOR SHAPE SANITY CHECK    ")
    print("==================================================")

    # 1. Instantiate the Full System
    # We use the full system because it contains the Encoder/Mapper/RG_Mapper
    BATCH_SIZE = 2
    NUM_RX = 4
    NUM_TX = 8
    
    system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type="transformer")
    
    # 2. Manually Trigger Data Generation (Mimic the 'call' function)
    print("\n[Step 1] Generating Data (x_rg)...")
    
    # A. Generate Bits
    b = system.binary_source([BATCH_SIZE, 1, NUM_RX, int(system.rg.num_data_symbols)])
    print(f"Bits Shape:       {b.shape} (Batch, 1, Users, Bits)")
    
    # B. Encode & Map
    c = system.encoder(b)
    x = system.mapper(c)
    
    # C. Map to Resource Grid (THIS IS THE CRITICAL PART)
    x_rg = system.rg_mapper(x)
    
    print(f"x_rg Shape:       {x_rg.shape}")
    
    # 3. VERIFICATION LOGIC
    # Expected: [Batch, Num_Tx (1), Streams (Users), Symbols, Freq]
    expected_shape = (BATCH_SIZE, 1, NUM_RX, 14, 72)
    
    if x_rg.shape == expected_shape:
        print("✅ SUCCESS: x_rg is exactly the 5D shape Sionna expects.")
    else:
        print(f"❌ WARNING: x_rg shape mismatch. Expected {expected_shape}")
        
        # Diagnostic: If it's 4D [B, K, S, F], we know why
        if len(x_rg.shape) == 4:
            print("   -> It is 4D. You likely need to expand_dims at axis 1.")

    # 4. Test Interaction with Transformer Precoder
    print("\n[Step 2] Testing Precoder Input...")
    
    # Generate a dummy channel
    system.new_topology(BATCH_SIZE)
    cir = system.channel_model(BATCH_SIZE, system.rg.num_ofdm_symbols, 1.0 / system.rg.ofdm_symbol_duration)
    h_freq = cir_to_ofdm_channel(system.frequencies, *cir, normalize=True)
    
    try:
        # Pass the x_rg we just generated into the precoder
        # This checks if your squeeze/reshape logic inside the model works
        x_precoded, h_eff, W = system.precoder(h_freq, x_rg=x_rg, training=False)
        
        print(f"✅ SUCCESS: Precoder accepted x_rg.")
        print(f"   x_precoded Out: {x_precoded.shape} (Should be [B, 1, M, S, F] or [B, M, 1, S, F])")
        print(f"   W Out:          {W.shape}")
        
    except Exception as e:
        print(f"❌ CRITICAL FAILURE in Precoder:\n{e}")

if __name__ == "__main__":
    sanity_check_xrg()