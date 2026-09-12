import torch

IN_DIM  = 128
OUT_DIM = 64

# Même input que C++
x = torch.tensor([(i % 8) / 8.0 - 0.5 for i in range(IN_DIM)], dtype=torch.float32)

# Même weights que C++
W = torch.tensor([[(( i + j) % 4) / 16.0 - 0.1 
                   for j in range(IN_DIM)] 
                   for i in range(OUT_DIM)], dtype=torch.float32)

b = torch.full((OUT_DIM,), 0.01)

y = W @ x + b
y_relu = torch.relu(y)

print("=== PyTorch output (first 8) ===")
for i in range(8):
    print(f"y[{i}] = {y[i].item():.2f}")

print("\n=== After ReLU (first 8) ===")
for i in range(8):
    print(f"y_relu[{i}] = {y_relu[i].item():.2f}")