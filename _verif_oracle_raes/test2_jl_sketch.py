"""
TEST 2: JL sketch log-det rank promoter.
Compare gradient of log det(eps I + Cov_batch) (572x572) vs
log det(eps I + R^T R / B) with r=8 sketch.
Quantify MP noise reduction.
"""
import numpy as np
import torch

torch.manual_seed(0); np.random.seed(0)

d = 572
k_true = 5     # true latent rank
B = 64         # batch size
r_sketch = 8
eps = 1e-3

# True low-rank + noise model: X = A z + sigma * noise, A: d x k
def gen_batch(A, sigma=0.05):
    z = torch.randn(B, k_true)
    X = z @ A.T + sigma * torch.randn(B, d)
    return X  # B x d

A_true = torch.randn(d, k_true) / np.sqrt(k_true)

def full_cov_logdet(X):
    # X: B x d
    Cov = (X.T @ X) / B  # d x d, rank <= B
    M = eps * torch.eye(d) + Cov
    sign, ld = torch.linalg.slogdet(M)
    return ld

def jl_logdet(X, P):
    # P: d x r
    R = X @ P  # B x r
    M = eps * torch.eye(r_sketch) + (R.T @ R) / B
    sign, ld = torch.linalg.slogdet(M)
    return ld

# Make A_true require grad (proxy for "structural parameters")
A = A_true.clone().requires_grad_(True)

# Compute gradient norms across many batches to estimate noise
n_trials = 30
grad_full_norms = []
grad_jl_norms = []
ld_full_vals = []
ld_jl_vals = []

for t in range(n_trials):
    P = torch.randn(d, r_sketch) / np.sqrt(r_sketch)  # fresh JL each batch
    X = gen_batch(A)
    # Full
    A.grad = None
    # Need X to depend on A for grad: rebuild
    z = torch.randn(B, k_true)
    noise = 0.05 * torch.randn(B, d)
    X_full = z @ A.T + noise
    ld_f = full_cov_logdet(X_full)
    ld_f.backward()
    gf = A.grad.detach().norm().item()
    grad_full_norms.append(gf); ld_full_vals.append(ld_f.item())

    A.grad = None
    X_jl = z @ A.T + noise  # same noise to compare
    ld_j = jl_logdet(X_jl, P)
    ld_j.backward()
    gj = A.grad.detach().norm().item()
    grad_jl_norms.append(gj); ld_jl_vals.append(ld_j.item())

print("=== TEST 2: JL sketch rank promoter ===")
print(f"d={d}, k_true={k_true}, B={B}, r_sketch={r_sketch}, eps={eps}")
print(f"Full   log det: mean={np.mean(ld_full_vals):>10.3f}, std={np.std(ld_full_vals):>8.3f}, |grad A| mean={np.mean(grad_full_norms):>10.3f}, std={np.std(grad_full_norms):>8.3f}")
print(f"JL r=8 log det: mean={np.mean(ld_jl_vals):>10.3f}, std={np.std(ld_jl_vals):>8.3f}, |grad A| mean={np.mean(grad_jl_norms):>10.3f}, std={np.std(grad_jl_norms):>8.3f}")
# Noise ratio: coeff of variation
cv_full = np.std(grad_full_norms) / (np.mean(grad_full_norms)+1e-12)
cv_jl   = np.std(grad_jl_norms)   / (np.mean(grad_jl_norms)+1e-12)
print(f"Gradient CV (noise/signal): full={cv_full:.3f} vs JL={cv_jl:.3f}, ratio={cv_full/cv_jl:.2f}x")
# Memory: full is d*d floats vs JL is d*r + B*r
mem_full = d*d*4 / 1024
mem_jl = (d*r_sketch + B*r_sketch)*4 / 1024
print(f"Memory: full {mem_full:.0f} KB vs JL {mem_jl:.1f} KB ({mem_full/mem_jl:.0f}x reduction)")
