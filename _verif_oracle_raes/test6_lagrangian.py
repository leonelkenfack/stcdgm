"""
TEST 6: Arrow-Hurwicz-Uzawa dynamics for 3 simultaneous constraints.
min L_smooth(theta) = 0.5*||theta - target||^2
s.t. g_i(theta) = <a_i, theta> - b_i <= 0, i=1,2,3.
Update: theta <- theta - lr_t*(grad L + sum lambda_i grad g_i)
        lambda_i <- relu(lambda_i + lr_l * g_i).
Sweep lr_l, monitor lambda trajectory.
"""
import numpy as np
np.random.seed(1)

d = 32
target = np.random.randn(d)
A = np.random.randn(3, d) / np.sqrt(d)
b = np.array([-0.5, -0.3, -0.2])  # constraints active

def run(lr_t, lr_l, n_iter=5000):
    theta = np.zeros(d); lam = np.zeros(3)
    hist_lam = []; hist_obj = []
    for k in range(n_iter):
        g = A @ theta - b   # 3
        grad = (theta - target) + A.T @ lam
        theta = theta - lr_t * grad
        lam = np.maximum(0, lam + lr_l * g)
        hist_lam.append(lam.copy()); hist_obj.append(0.5*np.sum((theta-target)**2))
    return np.array(hist_lam), np.array(hist_obj), theta, lam

print("=== TEST 6: 3-Lagrangian Arrow-Hurwicz-Uzawa stability ===")
print(f"{'lr_t':>8} {'lr_l':>8} {'final_obj':>10} {'|lam|':>10} {'lam_osc_std':>14} {'status':>15}")
for lr_l in [5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0]:
    hist_lam, hist_obj, theta, lam = run(2e-4, lr_l)
    final_obj = hist_obj[-1]
    final_lam = np.linalg.norm(lam)
    # last 500-step std as oscillation measure
    osc = hist_lam[-500:].std(axis=0).mean()
    # diverged?
    if not np.isfinite(final_obj) or final_obj > 1e6:
        status = 'DIVERGED'
    elif osc > 0.1*np.abs(hist_lam[-500:]).mean():
        status = 'OSCILLATES'
    else:
        status = 'CONVERGED'
    print(f"{2e-4:>8.0e} {lr_l:>8.0e} {final_obj:>10.4f} {final_lam:>10.3f} {osc:>14.5f} {status:>15}")

# Find lr_l threshold by binary search
def status_of(lr_l):
    hist_lam, hist_obj, theta, lam = run(2e-4, lr_l)
    if not np.isfinite(hist_obj[-1]) or hist_obj[-1] > 1e6: return 'D'
    osc = hist_lam[-500:].std(axis=0).mean(); m = np.abs(hist_lam[-500:]).mean()+1e-9
    return 'O' if osc>0.1*m else 'C'

lo, hi = 5e-3, 10.0
for _ in range(40):
    mid = (lo+hi)/2
    s = status_of(mid)
    if s == 'C': lo = mid
    else: hi = mid
print(f"\nThreshold lr_lambda for CONVERGED with lr_theta=2e-4: ~{lo:.3f}")
