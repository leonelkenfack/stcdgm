"""
TEST 8: Energy score alpha=1.5 heavy tail.
Y ~ GPD(xi) for xi in {0.05, 0.1, 0.2, 0.3, 0.4}.
Train tiny G(z) (unconditional) to match Y via energy score.
Track gradient variance; find xi threshold where training diverges.
"""
import numpy as np, torch, torch.nn as nn
torch.manual_seed(3); np.random.seed(3)

def gpd_sample(xi, n, sigma=1.0):
    U = np.random.uniform(0,1,n)
    if abs(xi) < 1e-8:
        return -sigma*np.log(1-U)
    return sigma/xi * ((1-U)**(-xi) - 1)

class Gen(nn.Module):
    def __init__(self, z=2, h=32):
        super().__init__()
        self.z=z
        self.net = nn.Sequential(nn.Linear(z,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,1))
    def forward(self, B, M):
        z = torch.randn(B*M, self.z)
        return self.net(z).view(B, M)

def es(y, ys, a=1.5):
    t1 = (ys - y).abs().pow(a).mean(1)
    t2 = 0.5*(ys.unsqueeze(2)-ys.unsqueeze(1)).abs().pow(a).mean((1,2))
    return (t1-t2).mean()

def run(xi, ep=300, M=16, B=128, alpha=1.5):
    G_ = Gen(); opt = torch.optim.Adam(G_.parameters(), lr=2e-3)
    Y = gpd_sample(xi, 10000)
    yt = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)
    grad_norms = []
    diverged = False
    for e in range(ep):
        idx = np.random.choice(10000, B, replace=False)
        yb = yt[idx]
        ys = G_(B, M)
        loss = es(yb, ys, alpha)
        if not torch.isfinite(loss): diverged=True; break
        opt.zero_grad(); loss.backward()
        gn = 0.0
        for p in G_.parameters():
            if p.grad is not None: gn += (p.grad**2).sum().item()
        gn = np.sqrt(gn)
        if not np.isfinite(gn) or gn > 1e6: diverged=True; break
        grad_norms.append(gn)
        opt.step()
    # tail metric
    with torch.no_grad():
        ys = G_(1, 5000).numpy().flatten()
    return np.array(grad_norms), diverged, ys, Y

print("=== TEST 8: Energy score alpha=1.5 vs GPD heavy tail ===")
print(f"{'xi':>6} {'alpha':>6} {'mean |grad|':>12} {'std |grad|':>12} {'CV':>8} {'diverged':>10} {'q95_pred':>10} {'q95_true':>10}")
for xi in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
    for a in [1.5, 1.0, 2.0]:
        gn, div, ys, Y = run(xi, alpha=a)
        if len(gn)==0:
            print(f"{xi:>6.2f} {a:>6.2f}  ALL DIVERGED at start"); continue
        cv = gn.std()/(gn.mean()+1e-12)
        q95p = np.quantile(ys, 0.95); q95t = np.quantile(Y, 0.95)
        print(f"{xi:>6.2f} {a:>6.2f} {gn.mean():>12.3f} {gn.std():>12.3f} {cv:>8.3f} {str(div):>10} {q95p:>10.3f} {q95t:>10.3f}")
