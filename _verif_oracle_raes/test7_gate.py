"""
TEST 7: Gate dynamics in Ĝ = mu + gate(c)*m(c) + g_theta(c, Z).
Synthetic: true y = mu(c) + 0.9*m(c) + noise where m is "physics approx".
Gate parameterized by sigmoid(s(c)); residual g_theta a small MLP.
Train with energy score. Track gate(c) trajectory.
Compare entropy bonus betas.
"""
import numpy as np, torch, torch.nn as nn
torch.manual_seed(2); np.random.seed(2)

def mu_fn(c):  return 0.5*c
def m_fn(c):   return torch.sin(2*c)  # known physics approx

class Model(nn.Module):
    def __init__(self, z=3, h=32):
        super().__init__()
        self.z=z
        self.gate_net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16,1))
        self.res = nn.Sequential(nn.Linear(1+z, h), nn.ReLU(), nn.Linear(h, 1))
    def gate(self, c): return torch.sigmoid(self.gate_net(c))
    def forward(self, c, M):
        B = c.shape[0]
        mu = 0.5*c
        m  = torch.sin(2*c)
        g  = self.gate(c)
        z = torch.randn(B, M, self.z); cc = c.unsqueeze(1).expand(B,M,1)
        res = self.res(torch.cat([cc, z], -1)).squeeze(-1)  # B x M
        anchor = (mu + g*m).expand(-1, M)
        return anchor + res, g

def es(y, ys, a=1.5):
    t1 = (ys - y).abs().pow(a).mean(1)
    t2 = 0.5*(ys.unsqueeze(2)-ys.unsqueeze(1)).abs().pow(a).mean((1,2))
    return (t1-t2).mean()

def train(beta_H, ep=400, N=4000):
    M_ = Model(); opt = torch.optim.Adam(M_.parameters(), lr=2e-3)
    c = np.random.uniform(-2, 2, N)
    ct = torch.tensor(c, dtype=torch.float32).unsqueeze(-1)
    y_true = 0.5*ct + 0.9*torch.sin(2*ct) + 0.2*torch.randn(N,1)
    for e in range(ep):
        idx = np.random.choice(N, 256, replace=False)
        cb = ct[idx]; yb = y_true[idx]
        ys, g = M_(cb, 16)
        loss = es(yb, ys, 1.5)
        # Entropy bonus: max H(gate) Bernoulli entropy = -[g log g + (1-g) log(1-g)]
        H = -(g*torch.log(g+1e-9) + (1-g)*torch.log(1-g+1e-9)).mean()
        loss = loss - beta_H * H
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ct_eval = torch.linspace(-2,2,11).unsqueeze(-1)
        g_eval = M_.gate(ct_eval).numpy().flatten()
    return g_eval

print("=== TEST 7: Gate dynamics ===")
print(f"True coefficient on m(c) = 0.9 (so 'correct' gate ~ 0.9 uniformly)")
print(f"{'beta_H':>10}  gate(c) at c=[-2,-1,0,1,2]")
for beta in [0.0, 1e-4, 1e-2, 1e-1]:
    g = train(beta)
    samp = g[[0, 2, 5, 8, 10]]
    print(f"{beta:>10.1e}  [{samp[0]:.3f} {samp[1]:.3f} {samp[2]:.3f} {samp[3]:.3f} {samp[4]:.3f}]  mean={g.mean():.3f}")

# Collapse experiment: m(c) is wrong (anti-correlated with truth)
class BadModel(Model):
    def forward(self, c, M):
        B = c.shape[0]
        mu = 0.5*c
        m  = -torch.sin(2*c)  # WRONG sign
        g  = self.gate(c)
        z = torch.randn(B, M, self.z); cc = c.unsqueeze(1).expand(B,M,1)
        res = self.res(torch.cat([cc, z], -1)).squeeze(-1)
        anchor = (mu + g*m).expand(-1, M)
        return anchor + res, g

def train_bad(beta_H, ep=400, N=4000):
    M_ = BadModel(); opt = torch.optim.Adam(M_.parameters(), lr=2e-3)
    c = np.random.uniform(-2, 2, N); ct = torch.tensor(c, dtype=torch.float32).unsqueeze(-1)
    y_true = 0.5*ct + 0.9*torch.sin(2*ct) + 0.2*torch.randn(N,1)
    for e in range(ep):
        idx = np.random.choice(N, 256, replace=False); cb=ct[idx]; yb=y_true[idx]
        ys, g = M_(cb, 16); loss = es(yb, ys, 1.5)
        H = -(g*torch.log(g+1e-9) + (1-g)*torch.log(1-g+1e-9)).mean()
        loss = loss - beta_H * H
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ct_eval = torch.linspace(-2,2,11).unsqueeze(-1)
        g_eval = M_.gate(ct_eval).numpy().flatten()
    return g_eval

print(f"\nWith WRONG physics m(c) (anti-correlated) — should collapse gate to 0:")
for beta in [0.0, 1e-4, 1e-2]:
    g = train_bad(beta)
    print(f"beta_H={beta:.0e}  mean gate={g.mean():.3f}  min={g.min():.3f}  max={g.max():.3f}")
