"""
TEST 1: Energy score vs MSE — Property P1 verification.
True conditional: p(y|c) = N(sin(c), 0.3 + 0.2*|c|).
Train small generator G(c,Z) with (a) MSE, (b) Energy score alpha=1.5.
Compare Wasserstein-1 to truth.
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

device = 'cpu'

def true_sample(c, n_per_c=1):
    mu = np.sin(c)
    sig = 0.3 + 0.2 * np.abs(c)
    return mu + sig * np.random.randn(len(c), n_per_c)

class Gen(nn.Module):
    def __init__(self, z_dim=4, hidden=64):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(1 + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, c, M):
        # c: [B,1], output M samples per c
        B = c.shape[0]
        z = torch.randn(B, M, self.z_dim, device=c.device)
        cc = c.unsqueeze(1).expand(B, M, 1)
        return self.net(torch.cat([cc, z], dim=-1)).squeeze(-1)  # [B,M]

def energy_score(y, ysim, alpha=1.5):
    # y: [B,1] true; ysim: [B,M] sim
    B, M = ysim.shape
    # term1: E |Y - y|^alpha
    t1 = (ysim - y).abs().pow(alpha).mean(dim=1)
    # term2: 0.5 * E |Y - Y'|^alpha
    diffs = (ysim.unsqueeze(2) - ysim.unsqueeze(1)).abs().pow(alpha)
    t2 = 0.5 * diffs.mean(dim=(1,2))
    return (t1 - t2).mean()

def w1_emp(a, b):
    a = np.sort(a); b = np.sort(b)
    n = min(len(a), len(b))
    qa = np.quantile(a, np.linspace(0,1,200))
    qb = np.quantile(b, np.linspace(0,1,200))
    return np.mean(np.abs(qa - qb))

def train_and_eval(loss_type, N_train, M=32, epochs=1200):
    G = Gen().to(device)
    opt = torch.optim.Adam(G.parameters(), lr=2e-3)
    # training data
    c_tr = np.random.uniform(-2, 2, N_train)
    y_tr = true_sample(c_tr, 1)[:,0]
    c_tr_t = torch.tensor(c_tr, dtype=torch.float32).unsqueeze(-1)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1)
    bs = min(256, N_train)
    for ep in range(epochs):
        idx = np.random.choice(N_train, bs, replace=False)
        cb = c_tr_t[idx]; yb = y_tr_t[idx]
        ysim = G(cb, M)  # [bs,M]
        if loss_type == 'mse':
            loss = ((ysim.mean(dim=1, keepdim=True) - yb)**2).mean()
        else:
            loss = energy_score(yb, ysim, alpha=1.5)
        opt.zero_grad(); loss.backward(); opt.step()
    # Eval: at fixed c values, compare empirical
    c_eval = np.array([-1.5, -0.5, 0.5, 1.5])
    results = []
    for cv in c_eval:
        ct = torch.tensor([[cv]], dtype=torch.float32)
        with torch.no_grad():
            ysim = G(ct, 1000).numpy().flatten()
        ytrue = true_sample(np.array([cv]), 1000).flatten()
        w1 = w1_emp(ysim, ytrue)
        # tail check
        q95_sim = np.quantile(ysim, 0.95); q95_true = np.quantile(ytrue, 0.95)
        std_sim = ysim.std(); std_true = ytrue.std()
        results.append((cv, w1, std_sim, std_true, q95_sim, q95_true))
    return results

print("=== TEST 1: Energy score vs MSE ===")
print(f"{'N':>6} {'loss':>10} {'mean W1':>10} {'mean std_sim':>14} {'mean std_true':>15} {'mean q95_sim':>14} {'mean q95_true':>15}")
for N in [500, 2000, 8000]:
    for lt in ['mse','energy']:
        res = train_and_eval(lt, N)
        w1 = np.mean([r[1] for r in res])
        ss = np.mean([r[2] for r in res])
        st = np.mean([r[3] for r in res])
        qs = np.mean([r[4] for r in res])
        qt = np.mean([r[5] for r in res])
        print(f"{N:>6} {lt:>10} {w1:>10.4f} {ss:>14.4f} {st:>15.4f} {qs:>14.4f} {qt:>15.4f}")
