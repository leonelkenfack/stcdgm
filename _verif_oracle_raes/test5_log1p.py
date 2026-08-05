"""
TEST 5: log1p training space vs linear space.
Y = expm1(X), X ~ N(mu(c), sig(c)).
Train (a) energy in log space, (b) MSE in linear space.
Compare tail bias in linear space.
"""
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

def true_X(c):
    mu = 0.5*c; sig = 0.5 + 0.3*np.abs(c)
    return mu, sig

class G(nn.Module):
    def __init__(self, z=4, h=64):
        super().__init__()
        self.z=z
        self.net = nn.Sequential(nn.Linear(1+z,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,1))
    def forward(self, c, M):
        B=c.shape[0]; z=torch.randn(B,M,self.z); cc=c.unsqueeze(1).expand(B,M,1)
        return self.net(torch.cat([cc,z],-1)).squeeze(-1)

def es(y, ys, a=1.5):
    t1 = (ys - y).abs().pow(a).mean(1)
    t2 = 0.5 * (ys.unsqueeze(2)-ys.unsqueeze(1)).abs().pow(a).mean((1,2))
    return (t1 - t2).mean()

def train(mode, N=4000, M=16, ep=400):
    G_ = G(); opt = torch.optim.Adam(G_.parameters(), lr=2e-3)
    c = np.random.uniform(-2,2,N)
    mu,sig = true_X(c); Xtr = mu + sig*np.random.randn(N)
    Ytr = np.expm1(Xtr)
    ct = torch.tensor(c, dtype=torch.float32).unsqueeze(-1)
    if mode == 'log_energy':
        target = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(-1)
    else:
        target = torch.tensor(Ytr, dtype=torch.float32).unsqueeze(-1)
    for e in range(ep):
        idx = np.random.choice(N, 256, replace=False)
        cb = ct[idx]; yb = target[idx]
        ys = G_(cb, M)
        if mode == 'log_energy':
            loss = es(yb, ys, 1.5)
        else:
            loss = ((ys.mean(1, keepdim=True)-yb)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # Eval on c=1.5 (heavy tail region)
    c_eval = np.array([1.5])
    ct_e = torch.tensor(c_eval, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        ys = G_(ct_e, 5000).numpy().flatten()
    if mode == 'log_energy':
        y_lin = np.expm1(ys)
    else:
        y_lin = ys
    mu_t, sig_t = true_X(c_eval)
    Xt = mu_t + sig_t*np.random.randn(5000)
    y_true = np.expm1(Xt)
    return y_lin, y_true

print("=== TEST 5: log1p vs linear training space (heavy-tail eval at c=1.5) ===")
for mode in ['log_energy', 'lin_mse']:
    y_pred, y_true = train(mode)
    print(f"mode={mode:>12} | mean: pred={y_pred.mean():.3f} true={y_true.mean():.3f}"
          f" | q95: pred={np.quantile(y_pred,0.95):.3f} true={np.quantile(y_true,0.95):.3f}"
          f" | q99: pred={np.quantile(y_pred,0.99):.3f} true={np.quantile(y_true,0.99):.3f}")
    bias_mean = (y_pred.mean()-y_true.mean())/y_true.mean()*100
    bias_q99 = (np.quantile(y_pred,0.99)-np.quantile(y_true,0.99))/np.quantile(y_true,0.99)*100
    print(f"  bias mean = {bias_mean:+.1f}%, bias q99 = {bias_q99:+.1f}%")
