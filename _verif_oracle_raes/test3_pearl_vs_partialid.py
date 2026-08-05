"""
TEST 3: Pearl back-door regression vs Manski / sensitivity / control-function on synthetic SCM.
SCM:
  U(t) = 0.97 U(t-1) + 0.3 sin(2 pi t/365) + 0.24 eta_U,
  w_500 = -0.4 X1 + 0.5 U + 0.6 eps_w,
  q_500 =  0.3 X2 + 0.4 U + 0.7 eps_q,
  P = softplus(-1.2 w + 0.9 q + 0.8 U + 0.5 X3 + eps_P),
where X1,X2,X3 ~ N(0,1) iid observed, U hidden, eps's N(0,1).
N = 5e4.
True do-effect at (w,q)=(-1,+1): E[P | do(w=-1, q=+1)] computed by Monte Carlo from SCM.
"""
import numpy as np
from scipy.special import logit

rng = np.random.default_rng(42)
N = 50_000

# Hidden confounder AR(1)
def sim_U(N):
    U = np.zeros(N)
    t = np.arange(N)
    for i in range(1, N):
        U[i] = 0.97*U[i-1] + 0.3*np.sin(2*np.pi*t[i]/365) + 0.24*rng.standard_normal()
    return U

def softplus(x): return np.log1p(np.exp(x))

U = sim_U(N)
X1 = rng.standard_normal(N); X2 = rng.standard_normal(N); X3 = rng.standard_normal(N)
ew = rng.standard_normal(N); eq = rng.standard_normal(N); eP = rng.standard_normal(N)
w = -0.4*X1 + 0.5*U + 0.6*ew
q =  0.3*X2 + 0.4*U + 0.7*eq
P = softplus(-1.2*w + 0.9*q + 0.8*U + 0.5*X3 + eP)

# === Truth: E[P | do(w=-1, q=+1)] ===
# do = override w,q, integrate over U,X3,eP
M = 100_000
U_mc = sim_U(M)
X3_mc = rng.standard_normal(M); eP_mc = rng.standard_normal(M)
true_do = softplus(-1.2*(-1) + 0.9*(1) + 0.8*U_mc + 0.5*X3_mc + eP_mc).mean()
print(f"True E[P|do(w=-1,q=1)] = {true_do:.3f}")

# === (1) Pearl back-door regression: P ~ w + q + X1 + X2 + X3 (linear OLS, then softplus latent estimate) ===
# Here we use the linear-in-latent model: fit P on (w,q,X1,X2,X3) by OLS to estimate the dose-response.
from numpy.linalg import lstsq
Xreg = np.column_stack([np.ones(N), w, q, X1, X2, X3])
beta, *_ = lstsq(Xreg, P, rcond=None)
b0, bw, bq, b1, b2, b3 = beta
# Pearl point prediction with X1,X2,X3 ~ N(0,1) marginally: E_X[ b0 + bw*(-1) + bq*1 + sum b_i*0 ]
pearl_est = b0 + bw*(-1) + bq*1 + b1*0 + b2*0 + b3*0
pearl_bias = (pearl_est - true_do)/true_do * 100
print(f"Pearl back-door OLS estimate = {pearl_est:.3f}   bias = {pearl_bias:+.1f}%")

# === (2) Manski bounds: assume Y = P bounded in [P_min, P_max] observed support ===
P_min, P_max = P.min(), P.max()
# For w near -1, q near +1, find observed conditional expectation, and "no-assumption" bounds
mask = (np.abs(w-(-1))<0.3) & (np.abs(q-1)<0.3)
p_obs = mask.mean()  # propensity of cell
EP_in = P[mask].mean() if mask.sum()>0 else 0.0
L_manski = p_obs * EP_in + (1-p_obs)*P_min
U_manski = p_obs * EP_in + (1-p_obs)*P_max
print(f"Manski bounds: [{L_manski:.3f}, {U_manski:.3f}], width = {U_manski - L_manski:.3f}")

# === (3) Sensitivity Γ (Rosenbaum-style) ===
# Slow process Γ_slow = exp(0.8*sd(U)/sd(eP)) for unmeasured U
sdU = U.std(); sd_eP = 1.0
Gamma_slow = np.exp(0.8*sdU/sd_eP * 0.15)  # approx, treating residual confounding
# Width via Theorem 1: width ≈ 2 * log(Γ) * sd(P)
sdP = P.std()
width_thm1 = 2*np.log(Gamma_slow) * sdP
print(f"Sensitivity Gamma_slow ~ {Gamma_slow:.3f}, width (Thm 1) ~ {width_thm1:.3f}")

# === (4) Control-function: regress w on X1 (instrument), residual = proxy for U; include in P regression ===
# Stage 1: w = α0 + α1 X1 + v_w ; q = β0 + β1 X2 + v_q
aw = lstsq(np.column_stack([np.ones(N), X1]), w, rcond=None)[0]
vw = w - (aw[0] + aw[1]*X1)
aq = lstsq(np.column_stack([np.ones(N), X2]), q, rcond=None)[0]
vq = q - (aq[0] + aq[1]*X2)
# Stage 2: include vw, vq as control functions
Xcf = np.column_stack([np.ones(N), w, q, X3, vw, vq])
gamma, *_ = lstsq(Xcf, P, rcond=None)
g0, gw, gq, gx3, gvw, gvq = gamma
cf_est = g0 + gw*(-1) + gq*1 + gx3*0 + gvw*0 + gvq*0  # set residuals to 0 (do-operator)
cf_bias = (cf_est - true_do)/true_do * 100
print(f"Control-function estimate = {cf_est:.3f}   bias = {cf_bias:+.1f}%")

# === (5) ORACLE-RAES PI-constrained (proxy): add U as observable (since A3 budget partially deconfounds) ===
# Best-case: include U as control (simulates moisture-budget residual as proxy for slow U)
XPI = np.column_stack([np.ones(N), w, q, U, X3])
delta, *_ = lstsq(XPI, P, rcond=None)
d0, dw, dq, du, dx3 = delta
pi_est = d0 + dw*(-1) + dq*1 + du*U.mean() + dx3*0
pi_bias = (pi_est - true_do)/true_do * 100
print(f"PI-constrained (oracle U via budget) = {pi_est:.3f}   bias = {pi_bias:+.1f}%")
