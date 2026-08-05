"""
TEST 4: Q-vector inversion accuracy at 22x26.
Generate synthetic baroclinic wave on 22x26 grid (8 levels).
Compute Q-vector forcing, solve QG-omega elliptic equation by finite differences,
compare omega_bal to true omega. Report R^2.
"""
import numpy as np
from scipy.sparse import diags, eye as speye, kron, lil_matrix
from scipy.sparse.linalg import spsolve

ny, nx = 22, 26
nz = 8
Lx = 5e6; Ly = 4e6   # ~mid-latitude domain (m)
dx = Lx/(nx-1); dy = Ly/(ny-1)

f0 = 1e-4   # Coriolis (1/s)
beta = 1.6e-11
R = 287.0
g = 9.81
# Sigma static stability (1/(kg/m3)) — approximate
sigma_static = 2e-6

# Pressure levels (Pa)
plev = np.linspace(1000e2, 200e2, nz)
dp = plev[1] - plev[0]   # negative

# Synthetic baroclinic wave: streamfunction psi = A*sin(2pi x/Lx)*sin(pi y/Ly) * (p - p_mid)
y = np.linspace(0, Ly, ny); x = np.linspace(0, Lx, nx)
X, Y = np.meshgrid(x, y)
p_mid = (plev[0]+plev[-1])/2
A_amp = 5e6

psi = np.zeros((nz, ny, nx))
for k in range(nz):
    psi[k] = A_amp * np.sin(2*np.pi*X/Lx) * np.sin(np.pi*Y/Ly) * ((plev[k]-p_mid)/(plev[0]-p_mid))

# Geostrophic winds u=-d psi/dy, v= d psi/dx
def ddy(f):
    out = np.zeros_like(f); out[1:-1] = (f[2:]-f[:-2])/(2*dy); out[0]=(f[1]-f[0])/dy; out[-1]=(f[-1]-f[-2])/dy; return out
def ddx(f):
    out = np.zeros_like(f); out[:,1:-1] = (f[:,2:]-f[:,:-2])/(2*dx); out[:,0]=(f[:,1]-f[:,0])/dx; out[:,-1]=(f[:,-1]-f[:,-2])/dx; return out

u = np.zeros_like(psi); v = np.zeros_like(psi)
T = np.zeros_like(psi)
for k in range(nz):
    u[k] = -ddy(psi[k]); v[k] = ddx(psi[k])
# Temperature via hypsometric: dphi/dp = -RT/p, with phi=f0*psi → T = -(p/R)*d(f0*psi)/dp
# d psi /dp:
dpsi_dp = np.zeros_like(psi)
for k in range(1, nz-1):
    dpsi_dp[k] = (psi[k+1]-psi[k-1])/(plev[k+1]-plev[k-1])
dpsi_dp[0] = (psi[1]-psi[0])/(plev[1]-plev[0])
dpsi_dp[-1] = (psi[-1]-psi[-2])/(plev[-1]-plev[-2])
for k in range(nz):
    T[k] = -(plev[k]/R) * f0 * dpsi_dp[k]

# True omega from QG-omega: just use synthetic forcing structure
# For verification, we construct "true" omega as having the same horizontal+vertical pattern but with amplitude consistent with the elliptic solution.
# Use direct method: pose true omega = cos(2 pi x/Lx)*sin(pi y/Ly)*sin(pi*(p - p_top)/(p_bot-p_top))
omega_true = np.zeros((nz, ny, nx))
for k in range(nz):
    omega_true[k] = 0.5 * np.cos(2*np.pi*X/Lx)*np.sin(np.pi*Y/Ly)*np.sin(np.pi*(plev[k]-plev[-1])/(plev[0]-plev[-1]))

# Compute Q-vector
# Q1 = -(R/sigma p)*[ d u/d x * d T/d x + d v/d x * d T/d y ]
# Q2 = -(R/sigma p)*[ d u/d y * d T/d x + d v/d y * d T/d y ]
# Forcing F = -2 div Q  +  beta term (skip beta)
Q1 = np.zeros_like(psi); Q2 = np.zeros_like(psi)
for k in range(nz):
    dudx = ddx(u[k]); dvdx = ddx(v[k]); dudy = ddy(u[k]); dvdy = ddy(v[k])
    dTdx = ddx(T[k]); dTdy = ddy(T[k])
    Q1[k] = -(R/(sigma_static*plev[k])) * (dudx*dTdx + dvdx*dTdy)
    Q2[k] = -(R/(sigma_static*plev[k])) * (dudy*dTdx + dvdy*dTdy)
divQ = np.zeros_like(psi)
for k in range(nz):
    divQ[k] = ddx(Q1[k]) + ddy(Q2[k])
F = -2.0 * divQ

# Elliptic operator: sigma * grad_h^2 omega + f0^2 d^2 omega/dp^2 = F
# Solve per-level by 2D Poisson with vertical coupling -- here, simplified: solve level-by-level (ignoring vertical coupling)
# This is approximate but standard for diagnostic checks.
def laplacian_matrix(ny, nx, dy, dx):
    Ix = speye(nx); Iy = speye(ny)
    Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / dx**2
    Dyy = diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / dy**2
    return kron(Iy, Dxx) + kron(Dyy, Ix)
L2 = laplacian_matrix(ny, nx, dy, dx).tolil()
# Dirichlet boundary: omega=0 on boundary
N = ny*nx
omega_bal = np.zeros_like(psi)
for k in range(nz):
    L = L2.copy() * sigma_static
    # Boundary
    for i in range(ny):
        for j in range(nx):
            idx = i*nx + j
            if i==0 or i==ny-1 or j==0 or j==nx-1:
                L[idx,:] = 0; L[idx,idx] = 1
    rhs = F[k].flatten().copy()
    # zero boundary forcing
    for i in range(ny):
        for j in range(nx):
            if i==0 or i==ny-1 or j==0 or j==nx-1:
                rhs[i*nx+j] = 0
    w_sol = spsolve(L.tocsr(), rhs)
    omega_bal[k] = w_sol.reshape(ny, nx)

# Forward consistency check: set true_omega from the elliptic solution itself (i.e. produce a (u,v,T) such that the QG-omega applied to it returns omega_bal close to a chosen omega field).
# Practical approach: define omega_true as the result of applying the inverse elliptic operator to the SAME forcing F we computed — that yields the exact "balanced omega" the SCM would produce, modulo discretization.
# Then we check how close the level-decoupled Poisson solve is to a 3D solve, including vertical coupling via f0^2 d2/dp2.

# Build full 3D elliptic operator: sigma*nabla_h^2 + f0^2 d2/dp2
Iz = speye(nz)
# vertical 2nd derivative wrt p (uniform plev assumed)
DPP = diags([1, -2, 1], [-1, 0, 1], shape=(nz, nz)) / (dp*dp)
L_h = laplacian_matrix(ny, nx, dy, dx) * sigma_static
L_3d = kron(Iz, L_h) + kron(DPP*(f0*f0), speye(ny*nx))
# Boundary conditions: omega=0 on lateral + top/bot
L_3d = L_3d.tolil()
def gidx(k,i,j): return k*ny*nx + i*nx + j
for k in range(nz):
    for i in range(ny):
        for j in range(nx):
            if i==0 or i==ny-1 or j==0 or j==nx-1 or k==0 or k==nz-1:
                idx = gidx(k,i,j)
                L_3d[idx,:] = 0; L_3d[idx,idx] = 1

rhs3 = F.flatten().copy()
for k in range(nz):
    for i in range(ny):
        for j in range(nx):
            if i==0 or i==ny-1 or j==0 or j==nx-1 or k==0 or k==nz-1:
                rhs3[gidx(k,i,j)] = 0
omega_3d = spsolve(L_3d.tocsr(), rhs3).reshape(nz, ny, nx)

# Compare 2D level-decoupled vs 3D
mask_interior = np.zeros_like(omega_true, dtype=bool)
mask_interior[1:-1, 2:-2, 2:-2] = True

# Reference omega: 3D solve (truly balanced for the QG-omega operator)
a2 = omega_bal[mask_interior]; a3 = omega_3d[mask_interior]
# Pattern correlation between 2D and 3D
corr_2D_3D = np.corrcoef(a2, a3)[0,1] if a2.std()>0 and a3.std()>0 else 0.0

# To approximate "truth": run the 3D solve with twice-finer grid then sub-sample
ny2, nx2 = 43, 51
dx2 = Lx/(nx2-1); dy2 = Ly/(ny2-1)
y2 = np.linspace(0, Ly, ny2); x2 = np.linspace(0, Lx, nx2)
X2, Y2 = np.meshgrid(x2, y2)
psi2 = np.zeros((nz, ny2, nx2))
for k in range(nz):
    psi2[k] = A_amp * np.sin(2*np.pi*X2/Lx) * np.sin(np.pi*Y2/Ly) * ((plev[k]-p_mid)/(plev[0]-p_mid))
def ddy2(f): out=np.zeros_like(f); out[1:-1]=(f[2:]-f[:-2])/(2*dy2); out[0]=(f[1]-f[0])/dy2; out[-1]=(f[-1]-f[-2])/dy2; return out
def ddx2(f): out=np.zeros_like(f); out[:,1:-1]=(f[:,2:]-f[:,:-2])/(2*dx2); out[:,0]=(f[:,1]-f[:,0])/dx2; out[:,-1]=(f[:,-1]-f[:,-2])/dx2; return out
u2 = np.zeros_like(psi2); v2 = np.zeros_like(psi2); T2 = np.zeros_like(psi2)
dpsi_dp2 = np.zeros_like(psi2)
for k in range(nz):
    u2[k] = -ddy2(psi2[k]); v2[k] = ddx2(psi2[k])
for k in range(1, nz-1):
    dpsi_dp2[k] = (psi2[k+1]-psi2[k-1])/(plev[k+1]-plev[k-1])
dpsi_dp2[0]=(psi2[1]-psi2[0])/(plev[1]-plev[0]); dpsi_dp2[-1]=(psi2[-1]-psi2[-2])/(plev[-1]-plev[-2])
for k in range(nz):
    T2[k] = -(plev[k]/R)*f0*dpsi_dp2[k]
Q1f=np.zeros_like(psi2); Q2f=np.zeros_like(psi2)
for k in range(nz):
    du=ddx2(u2[k]); dv=ddx2(v2[k]); duy=ddy2(u2[k]); dvy=ddy2(v2[k])
    dTx=ddx2(T2[k]); dTy=ddy2(T2[k])
    Q1f[k]=-(R/(sigma_static*plev[k]))*(du*dTx + dv*dTy)
    Q2f[k]=-(R/(sigma_static*plev[k]))*(duy*dTx + dvy*dTy)
divQ2=np.zeros_like(psi2)
for k in range(nz):
    divQ2[k] = ddx2(Q1f[k]) + ddy2(Q2f[k])
Ff = -2*divQ2
L_h2 = laplacian_matrix(ny2, nx2, dy2, dx2)*sigma_static
L_3d2 = kron(Iz, L_h2) + kron(DPP*(f0*f0), speye(ny2*nx2))
L_3d2 = L_3d2.tolil()
def gidx2(k,i,j): return k*ny2*nx2 + i*nx2 + j
for k in range(nz):
    for i in range(ny2):
        for j in range(nx2):
            if i==0 or i==ny2-1 or j==0 or j==nx2-1 or k==0 or k==nz-1:
                idx=gidx2(k,i,j); L_3d2[idx,:]=0; L_3d2[idx,idx]=1
rhsf = Ff.flatten().copy()
for k in range(nz):
    for i in range(ny2):
        for j in range(nx2):
            if i==0 or i==ny2-1 or j==0 or j==nx2-1 or k==0 or k==nz-1:
                rhsf[gidx2(k,i,j)] = 0
omega_fine = spsolve(L_3d2.tocsr(), rhsf).reshape(nz, ny2, nx2)
# Sub-sample to 22x26 (take every other point starting from index 0)
omega_truth = omega_fine[:, ::2, ::2][:, :ny, :nx]

t2 = omega_bal[mask_interior]; tt = omega_truth[mask_interior]; t3 = omega_3d[mask_interior]
def R2_fn(a, b):
    ss_res = np.sum((a-b)**2); ss_tot = np.sum((b-b.mean())**2)
    return 1 - ss_res/ss_tot if ss_tot>0 else float('nan')
def stats(name, a, b):
    if a.std()<1e-20 or b.std()<1e-20:
        corr = float('nan')
    else:
        corr = np.corrcoef(a,b)[0,1]
    r2 = R2_fn(a, b)
    print(f"{name:>30}: corr={corr:.3f}  R2={r2:.3f}  std_ratio={a.std()/(b.std()+1e-20):.3f}")
print("=== TEST 4: Q-vector inversion at 22x26 ===")
print(f"Grid {ny}x{nx}, dx={dx/1e3:.0f} km, dy={dy/1e3:.0f} km, {nz} levels")
print(f"Reference: fine-grid 43x51 3D solve, sub-sampled to 22x26.")
stats("2D-decoupled vs truth(fine 3D)", t2, tt)
stats("3D coarse(22x26)  vs truth(fine 3D)", t3, tt)
stats("2D-decoupled vs 3D-coarse", t2, t3)
print(f"omega scales: 2D std={t2.std():.3e}, 3D std={t3.std():.3e}, truth std={tt.std():.3e}")
