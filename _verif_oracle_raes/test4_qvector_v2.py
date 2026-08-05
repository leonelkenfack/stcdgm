"""
TEST 4 v2: Forward-consistency Q-vector test.
Construct omega_truth on a fine 88x104 grid via 3D QG-omega solve.
Sub-sample inputs to 22x26 and 44x52, re-solve, compare to coarsened truth.
Report R2 vs resolution.
"""
import numpy as np
from scipy.sparse import diags, eye as speye, kron
from scipy.sparse.linalg import spsolve

def build_omega(ny, nx, nz=8, Lx=5e6, Ly=4e6):
    dx = Lx/(nx-1); dy = Ly/(ny-1)
    f0 = 1e-4; R = 287.0; sigma = 2e-6
    plev = np.linspace(1000e2, 200e2, nz); dp = plev[1]-plev[0]
    y = np.linspace(0, Ly, ny); x = np.linspace(0, Lx, nx)
    X, Y = np.meshgrid(x, y)
    p_mid = (plev[0]+plev[-1])/2
    A_amp = 5e6
    psi = np.zeros((nz, ny, nx))
    for k in range(nz):
        psi[k] = A_amp * np.sin(2*np.pi*X/Lx) * np.sin(np.pi*Y/Ly) * ((plev[k]-p_mid)/(plev[0]-p_mid))
    def ddy(f): out=np.zeros_like(f); out[1:-1]=(f[2:]-f[:-2])/(2*dy); out[0]=(f[1]-f[0])/dy; out[-1]=(f[-1]-f[-2])/dy; return out
    def ddx(f): out=np.zeros_like(f); out[:,1:-1]=(f[:,2:]-f[:,:-2])/(2*dx); out[:,0]=(f[:,1]-f[:,0])/dx; out[:,-1]=(f[:,-1]-f[:,-2])/dx; return out
    u = np.zeros_like(psi); v = np.zeros_like(psi); T = np.zeros_like(psi); dpsi_dp=np.zeros_like(psi)
    for k in range(nz):
        u[k] = -ddy(psi[k]); v[k] = ddx(psi[k])
    for k in range(1, nz-1):
        dpsi_dp[k] = (psi[k+1]-psi[k-1])/(plev[k+1]-plev[k-1])
    dpsi_dp[0]=(psi[1]-psi[0])/(plev[1]-plev[0]); dpsi_dp[-1]=(psi[-1]-psi[-2])/(plev[-1]-plev[-2])
    for k in range(nz):
        T[k] = -(plev[k]/R)*f0*dpsi_dp[k]
    Q1=np.zeros_like(psi); Q2=np.zeros_like(psi)
    for k in range(nz):
        du=ddx(u[k]); dv=ddx(v[k]); duy=ddy(u[k]); dvy=ddy(v[k])
        dTx=ddx(T[k]); dTy=ddy(T[k])
        Q1[k] = -(R/(sigma*plev[k]))*(du*dTx + dv*dTy)
        Q2[k] = -(R/(sigma*plev[k]))*(duy*dTx + dvy*dTy)
    divQ = np.zeros_like(psi)
    for k in range(nz):
        divQ[k] = ddx(Q1[k]) + ddy(Q2[k])
    F = -2*divQ
    # 3D elliptic operator
    def lap2d(ny, nx, dy, dx):
        Ix = speye(nx); Iy = speye(ny)
        Dxx = diags([1,-2,1],[-1,0,1],shape=(nx,nx))/dx**2
        Dyy = diags([1,-2,1],[-1,0,1],shape=(ny,ny))/dy**2
        return kron(Iy,Dxx) + kron(Dyy,Ix)
    Iz = speye(nz)
    DPP = diags([1,-2,1],[-1,0,1],shape=(nz,nz))/(dp*dp)
    L_h = lap2d(ny,nx,dy,dx)*sigma
    L = kron(Iz, L_h) + kron(DPP*(f0*f0), speye(ny*nx))
    L = L.tolil()
    def gidx(k,i,j): return k*ny*nx + i*nx + j
    for k in range(nz):
        for i in range(ny):
            for j in range(nx):
                if i==0 or i==ny-1 or j==0 or j==nx-1 or k==0 or k==nz-1:
                    idx=gidx(k,i,j); L[idx,:]=0; L[idx,idx]=1
    rhs = F.flatten().copy()
    for k in range(nz):
        for i in range(ny):
            for j in range(nx):
                if i==0 or i==ny-1 or j==0 or j==nx-1 or k==0 or k==nz-1:
                    rhs[gidx(k,i,j)] = 0
    omega = spsolve(L.tocsr(), rhs).reshape(nz, ny, nx)
    return omega

print("=== TEST 4 v2: Q-vector inversion convergence by resolution ===")
print("Reference: 88x104 3D QG-omega solve as 'truth'.")
om_truth = build_omega(88, 104)
print(f"Truth grid 88x104, std={om_truth.std():.3e}, max|omega|={np.abs(om_truth).max():.3e}")

for ny, nx in [(22, 26), (44, 52)]:
    om = build_omega(ny, nx)
    # subsample truth to this grid by nearest match
    step_y = (88-1)//(ny-1); step_x = (104-1)//(nx-1)
    om_ref = om_truth[:, ::step_y, ::step_x][:, :ny, :nx]
    # compare on interior
    mask = np.zeros_like(om, dtype=bool)
    mask[1:-1, 2:-2, 2:-2] = True
    a = om[mask]; b = om_ref[mask]
    corr = np.corrcoef(a,b)[0,1]
    ss_res = np.sum((a-b)**2); ss_tot = np.sum((b-b.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot>0 else float('nan')
    print(f"Grid {ny:>3}x{nx:>3}: corr={corr:.3f}, R2={R2:.3f}, std_ratio={a.std()/b.std():.3f}, |b|max={np.abs(b).max():.3e}")
