"""
2D finite-volume heat conduction solver (cell-centered), OpenFOAM-like style.

Requirements:
    numpy, scipy, matplotlib

This solver assembles A T = b using face fluxes:
    flux_f = -k_f * (T_nb - T_p)/d * A_f
with harmonic interpolation for k_f when k varies.

Boundary conditions supported:
  - Dirichlet: T = T0
  - Neumann:    -k dT/dn = q (flux given, positive out of domain)
  - Robin (convective): -k dT/dn = h*(T - T_inf)
  - Periodic (in x or y)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Callable, Optional

# ---------- Helper data structures ----------
class Dirichlet:
    def __init__(self, value: float):
        self.value = float(value)

class Neumann:
    def __init__(self, flux: float):
        # positive flux means heat leaving domain (consistent with sign conv)
        self.flux = float(flux)

class Robin:
    def __init__(self, h: float, T_inf: float):
        # -k dT/dn = h (T - T_inf)
        self.h = float(h)
        self.T_inf = float(T_inf)

class Periodic:
    def __init__(self):
        pass

BC = Dict[str, object]  # keys: 'left','right','bottom','top' -> Dirichlet/Neumann/Robin/Periodic

# ---------- Mesh and geometry ----------
def make_mesh(Lx: float, Ly: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Create a structured cell-centered grid.
    Returns arrays of x_centers (nx), y_centers (ny), dx, dy
    """
    dx = Lx / nx
    dy = Ly / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    return x, y, dx, dy

# ---------- Assembly ----------
def assemble_steady(nx: int, ny: int,
                    dx: float, dy: float,
                    k_field: np.ndarray,     # shape (ny, nx) (row-major y,x)
                    Q_field: Optional[np.ndarray],  # volumetric source W/m3, shape (ny,nx) or None
                    bc: BC) -> Tuple[sp.csr_matrix, np.ndarray, callable]:
    """
    Assemble sparse matrix A and RHS b for steady conduction.
    Returns A (N x N), b (N), and a function to map solution vector back to 2D T (ny,nx).
    """

    N = nx * ny
    row = []
    col = []
    data = []
    b = np.zeros(N, dtype=float)

    def idx(i, j):
        # i: x index 0..nx-1, j: y index 0..ny-1 -> linear index
        return j * nx + i

    # iterate cells
    for j in range(ny):
        for i in range(nx):
            P = idx(i, j) #cell center
            TP_coeff = 0.0  # accumulate diagonal
            rhs = 0.0

            kP = k_field[j, i]
            # Face areas (2D) = length of face out-of-plane = 1 m (unit depth) so face area = dy for vertical faces, dx for horizontal
            Ae = dy; Aw = dy; An = dx; As = dx
            # distances from cell center to neighbor center (or to ghost) in x,y
            de = dx; dw = dx; dn = dy; ds = dy

            # helper to compute face conductivity using harmonic mean
            def k_face(k_left, k_right, dist_left, dist_right):
                # harmonic with distances: k_f = area * 2 / (d_left/k_left + d_right/k_right)
                # but for conductivity (not multiplied by area) we use harmonic mean:
                if k_left == k_right:
                    return k_left
                # weighted by distances:
                return (dist_left + dist_right) / (dist_left / k_left + dist_right / k_right)

            # EAST face
            if i < nx - 1:
                kE = k_field[j, i + 1]
                kf = k_face(kP, kE, de / 2.0, de / 2.0)
                Ae_over_de = Ae * kf / de
                TP_coeff += Ae_over_de
                row.append(P); col.append(idx(i + 1, j)); data.append(-Ae_over_de)  # neighbor coeff
            else:
                # boundary face at East
                bc_e = bc.get('right', None)
                if isinstance(bc_e, Dirichlet):
                    # Strong Dirichlet --> move to RHS: aP*T_P + ... = aE*T_E => move T_E known to rhs
                    # approximate ghost T so that (T_ghost + T_P)/2 = T_boundary if you use linear ghost; simplest: enforce by large coefficient
                    Tbd = bc_e.value
                    # flux contribution: -(k_f * (T_bd - T_P)/ (de/2)) * Ae  => becomes + (k_f * Ae/(de/2)) * T_P  + RHS term
                    # choose kf = kP (single-sided)
                    kf = kP
                    coeff = Ae * kf / (de / 2.0)
                    TP_coeff += coeff
                    rhs += coeff * Tbd
                elif isinstance(bc_e, Neumann):
                    # prescribed outward flux q (W/m2). Positive q means heat going out of domain.
                    # For cell balance, add -q * Ae to RHS (because divergence sum + QV = 0)
                    rhs += -bc_e.flux * Ae
                elif isinstance(bc_e, Robin):
                    # -k dT/dn = h (T - T_inf)
                    # approximate dT/dn ~ (T_ghost - T_P)/(de/2) with T_ghost -> produce diagonal and RHS
                    h = bc_e.h
                    Tinf = bc_e.T_inf
                    kf = kP
                    # flux = h*(T_P - Tinf)  (sign conv as used)
                    # Matrix contribution: (h*Ae) * T_P and rhs = h*Ae*Tinf
                    coeff = h * Ae
                    TP_coeff += coeff
                    rhs += coeff * Tinf
                elif isinstance(bc_e, Periodic):
                    # map to periodic neighbor on left (i.e. wrap)
                    # index of periodic pair:
                    ip = 0  # leftmost cell index on other side
                    kE = k_field[j, 0]
                    kf = k_face(kP, kE, de / 2.0, de / 2.0)
                    Ae_over_de = Ae * kf / de
                    TP_coeff += Ae_over_de
                    row.append(P); col.append(idx(0, j)); data.append(-Ae_over_de)
                else:
                    # no BC specified -> assume insulated (Neumann q=0)
                    rhs += 0.0

            # WEST face
            if i > 0:
                kW = k_field[j, i - 1]
                kf = k_face(kP, kW, dw / 2.0, dw / 2.0)
                Aw_over_dw = Aw * kf / dw
                TP_coeff += Aw_over_dw
                row.append(P); col.append(idx(i - 1, j)); data.append(-Aw_over_dw)
            else:
                bc_w = bc.get('left', None)
                if isinstance(bc_w, Dirichlet):
                    Tbd = bc_w.value
                    kf = kP
                    coeff = Aw * kf / (dw / 2.0)
                    TP_coeff += coeff
                    rhs += coeff * Tbd
                elif isinstance(bc_w, Neumann):
                    rhs += -bc_w.flux * Aw
                elif isinstance(bc_w, Robin):
                    h = bc_w.h; Tinf = bc_w.T_inf
                    coeff = h * Aw
                    TP_coeff += coeff
                    rhs += coeff * Tinf
                elif isinstance(bc_w, Periodic):
                    # map to rightmost column
                    kW = k_field[j, nx - 1]
                    kf = k_face(kP, kW, dw / 2.0, dw / 2.0)
                    Aw_over_dw = Aw * kf / dw
                    TP_coeff += Aw_over_dw
                    row.append(P); col.append(idx(nx - 1, j)); data.append(-Aw_over_dw)
                else:
                    pass

            # NORTH face (j+1)
            if j < ny - 1:
                kN = k_field[j + 1, i]
                kf = k_face(kP, kN, dn / 2.0, dn / 2.0)
                An_over_dn = An * kf / dn
                TP_coeff += An_over_dn
                row.append(P); col.append(idx(i, j + 1)); data.append(-An_over_dn)
            else:
                bc_n = bc.get('top', None)
                if isinstance(bc_n, Dirichlet):
                    Tbd = bc_n.value
                    kf = kP
                    coeff = An * kf / (dn / 2.0)
                    TP_coeff += coeff
                    rhs += coeff * Tbd
                elif isinstance(bc_n, Neumann):
                    rhs += -bc_n.flux * An
                elif isinstance(bc_n, Robin):
                    h = bc_n.h; Tinf = bc_n.T_inf
                    coeff = h * An
                    TP_coeff += coeff
                    rhs += coeff * Tinf
                elif isinstance(bc_n, Periodic):
                    kN = k_field[0, i]
                    kf = k_face(kP, kN, dn / 2.0, dn / 2.0)
                    An_over_dn = An * kf / dn
                    TP_coeff += An_over_dn
                    row.append(P); col.append(idx(i, 0)); data.append(-An_over_dn)
                else:
                    pass

            # SOUTH face (j-1)
            if j > 0:
                kS = k_field[j - 1, i]
                kf = k_face(kP, kS, ds / 2.0, ds / 2.0)
                As_over_ds = As * kf / ds
                TP_coeff += As_over_ds
                row.append(P); col.append(idx(i, j - 1)); data.append(-As_over_ds)
            else:
                bc_s = bc.get('bottom', None)
                if isinstance(bc_s, Dirichlet):
                    Tbd = bc_s.value
                    kf = kP
                    coeff = As * kf / (ds / 2.0)
                    TP_coeff += coeff
                    rhs += coeff * Tbd
                elif isinstance(bc_s, Neumann):
                    rhs += -bc_s.flux * As
                elif isinstance(bc_s, Robin):
                    h = bc_s.h; Tinf = bc_s.T_inf
                    coeff = h * As
                    TP_coeff += coeff
                    rhs += coeff * Tinf
                elif isinstance(bc_s, Periodic):
                    kS = k_field[ny - 1, i]
                    kf = k_face(kP, kS, ds / 2.0, ds / 2.0)
                    As_over_ds = As * kf / ds
                    TP_coeff += As_over_ds
                    row.append(P); col.append(idx(i, ny - 1)); data.append(-As_over_ds)
                else:
                    pass

            # Add source contribution Q (volumetric). For steady equation it's added to RHS
            if Q_field is not None:
                QP = Q_field[j, i]
                # integral of Q over cell (V=dx*dy*1 depth)
                rhs += -QP * dx * dy  # depends on sign convention; we used divergence + Q = 0 earlier
            # finalize diagonal
            row.append(P); col.append(P); data.append(TP_coeff)
            b[P] = rhs
                
    """_summary of constructing the matrix_
    For example: given a 3 x 3 grid
    (i,j):   (0,0)=0  (1,0)=1  (2,0)=2
                (0,1)=3  (1,1)=4  (2,1)=5
                (0,2)=6  (1,2)=7  (2,2)=8
    For 4th cell ->
        4T4 - 1T5 - 1T3 - 1T7 - 1T1 = 0
    For Corner cell 0th cell ->
        6T0 - 1T1 - 1T3 = 0
    The constructed matrix looks like for all cells:
    A = [
        [ 6 -1  0 -1  0  0  0  0  0],
        [-1  5 -1  0 -1  0  0  0  0],
        [ 0 -1  6  0  0 -1  0  0  0],
        [-1  0  0  5 -1  0 -1  0  0],
        [ 0 -1  0 -1  4 -1  0 -1  0],
        [ 0  0 -1  0 -1  5  0  0 -1],
        [ 0  0  0 -1  0  0  6 -1  0],
        [ 0  0  0  0 -1  0 -1  5 -1],
        [ 0  0  0  0  0 -1  0 -1  6]
    ]
    Returns:
        _type_: _description_
    """
    A = sp.csr_matrix((data, (row, col)), shape=(N, N))
    def reshape_sol(x):
        return x.reshape((ny, nx))
    return A, b, reshape_sol

# ---------- Example usage ----------
def demo():
    Lx, Ly = 1.0, 1.0
    nx, ny = 60, 30
    x, y, dx, dy = make_mesh(Lx, Ly, nx, ny)

    # thermal conductivity field (W/mK): example: piecewise or uniform
    k_field = np.ones((ny, nx)) * 1.0
    # e.g. left half higher conductivity
    k_field[:, :nx//2] = 5.0

    # volumetric source (W/m3)
    Q = np.zeros((ny, nx))
    # apply a small heat source in center
    Q[ny//2 - 1:ny//2 + 1, nx//2 - 1:nx//2 + 1] = 1e5

    # boundary conditions
    bc = {
        'left': Dirichlet(300.0),        # T=300K at x=0
        'right': Dirichlet(200.0),
        #'right': Robin(h=10.0, T_inf=290.0),  # convective on right face
        'top': Dirichlet(200.0),
        'bottom': Dirichlet(300.0),
        #'top': Neumann(flux=0.0),        # insulated
        #'bottom': Neumann(flux=0.0),
    }

    A, b, reshape_sol = assemble_steady(nx, ny, dx, dy, k_field, Q, bc)
    print("Assembled A shape:", A.shape, "nnz:", A.nnz)

    # solve (sparse)
    Tvec = spla.spsolve(A.tocsr(), b)
    T = reshape_sol(Tvec)

    # plot
    plt.figure(figsize=(6,3))
    X, Y = np.meshgrid(x, y)
    cs = plt.contourf(X, Y, T, 20, cmap='inferno')
    plt.colorbar(cs, label='T (K)')
    plt.title('Steady temperature (K)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()
