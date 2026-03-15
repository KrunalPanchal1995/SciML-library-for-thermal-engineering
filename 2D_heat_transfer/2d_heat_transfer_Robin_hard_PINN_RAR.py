"""
2D heterogeneous conduction with mixed BCs including Robin (convection).

PDE: div( k(x,y) * grad(T) ) = 0  on [0,1]x[0,1]

BCs:
  - Left (x=0):   Dirichlet  T = 0
  - Bottom (y=0): Dirichlet  T = sin(pi x)
  - Top (y=1):    Robin  -k dT/dn = h_top (T - Tinf_top)
  - Right (x=1):  Robin  -k dT/dn = h_right (T - Tinf_right)

We hard-enforce the Dirichlet BCs (left & bottom) using an output_transform,
and enforce Robin BCs via loss terms.
"""

import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parallel CPU + backend setup
# ----------------------------
torch.set_num_threads(60)
torch.set_num_interop_threads(40)

# Must be set BEFORE importing deepxde
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import deepxde.backend as bkd



# -----------------------
# Configuration
# -----------------------
dde.config.set_default_float("float64")  # helps second-derivative stability
TRAIN = True  # set False to just restore + predict/plot
SAVE_DIR = Path("checkpoints_heat_robin")
SAVE_DIR.mkdir(exist_ok=True)
MODEL_PREFIX = str(SAVE_DIR / "heat_hetero_robin")  # prefix (DeepXDE will add suffixes)

# Collocation / training
NUM_DOMAIN = 16384
NUM_BOUNDARY = 2048
TRAIN_DIST = "Sobol"   # good coverage in 2D
RESAMPLE_PERIOD = 1000

# Optimizers
ADAM_ITERS = 20000
LBFGS_MAXITER = 5000  # you can crank this up later

# Robin parameters
h_top = 10.0
h_right = 5.0
Tinf_top = 0.0
Tinf_right = 0.0


# -----------------------
# Geometry
# -----------------------
geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])


# -----------------------
# Heterogeneous conductivity k(x,y)
# Smooth 2-material transition around x=0.5 (avoid non-differentiable step)
# -----------------------

def k_fn(X):
    """
    Returns k(X) as a torch tensor for BOTH cases:
      - X is torch.Tensor (PDE residual)
      - X is np.ndarray   (BC evaluation)
    """
    k1, k2 = 1.0, 10.0
    eps = 0.02

    if isinstance(X, np.ndarray):
        x = X[:, 0:1]
        y = X[:, 1:2]
        s = 0.5 * (1.0 + np.tanh((x - 0.5) / eps))
        k = k1 + (k2 - k1) * s
        k = k * (1.0 + 0.1 * np.sin(2.0 * np.pi * y))
        # Convert to torch tensor (float64 preserved because numpy array is float64)
        return torch.as_tensor(k)

    # torch Tensor branch (keeps autodiff)
    x = X[:, 0:1]
    y = X[:, 1:2]
    s = 0.5 * (1.0 + bkd.tanh((x - 0.5) / eps))
    k = k1 + (k2 - k1) * s
    k = k * (1.0 + 0.1 * bkd.sin(2.0 * np.pi * y))
    return k


# -----------------------
# PDE: div( k grad T ) = 0
# Implement via flux + divergence:
#   qx = k * Tx, qy = k * Ty
#   div = d/dx(qx) + d/dy(qy)
# -----------------------
def pde(X, T):
    Tx = dde.grad.jacobian(T, X, i=0, j=0)  # dT/dx
    Ty = dde.grad.jacobian(T, X, i=0, j=1)  # dT/dy

    k = k_fn(X)
    qx = k * Tx
    qy = k * Ty

    dqx_dx = dde.grad.jacobian(qx, X, i=0, j=0)
    dqy_dy = dde.grad.jacobian(qy, X, i=0, j=1)
    return dqx_dx + dqy_dy


# -----------------------
# Boundary indicators
# -----------------------
def on_left(X, on_boundary):
    return on_boundary and dde.utils.isclose(X[0], 0.0)

def on_bottom(X, on_boundary):
    return on_boundary and dde.utils.isclose(X[1], 0.0)

def on_top(X, on_boundary):
    return on_boundary and dde.utils.isclose(X[1], 1.0)

def on_right(X, on_boundary):
    return on_boundary and dde.utils.isclose(X[0], 1.0)


# -----------------------
# Dirichlet BCs (we'll hard-enforce via output_transform, but keep these for monitoring)
# Left: T=0
# Bottom: T=sin(pi x)
# -----------------------
bc_left = dde.icbc.DirichletBC(geom, lambda X: 0.0, on_left)
bc_bottom = dde.icbc.DirichletBC(geom, lambda X: np.sin(np.pi * X[:, 0:1]), on_bottom)


# -----------------------
# Robin BCs:
# DeepXDE RobinBC enforces dT/dn = g(X, T)  :contentReference[oaicite:2]{index=2}
#
# Convection: -k dT/dn = h (T - Tinf)
# => dT/dn = -(h/k) (T - Tinf)
# -----------------------
def g_top(X, T):
    k = k_fn(X)
    return -(h_top / k) * (T - Tinf_top)

def g_right(X, T):
    k = k_fn(X)
    return -(h_right / k) * (T - Tinf_right)

bc_top = dde.icbc.RobinBC(geom, g_top, on_top)
bc_right = dde.icbc.RobinBC(geom, g_right, on_right)


# -----------------------
# Data object
# -----------------------
exclusions = np.array([[0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0],
                       [1.0, 1.0]])



data = dde.data.PDE(
    geom,
    pde,
    [bc_left, bc_bottom, bc_top, bc_right],
    num_domain=NUM_DOMAIN,
    num_boundary=NUM_BOUNDARY,
    train_distribution=TRAIN_DIST,
    exclusions=exclusions,
)


# -----------------------
# Network + output transform (hard enforce Dirichlet on left & bottom)
#
# Want:
#   T(0,y) = 0
#   T(x,0) = sin(pi x)
#
# A simple ansatz that satisfies both:
#   T(x,y) = (1 - y) * sin(pi x) + x*y * N(x,y)
#
# Check:
#   y=0 -> T=sin(pi x)
#   x=0 -> T=(1-y)*sin(0) + 0 = 0
# No constraint forced on top/right, so Robin is still meaningful there.
# -----------------------
net = dde.maps.FNN([2, 128, 128, 128, 128, 1], "tanh", "Glorot normal")

def out_transform(X, N):
    x = X[:, 0:1]
    y = X[:, 1:2]
    base = (1.0 - y) * bkd.sin(np.pi * x)
    corr = x * y * N
    return base + corr

net.apply_output_transform(out_transform)


# -----------------------
# Model
# -----------------------
model = dde.Model(data, net)


# -----------------------
# Train or restore
# -----------------------
# DeepXDE supports save/restore via model.save / model.restore  :contentReference[oaicite:3]{index=3}
# Note: you must reconstruct the same model/net before restore.
save_path_txt = SAVE_DIR / "last_save_path.txt"

if TRAIN:
    resampler = dde.callbacks.PDEPointResampler(period=RESAMPLE_PERIOD)

    model.compile("adam", lr=1e-3)
    losshistory, train_state = model.train(
        iterations=ADAM_ITERS,
        callbacks=[resampler],
        display_every=1000,
    )

    dde.optimizers.set_LBFGS_options(maxiter=LBFGS_MAXITER)
    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=200)

    # Save weights
    save_path = model.save(MODEL_PREFIX)  # returns actual path  :contentReference[oaicite:4]{index=4}
    save_path_txt.write_text(save_path)
    print("Saved model to:", save_path)

else:
    if not save_path_txt.exists():
        raise FileNotFoundError(f"No saved path found at {save_path_txt}. Train once with TRAIN=True.")
    save_path = save_path_txt.read_text().strip()
    model.restore(save_path)
    print("Restored model from:", save_path)


# -----------------------
# Predict on a grid + save solution
# -----------------------
nx = 201
xs = np.linspace(0.0, 1.0, nx)
ys = np.linspace(0.0, 1.0, nx)
XX, YY = np.meshgrid(xs, ys)
Xg = np.vstack([XX.ravel(), YY.ravel()]).T

T_pred = model.predict(Xg).reshape(nx, nx)

# also save k-field for reference/plotting
# k_fn expects a backend tensor inside training, but for postprocessing use numpy version:
def k_np(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    k1, k2 = 1.0, 10.0
    eps = 0.02
    s = 0.5 * (1.0 + np.tanh((x - 0.5) / eps))
    k = k1 + (k2 - k1) * s
    k = k * (1.0 + 0.1 * np.sin(2.0 * np.pi * y))
    return k

K_field = k_np(Xg).reshape(nx, nx)

# Save predicted fields so you don't need to retrain
np.savez(SAVE_DIR / "solution_grid.npz", xs=xs, ys=ys, T=T_pred, K=K_field)
print("Saved grid solution to:", SAVE_DIR / "solution_grid.npz")


# -----------------------
# Plotting
# -----------------------
plt.figure()
plt.contourf(XX, YY, T_pred, levels=50)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("PINN Temperature T(x,y) with mixed BCs + Robin + heterogeneous k")
plt.tight_layout()
plt.savefig(SAVE_DIR / "T_field.png", dpi=200)

plt.figure()
plt.contourf(XX, YY, K_field, levels=50)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Conductivity field k(x,y)")
plt.tight_layout()
plt.savefig(SAVE_DIR / "k_field.png", dpi=200)

plt.show()
