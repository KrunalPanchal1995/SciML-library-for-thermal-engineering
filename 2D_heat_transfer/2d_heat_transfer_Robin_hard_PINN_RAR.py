import os
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(60)
torch.set_num_interop_threads(40)

os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import deepxde.backend as bkd

dde.config.set_default_float("float64")

# -----------------------
# Config
# -----------------------
TRAIN = True
SAVE_DIR = Path("checkpoints_heat_robin_RAR")
SAVE_DIR.mkdir(exist_ok=True)
MODEL_PREFIX = str(SAVE_DIR / "heat_hetero_robin")

NUM_DOMAIN = 16384        # make Sobol happy
NUM_BOUNDARY = 2048
TRAIN_DIST = "Sobol"
RESAMPLE_PERIOD = 1000

# Base training per stage
ADAM_ITERS = 5000
LBFGS_MAXITER = 5000

# RAR
RAR_STAGES = 3            # number of refinement rounds
NCAND = 200000            # candidate points to probe residual
K_ADD = 4000              # points to add each stage

# Robin parameters
h_top, h_right = 10.0, 5.0
Tinf_top, Tinf_right = 0.0, 0.0

# -----------------------
# Geometry
# -----------------------
geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

exclusions = np.array([[0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0],
                       [1.0, 1.0]])

# -----------------------
# k(x,y): numpy + torch safe
# -----------------------
def k_fn(X):
    k1, k2 = 1.0, 10.0
    eps = 0.02

    if isinstance(X, np.ndarray):
        x = X[:, 0:1]
        y = X[:, 1:2]
        s = 0.5 * (1.0 + np.tanh((x - 0.5) / eps))
        k = k1 + (k2 - k1) * s
        k = k * (1.0 + 0.1 * np.sin(2.0 * np.pi * y))
        return torch.as_tensor(k)

    x = X[:, 0:1]
    y = X[:, 1:2]
    s = 0.5 * (1.0 + bkd.tanh((x - 0.5) / eps))
    k = k1 + (k2 - k1) * s
    k = k * (1.0 + 0.1 * bkd.sin(2.0 * np.pi * y))
    return k

# -----------------------
# PDE: div(k grad T)=0
# -----------------------
def pde(X, T):
    Tx = dde.grad.jacobian(T, X, i=0, j=0)
    Ty = dde.grad.jacobian(T, X, i=0, j=1)
    k = k_fn(X)
    qx = k * Tx
    qy = k * Ty
    dqx_dx = dde.grad.jacobian(qx, X, i=0, j=0)
    dqy_dy = dde.grad.jacobian(qy, X, i=0, j=1)
    return dqx_dx + dqy_dy

# -----------------------
# Boundary indicators
# -----------------------
def on_left(X, on_boundary):   return on_boundary and dde.utils.isclose(X[0], 0.0)
def on_bottom(X, on_boundary): return on_boundary and dde.utils.isclose(X[1], 0.0)
def on_top(X, on_boundary):    return on_boundary and dde.utils.isclose(X[1], 1.0)
def on_right(X, on_boundary):  return on_boundary and dde.utils.isclose(X[0], 1.0)

# Dirichlet (hard enforced; keep for monitoring)
bc_left = dde.icbc.DirichletBC(geom, lambda X: 0.0, on_left)
bc_bottom = dde.icbc.DirichletBC(geom, lambda X: np.sin(np.pi * X[:, 0:1]), on_bottom)

# Robin: dT/dn = -(h/k)(T-Tinf)
def g_top(X, T):
    k = k_fn(X)
    if hasattr(T, "dtype"):
        k = k.to(dtype=T.dtype, device=T.device)
    return -(h_top / k) * (T - Tinf_top)

def g_right(X, T):
    k = k_fn(X)
    if hasattr(T, "dtype"):
        k = k.to(dtype=T.dtype, device=T.device)
    return -(h_right / k) * (T - Tinf_right)

bc_top = dde.icbc.RobinBC(geom, g_top, on_top)
bc_right = dde.icbc.RobinBC(geom, g_right, on_right)

# -----------------------
# Optional: oversample Robin edges as anchors (high ROI)
# -----------------------
def sample_top(n):
    x = np.random.rand(n, 1); y = np.ones((n, 1))
    return np.hstack([x, y])

def sample_right(n):
    x = np.ones((n, 1)); y = np.random.rand(n, 1)
    return np.hstack([x, y])

anchors = np.vstack([sample_top(8000), sample_right(8000)])  # start with Robin-heavy anchors

# -----------------------
# Network + hard Dirichlet transform (left & bottom)
# T(x,y) = (1-y) sin(pi x) + x*y*N(x,y)
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
# Helper: build data/model (reuse same net weights!)
# -----------------------
def build_model(current_anchors):
    data = dde.data.PDE(
        geom, pde,
        [bc_left, bc_bottom, bc_top, bc_right],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        train_distribution=TRAIN_DIST,
        anchors=current_anchors,
        exclusions=exclusions,
    )
    return dde.Model(data, net)

def train_stage(model):
    resampler = dde.callbacks.PDEPointResampler(period=RESAMPLE_PERIOD)

    model.compile("adam", lr=5e-4, loss_weights=[1, 0, 0, 100, 100])
    model.train(iterations=ADAM_ITERS, callbacks=[resampler], display_every=1000)

    dde.optimizers.set_LBFGS_options(maxiter=LBFGS_MAXITER)
    model.compile("L-BFGS", loss_weights=[1, 0, 0, 100, 100])
    model.train(display_every=200)

# -----------------------
# Train + RAR loop
# -----------------------
model = build_model(anchors)

if TRAIN:
    # Stage 0: initial training
    train_stage(model)

    # RAR stages
    for s in range(RAR_STAGES):
        Xcand = geom.random_points(NCAND)
        r = np.abs(model.predict(Xcand, operator=pde)).reshape(-1)
        Xnew = Xcand[np.argsort(-r)[:K_ADD]]

        anchors = np.vstack([anchors, Xnew])
        print(f"RAR stage {s+1}/{RAR_STAGES}: added {K_ADD} points, anchors now {anchors.shape[0]}")

        # rebuild model with same net (weights preserved), but new data
        model = build_model(anchors)
        train_stage(model)

    save_path = model.save(MODEL_PREFIX)
    (SAVE_DIR / "last_save_path.txt").write_text(save_path)
    print("Saved model to:", save_path)

else:
    save_path = (SAVE_DIR / "last_save_path.txt").read_text().strip()
    model.restore(save_path)
    print("Restored model from:", save_path)

# -----------------------
# Predict + save + plot
# -----------------------
nx = 201
xs = np.linspace(0.0, 1.0, nx)
ys = np.linspace(0.0, 1.0, nx)
XX, YY = np.meshgrid(xs, ys)
Xg = np.vstack([XX.ravel(), YY.ravel()]).T
T_pred = model.predict(Xg).reshape(nx, nx)

np.savez(SAVE_DIR / "solution_grid.npz", xs=xs, ys=ys, T=T_pred)
print("Saved grid solution to:", SAVE_DIR / "solution_grid.npz")

plt.figure()
plt.contourf(XX, YY, T_pred, levels=50)
plt.colorbar()
plt.xlabel("x"); plt.ylabel("y")
plt.title("T(x,y): heterogeneous k + mixed BC + Robin (RAR)")
plt.tight_layout()
plt.savefig(SAVE_DIR / "T_field.png", dpi=200)