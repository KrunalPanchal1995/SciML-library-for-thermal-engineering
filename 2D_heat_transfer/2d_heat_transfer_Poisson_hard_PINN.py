import os
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

dde.config.set_default_float("float64")

# ----------------------------
# Problem definition
# ----------------------------
geom = dde.geometry.Rectangle([0, 0], [1, 1])

def pde_poisson(x, T):
    T_xx = dde.grad.hessian(T, x, i=0, j=0)
    T_yy = dde.grad.hessian(T, x, i=1, j=1)
    q = 10.0 * bkd.sin(np.pi * x[:, 0:1]) * bkd.sin(np.pi * x[:, 1:2])
    return T_xx + T_yy + q  # ΔT + q = 0

# All-zero Dirichlet BCs
bc0 = dde.icbc.DirichletBC(geom, lambda x: 0.0, lambda x, on_boundary: on_boundary)

data = dde.data.PDE(
    geom,
    pde_poisson,
    [bc0],
    num_domain=20000,
    num_boundary=2000,
)

net = dde.maps.FNN([2] + [128, 128, 128, 128] + [1], "tanh", "Glorot normal")

# Hard-enforce zero Dirichlet BCs: T = x(1-x)y(1-y) * N(x,y)
def output_transform_zeroBC(x, y):
    X = x[:, 0:1]
    Y = x[:, 1:2]
    return X * (1 - X) * Y * (1 - Y) * y

net.apply_output_transform(output_transform_zeroBC)

model = dde.Model(data, net)

# ----------------------------
# Save/restore paths
# ----------------------------
os.makedirs("artifacts", exist_ok=True)
MODEL_PREFIX = os.path.join("artifacts", "poisson_pinn_torch")  # DeepXDE will add suffixes
PLOT_PATH = os.path.join("artifacts", "poisson_compare.png")

# If True: skip training if a saved checkpoint exists
SKIP_TRAIN_IF_SAVED = True

# DeepXDE creates files like:
#   poisson_pinn_torch-xxxx.pt (torch checkpoint) + metadata
# Easiest check: look for any file that starts with prefix.
def model_checkpoint_exists(prefix: str) -> bool:
    folder = os.path.dirname(prefix) or "."
    base = os.path.basename(prefix)
    return any(fn.startswith(base) for fn in os.listdir(folder))

# ----------------------------
# Train (or restore)
# ----------------------------
if SKIP_TRAIN_IF_SAVED and model_checkpoint_exists(MODEL_PREFIX):
    print(f"[INFO] Found saved model under '{MODEL_PREFIX}*'. Restoring and skipping training.")
    model.restore(MODEL_PREFIX)
else:
    print("[INFO] Training model...")
    resampler = dde.callbacks.PDEPointResampler(period=1000)

    model.compile("adam", lr=1e-3)
    model.train(iterations=20000, callbacks=[resampler])

    dde.optimizers.set_LBFGS_options(maxiter=5000)
    model.compile("L-BFGS")
    model.train()

    print(f"[INFO] Saving model to '{MODEL_PREFIX}*'")
    model.save(MODEL_PREFIX)

# ----------------------------
# Validation + visualization
# ----------------------------
def T_true_poisson(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return (10.0 / (2.0 * np.pi**2)) * np.sin(np.pi * x) * np.sin(np.pi * y)

nx = 151
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, nx)
XX, YY = np.meshgrid(xs, ys)
Xg = np.vstack([XX.ravel(), YY.ravel()]).T

T_pred = model.predict(Xg).reshape(nx, nx)
T_ex = T_true_poisson(Xg).reshape(nx, nx)
abs_err = np.abs(T_pred - T_ex)

rel_l2 = np.linalg.norm(T_pred - T_ex) / np.linalg.norm(T_ex)
print("Relative L2 error (Poisson):", rel_l2)

# Plot: Prediction vs True vs Error
fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

im0 = axs[0].imshow(T_pred, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
axs[0].set_title("PINN prediction")
axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
plt.colorbar(im0, ax=axs[0], fraction=0.046)

im1 = axs[1].imshow(T_ex, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
axs[1].set_title("True solution")
axs[1].set_xlabel("x"); axs[1].set_ylabel("y")
plt.colorbar(im1, ax=axs[1], fraction=0.046)

im2 = axs[2].imshow(abs_err, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
axs[2].set_title("Absolute error |T_pred - T_true|")
axs[2].set_xlabel("x"); axs[2].set_ylabel("y")
plt.colorbar(im2, ax=axs[2], fraction=0.046)

fig.suptitle(f"Poisson PINN (PyTorch) — Relative L2 error = {rel_l2:.3e}", y=1.05)

plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
print(f"[INFO] Saved comparison plot: {PLOT_PATH}")
plt.show()
