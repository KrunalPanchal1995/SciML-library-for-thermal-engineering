# pip install deepxde torch
import deepxde as dde
import numpy as np

# --- PDE definition (Laplace) ---
def pde(x, T):
    T_xx = dde.grad.hessian(T, x, i=0, j=0)
    T_yy = dde.grad.hessian(T, x, i=1, j=1)
    return T_xx + T_yy

# --- Domain ---
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# --- Boundary indicator helpers ---
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1.0)

def boundary_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)

def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1.0)

# --- BCs ---
bc_left   = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_left)
bc_right  = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_right)
bc_bottom = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_bottom)
bc_top    = dde.icbc.DirichletBC(geom, lambda x: np.sin(np.pi * x[:, 0:1]), boundary_top)

data = dde.data.PDE(
    geom,
    pde,
    [bc_left, bc_right, bc_bottom, bc_top],
    num_domain=4000,
    num_boundary=400,
)

# --- Network ---
net = dde.maps.FNN([2] + [64, 64, 64] + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss_weights=[1, 10, 10, 10, 10])  # [PDE, BCs...]

losshistory, train_state = model.train(iterations=15000)

# --- Validate vs analytic ---
def T_true(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)

# grid
nx = 101
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, nx)
XX, YY = np.meshgrid(xs, ys)
X = np.vstack([XX.ravel(), YY.ravel()]).T

T_pred = model.predict(X)
T_ex   = T_true(X)

rel_l2 = np.linalg.norm(T_pred - T_ex) / np.linalg.norm(T_ex)
print("Relative L2 error:", rel_l2)