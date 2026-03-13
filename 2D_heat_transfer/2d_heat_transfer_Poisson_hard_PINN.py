import deepxde as dde
import numpy as np
import deepxde.backend as bkd

dde.config.set_default_float("float64")

geom = dde.geometry.Rectangle([0, 0], [1, 1])

def pde_poisson(x, T):
    T_xx = dde.grad.hessian(T, x, i=0, j=0)
    T_yy = dde.grad.hessian(T, x, i=1, j=1)
    q = 10.0 * bkd.sin(np.pi * x[:, 0:1]) * bkd.sin(np.pi * x[:, 1:2])
    return T_xx + T_yy + q  # ΔT + q = 0

# all-zero Dirichlet BCs
bc0 = dde.icbc.DirichletBC(geom, lambda x: 0.0, lambda x, on_boundary: on_boundary)

data = dde.data.PDE(
    geom,
    pde_poisson,
    [bc0],
    num_domain=20000,
    num_boundary=2000,
)

net = dde.maps.FNN([2] + [128, 128, 128, 128] + [1], "tanh", "Glorot normal")

# Hard-enforce zero Dirichlet BCs on all sides:
# T = x(1-x)y(1-y) * N(x,y)
def output_transform_zeroBC(x, y):
    X = x[:, 0:1]
    Y = x[:, 1:2]
    return X * (1 - X) * Y * (1 - Y) * y

net.apply_output_transform(output_transform_zeroBC)

model = dde.Model(data, net)

resampler = dde.callbacks.PDEPointResampler(period=1000)

model.compile("adam", lr=1e-3)
model.train(iterations=20000, callbacks=[resampler])

dde.optimizers.set_LBFGS_options(maxiter=50000)
model.compile("L-BFGS")
model.train()

def T_true_poisson(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return (10.0 / (2.0 * np.pi**2)) * np.sin(np.pi * x) * np.sin(np.pi * y)

nx = 151
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, nx)
XX, YY = np.meshgrid(xs, ys)
Xg = np.vstack([XX.ravel(), YY.ravel()]).T

T_pred = model.predict(Xg)
T_ex = T_true_poisson(Xg)
rel_l2 = np.linalg.norm(T_pred - T_ex) / np.linalg.norm(T_ex)
print("Relative L2 error (Poisson):", rel_l2)