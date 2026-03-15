import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn

# --- Reference stiff RHS ---
def f_true(t, y):
    y1, y2, y3 = y
    return np.array([
        -0.04*y1 + 1e4*y2*y3,
         0.04*y1 - 1e4*y2*y3 - 3e7*y2*y2,
         3e7*y2*y2
    ])

# --- Generate training snapshots (states + RHS) from reference integration ---
t_eval = np.logspace(-6, 5, 2000)
sol = solve_ivp(f_true, (t_eval[0], t_eval[-1]), [1.,0.,0.], method="BDF", t_eval=t_eval, rtol=1e-10, atol=1e-14)

Y = sol.y.T  # (N, 3)
F = np.array([f_true(0.0, y) for y in Y])  # (N, 3)

# scaling helps a lot (log for stiff states; here keep simple but normalize)
Ymean, Ystd = Y.mean(0), Y.std(0) + 1e-12
Fmean, Fstd = F.mean(0), F.std(0) + 1e-12

Yt = torch.tensor((Y - Ymean)/Ystd, dtype=torch.float32)
Ft = torch.tensor((F - Fmean)/Fstd, dtype=torch.float32)

# --- Neural RHS surrogate ---
net = nn.Sequential(
    nn.Linear(3, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 3)
)

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

for it in range(20000):
    pred = net(Yt)
    loss = ((pred - Ft)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 2000 == 0:
        print(it, float(loss))

# --- Wrap learned RHS back to physical scaling ---
@torch.no_grad()
def f_nn(y_np):
    y = torch.tensor((y_np - Ymean)/Ystd, dtype=torch.float32)
    f = net(y).numpy()
    return f * Fstd + Fmean

# --- Validate by integrating learned ODE (still stiff; use BDF to be fair) ---
def f_sur(t, y):  # SciPy signature
    return f_nn(y)

sol_nn = solve_ivp(f_sur, (t_eval[0], t_eval[-1]), [1.,0.,0.], method="BDF", t_eval=t_eval)

# relative trajectory error
err = np.linalg.norm(sol_nn.y - sol.y) / np.linalg.norm(sol.y)
print("Trajectory relative L2:", err)
