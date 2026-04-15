"""
Neural ODE Surrogate for H2/O2 Ignition Kinetics
=================================================
Module 2 — SciML Curriculum

Features:
  - Cantera-based ground truth data generation
  - Neural ODE with adjoint sensitivity (torchdiffeq)
  - Stiffness diagnostic: Jacobian eigenvalue ratio every 50 epochs
  - OH species profile comparison plot (GT vs Neural ODE)

Dependencies:
    pip install cantera torch torchdiffeq matplotlib numpy scipy
"""

import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint_adjoint as odeint
import torch.autograd.functional as AF
torch.set_num_threads(40)
torch.set_num_interop_threads(40)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Mechanism details
# h2o2.yaml species order (Cantera default):
# 0:H2  1:H  2:O  3:O2  4:OH  5:H2O  6:HO2  7:H2O2  8:AR
OH_IDX   = 4
N_SPECIES = 10        # 9 species + temperature (normalized)
STATE_DIM = N_SPECIES + 1   # 10-dimensional state vector

# ─────────────────────────────────────────────────────────────
# 1.  DATA GENERATION  (Cantera)
# ─────────────────────────────────────────────────────────────

def generate_ignition_data(
    T0_range: tuple = (900, 1400),   # K
    P: float = None,                 # Pa  (filled below)
    phi: float = 1.0,
    n_samples: int = 120,
    t_end: float = 1e-3,             # 1 ms
    n_points: int = 300,
    save_path: str = "h2_ignition_data.npy",
):
    """
    Integrate H2/O2/Ar constant-pressure ignition trajectories.
    Returns a dict with arrays of shape (N, T, S) where
    S = N_species + 1  (mole-fractions + normalised T).
    """
    try:
        import cantera as ct
    except ImportError:
        raise ImportError("Cantera not found.  Install with:  pip install cantera")

    if P is None:
        P = ct.one_atm

    if os.path.exists(save_path):
        print(f"[DATA] Loading cached dataset from '{save_path}'")
        return np.load(save_path, allow_pickle=True).item()

    print(f"[DATA] Generating {n_samples} ignition trajectories …")
    gas = ct.Solution("h2o2.yaml")
    species_names = gas.species_names          # list of 9 strings

    T0_values    = np.linspace(*T0_range, n_samples)
    trajectories = np.zeros((n_samples, n_points, STATE_DIM), dtype=np.float32)
    t_span       = np.linspace(0, t_end, n_points)

    for idx, T0 in enumerate(T0_values):
        gas.set_equivalence_ratio(phi, "H2", "O2:1, AR:3.76")
        gas.TP = T0, P

        r   = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([r])
        sim.atol = 1e-15
        sim.rtol = 1e-9

        for i, t in enumerate(t_span):
            try:
                sim.advance(t)
            except Exception:
                # If integrator fails, repeat last successful state
                pass
            trajectories[idx, i, :N_SPECIES] = r.thermo.X          # mole fracs
            trajectories[idx, i,  N_SPECIES] = r.thermo.T / 3000.0  # normalised T

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{n_samples} done  (T0={T0:.0f} K)")

    dataset = {
        "trajectories": trajectories,   # (N, T, S)
        "t_span":       t_span,
        "species":      species_names,
        "T0_values":    T0_values,
    }
    np.save(save_path, dataset)
    print(f"[DATA] Saved to '{save_path}'")
    return dataset


# ─────────────────────────────────────────────────────────────
# 2.  NEURAL ODE  ARCHITECTURE
# ─────────────────────────────────────────────────────────────

class ChemistryODEFunc(nn.Module):
    """
    Parameterises  dy/dt = NN_θ(y)  in the 10-D state space.
    Tanh activations — smooth, bounded derivatives; important
    for stable adjoint solve.
    """

    def __init__(self, state_dim: int = STATE_DIM,
                 hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()

        layers = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

        # Small-weight init → near-zero initial velocities (stable start)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # t: scalar  |  y: (batch, state_dim)
        return self.net(y)


class NeuralODESurrogate(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = 128):
        super().__init__()
        self.odefunc = ChemistryODEFunc(state_dim, hidden_dim)

    def forward(self, y0: torch.Tensor,
                t_span: torch.Tensor,
                solver: str = "implicit_adams") -> torch.Tensor:
        """
        y0:     (batch, state_dim)
        t_span: (n_points,)
        Returns (n_points, batch, state_dim)
        """
        return odeint(
            self.odefunc, y0, t_span,
            method=solver,
            rtol=1e-5, atol=1e-6,
            adjoint_params=list(self.odefunc.parameters()),
        )


# ─────────────────────────────────────────────────────────────
# 3.  STIFFNESS DIAGNOSTIC
# ─────────────────────────────────────────────────────────────

def compute_stiffness_ratio(odefunc: ChemistryODEFunc,
                            y_sample: torch.Tensor) -> dict:
    """
    Compute the Jacobian  J = ∂(NN_θ(y)) / ∂y  at a single
    state vector y_sample ∈ ℝ^{state_dim}.

    Stiffness ratio  =  |λ_max| / |λ_min|
    A high ratio → stiff dynamics → consider implicit_adams.

    Parameters
    ----------
    odefunc  : ChemistryODEFunc
    y_sample : (state_dim,) tensor  — a single state point

    Returns
    -------
    dict with keys: ratio, lambda_max, lambda_min, eigenvalues
    """
    odefunc.eval()
    y_sample = y_sample.detach().requires_grad_(True)

    # Wrap so autograd.functional sees a single-input function
    def func(y):
        return odefunc(torch.tensor(0.0), y.unsqueeze(0)).squeeze(0)

    # J has shape (state_dim, state_dim)
    J = AF.jacobian(func, y_sample, create_graph=False, strict=False)
    J_np = J.detach().cpu().numpy()

    # Eigenvalues of the Jacobian (complex in general)
    eigvals = np.linalg.eigvals(J_np)
    abs_eigs = np.abs(eigvals)

    # Guard against near-zero eigenvalues
    nonzero = abs_eigs[abs_eigs > 1e-12]
    if len(nonzero) < 2:
        return {"ratio": 1.0,
                "lambda_max": float(abs_eigs.max()),
                "lambda_min": float(abs_eigs.min()),
                "eigenvalues": abs_eigs}

    lam_max = nonzero.max()
    lam_min = nonzero.min()
    ratio   = lam_max / lam_min

    odefunc.train()
    return {
        "ratio":       float(ratio),
        "lambda_max":  float(lam_max),
        "lambda_min":  float(lam_min),
        "eigenvalues": abs_eigs,
    }


# ─────────────────────────────────────────────────────────────
# 4.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def prepare_dataloaders(data_dict: dict,
                        train_split: float = 0.8,
                        batch_size: int = 16):
    trajs = torch.tensor(data_dict["trajectories"], dtype=torch.float32)
    t     = torch.tensor(data_dict["t_span"],        dtype=torch.float32)

    n_train  = int(len(trajs) * train_split)
    train_ds = TensorDataset(trajs[:n_train])
    val_ds   = TensorDataset(trajs[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, t, trajs


def train(model: NeuralODESurrogate,
          train_loader: DataLoader,
          val_loader:   DataLoader,
          t_span:       torch.Tensor,
          all_trajs:    torch.Tensor,
          n_epochs:     int   = 300,
          lr:           float = 3e-3,
          physics_weight: float = 1e-3,
          stiff_check_every: int = 50,
          solver: str = "implicit_adams") -> dict:
    """
    Full training loop with:
      • MSE trajectory loss
      • Mass-conservation penalty (mole fractions sum to 1)
      • Stiffness diagnostic every `stiff_check_every` epochs
    """

    model = model.to(DEVICE)
    t_span_dev = t_span.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = {
        "train_loss":    [],
        "val_loss":      [],
        "stiff_ratios":  [],   # (epoch, ratio)
        "stiff_epochs":  [],
    }

    print(f"\n{'='*60}")
    print(f"  Training Neural ODE  |  solver={solver}  |  epochs={n_epochs}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):

        # ── TRAIN ──────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for (batch,) in train_loader:
            batch      = batch.to(DEVICE)          # (B, T, S)
            y0         = batch[:, 0, :]             # (B, S)
            y_gt       = batch.permute(1, 0, 2)     # (T, B, S)

            optimizer.zero_grad()
            y_pred = model(y0, t_span_dev, solver)  # (T, B, S)

            # Trajectory MSE
            loss_traj = torch.mean((y_pred - y_gt) ** 2)

            # Physics: species mole fractions (cols 0..N_SPECIES-1) sum to 1
            mole_sum  = y_pred[..., :N_SPECIES].sum(dim=-1)   # (T, B)
            loss_cons = torch.mean((mole_sum - 1.0) ** 2)

            loss = loss_traj + physics_weight * loss_cons
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_train)

        # ── VALIDATE ────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch  = batch.to(DEVICE)
                y0     = batch[:, 0, :]
                y_gt   = batch.permute(1, 0, 2)
                y_pred = model(y0, t_span_dev, solver)
                val_loss += torch.mean((y_pred - y_gt) ** 2).item()
        avg_val = val_loss / len(val_loader)
        history["val_loss"].append(avg_val)

        # ── STIFFNESS DIAGNOSTIC ─────────────────────────────────
        if epoch % stiff_check_every == 0:
            # Pick a random sample from the full training set
            rand_idx     = np.random.randint(len(all_trajs))
            rand_t_idx   = np.random.randint(t_span.shape[0])
            y_sample     = all_trajs[rand_idx, rand_t_idx, :].to(DEVICE)

            diag = compute_stiffness_ratio(model.odefunc, y_sample)
            history["stiff_ratios"].append(diag["ratio"])
            history["stiff_epochs"].append(epoch)

            solver_advice = (
                "⚠  consider implicit_adams" if diag["ratio"] > 1000 else "✓ dopri5 suitable"
            )
            print(
                f"  Epoch {epoch:04d} | train={avg_train:.5f} | val={avg_val:.5f} "
                f"| λ_max/λ_min={diag['ratio']:8.1f} | {solver_advice}"
            )
        elif epoch % 10 == 0:
            print(f"  Epoch {epoch:04d} | train={avg_train:.5f} | val={avg_val:.5f}")

        scheduler.step()

    print(f"{'='*60}\n[TRAIN] Done.\n")
    return history


# ─────────────────────────────────────────────────────────────
# 5.  PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_results(model:      NeuralODESurrogate,
                 data_dict:  dict,
                 t_span:     torch.Tensor,
                 history:    dict,
                 n_profiles: int = 4,
                 save_path:  str = "neural_ode_results.png"):
    """
    Three-panel figure:
      (a) Training & validation loss curves
      (b) Stiffness ratio (λ_max/λ_min) over epochs
      (c) OH mole-fraction profiles: ground truth vs Neural ODE
          for n_profiles randomly selected initial conditions
    """
    model.eval()
    trajs    = data_dict["trajectories"]   # (N, T, S)
    t_ms     = data_dict["t_span"] * 1e3  # convert s → ms
    T0_vals  = data_dict["T0_values"]

    # ── Pick diverse T0 samples ──────────────────────────────
    sample_indices = np.linspace(0, len(trajs) - 1, n_profiles, dtype=int)

    # ── Run Neural ODE predictions ───────────────────────────
    t_dev = t_span.to(DEVICE)
    preds = []
    with torch.no_grad():
        for idx in sample_indices:
            y0    = torch.tensor(trajs[idx, 0:1, :],
                                 dtype=torch.float32, device=DEVICE)   # (1, S)
            y_hat = model(y0, t_dev, "implicit_adams")                         # (T, 1, S)
            preds.append(y_hat[:, 0, :].cpu().numpy())                 # (T, S)

    # ── Figure layout ────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor="#0f0f0f")
    gs  = gridspec.GridSpec(
        2, 2,
        hspace=0.38, wspace=0.32,
        left=0.07, right=0.96, top=0.92, bottom=0.08,
    )
    ax_loss   = fig.add_subplot(gs[0, 0])
    ax_stiff  = fig.add_subplot(gs[0, 1])
    ax_oh     = fig.add_subplot(gs[1, :])   # wide OH panel

    TEXT_C  = "#e8e8e8"
    GRID_C  = "#2a2a2a"
    COLORS  = ["#00d4ff", "#ff6b35", "#a8ff78", "#f7971e",
               "#c471ed", "#12c2e9"]

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors=TEXT_C, labelsize=10)
        ax.xaxis.label.set_color(TEXT_C)
        ax.yaxis.label.set_color(TEXT_C)
        ax.title.set_color(TEXT_C)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, color=GRID_C, linewidth=0.6, linestyle="--")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # ── (a) Loss curves ──────────────────────────────────────
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax_loss.semilogy(epochs, history["train_loss"],
                     color="#00d4ff", lw=1.8, label="Train loss")
    ax_loss.semilogy(epochs, history["val_loss"],
                     color="#ff6b35", lw=1.8, linestyle="--", label="Val loss")
    ax_loss.legend(facecolor="#222", edgecolor="#444",
                   labelcolor=TEXT_C, fontsize=10)
    style_ax(ax_loss, "(a)  Training & Validation Loss",
             "Epoch", "MSE Loss (log scale)")

    # ── (b) Stiffness ratio ───────────────────────────────────
    se = history["stiff_epochs"]
    sr = history["stiff_ratios"]
    ax_stiff.semilogy(se, sr, color="#a8ff78", lw=2,
                      marker="o", ms=6, mec="#0f0f0f", mew=1.5)
    ax_stiff.axhline(1000, color="#ff6b35", lw=1.2, linestyle=":",
                     label="Stiffness threshold (1000)")
    ax_stiff.legend(facecolor="#222", edgecolor="#444",
                    labelcolor=TEXT_C, fontsize=9)
    style_ax(ax_stiff, r"(b)  Stiffness Ratio  $|\lambda_{max}| / |\lambda_{min}|$",
             "Epoch", "Stiffness ratio (log scale)")

    # Annotate final ratio
    if sr:
        ax_stiff.annotate(
            f"Final: {sr[-1]:.1f}",
            xy=(se[-1], sr[-1]),
            xytext=(-50, 12), textcoords="offset points",
            color=TEXT_C, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=TEXT_C, lw=0.8),
        )

    # ── (c) OH profiles ───────────────────────────────────────
    for k, (idx, pred) in enumerate(zip(sample_indices, preds)):
        T0  = T0_vals[idx]
        col = COLORS[k % len(COLORS)]

        oh_gt   = trajs[idx, :, OH_IDX]      # ground truth
        oh_pred = pred[:, OH_IDX]            # Neural ODE

        ax_oh.plot(t_ms, oh_gt,
                   color=col, lw=2.0, alpha=0.9,
                   label=f"GT  T₀={T0:.0f} K")
        ax_oh.plot(t_ms, oh_pred,
                   color=col, lw=1.5, linestyle="--",
                   alpha=0.75, label=f"NODE T₀={T0:.0f} K")

    # Legend: separate GT vs NODE
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="white", lw=2.0,           label="Ground truth (Cantera)"),
        Line2D([0], [0], color="white", lw=1.5, ls="--",  label="Neural ODE prediction"),
    ]
    ax_oh.legend(
        handles=legend_elems,
        facecolor="#222", edgecolor="#444",
        labelcolor=TEXT_C, fontsize=10, loc="upper right",
    )
    # T0 color legend
    for k, idx in enumerate(sample_indices):
        T0 = T0_vals[idx]
        ax_oh.plot([], [], color=COLORS[k % len(COLORS)],
                   lw=2.5, label=f"T₀ = {T0:.0f} K")
    ax_oh.legend(
        facecolor="#222", edgecolor="#444",
        labelcolor=TEXT_C, fontsize=9,
        loc="upper left", ncol=2,
    )
    style_ax(ax_oh,
             "(c)  OH Mole Fraction: Ground Truth vs Neural ODE",
             "Time  [ms]", "OH mole fraction  [-]")

    # Overall title
    fig.suptitle(
        "Neural ODE Surrogate — H₂/O₂ Ignition Kinetics",
        fontsize=15, fontweight="bold", color=TEXT_C, y=0.97,
    )

    plt.savefig(save_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"[PLOT] Saved to '{save_path}'")


# ─────────────────────────────────────────────────────────────
# 6.  BENCHMARKING: Neural ODE vs Cantera wall-clock time
# ─────────────────────────────────────────────────────────────

def benchmark(model: NeuralODESurrogate,
              data_dict: dict,
              t_span: torch.Tensor,
              n_test_samples: int = 20):
    """
    Compare inference wall-clock time:
      Cantera (LSODA)  vs  Neural ODE (dopri5)
    """
    try:
        import cantera as ct
    except ImportError:
        print("[BENCH] Cantera not available — skipping benchmark.")
        return

    trajs   = data_dict["trajectories"]
    T0_vals = data_dict["T0_values"]
    t_end   = data_dict["t_span"][-1]
    n_pts   = len(data_dict["t_span"])

    test_idx = np.random.choice(len(trajs), n_test_samples, replace=False)

    # ── Cantera timing ────────────────────────────────────────
    gas = ct.Solution("h2o2.yaml")
    t0 = time.perf_counter()
    for idx in test_idx:
        T0 = T0_vals[idx]
        gas.set_equivalence_ratio(1.0, "H2", "O2:1, AR:3.76")
        gas.TP = T0, ct.one_atm
        r   = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([r])
        for t in np.linspace(0, t_end, n_pts):
            try:
                sim.advance(t)
            except Exception:
                pass
    cantera_time = (time.perf_counter() - t0) / n_test_samples

    # ── Neural ODE timing ─────────────────────────────────────
    model.eval()
    t_dev = t_span.to(DEVICE)

    y0_batch = torch.tensor(
        trajs[test_idx, 0, :], dtype=torch.float32, device=DEVICE
    )   # (n_test, S)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(y0_batch, t_dev, "implicit_adams")
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    node_time = (time.perf_counter() - t0) / n_test_samples

    speedup = cantera_time / node_time
    print("\n" + "─" * 50)
    print("  BENCHMARK RESULTS")
    print("─" * 50)
    print(f"  Cantera (per sample):    {cantera_time*1e3:8.2f} ms")
    print(f"  Neural ODE (per sample): {node_time*1e3:8.2f} ms")
    print(f"  Speedup:                 {speedup:8.1f}×")
    print("─" * 50)
    if speedup < 1:
        print("  NOTE: Neural ODE slower — network not yet competitive.")
        print("        This is expected early in training / on CPU.")
    else:
        print(f"  Neural ODE is {speedup:.1f}× faster than Cantera LSODA.")
    print()


# ─────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────

def main():

    # ── 7.1  Generate / Load Data ─────────────────────────────
    data = generate_ignition_data(
        T0_range=(900, 1400),
        n_samples=120,
        t_end=1e-3,
        n_points=300,
    )
    trajs  = data["trajectories"]
    t_span = torch.tensor(data["t_span"], dtype=torch.float32)

    print(f"\n[DATA] Shape: {trajs.shape}   "
          f"(samples × time-points × state_dim)")
    print(f"[DATA] Species order: {data['species']}")
    print(f"[DATA] OH index: {OH_IDX}  →  '{data['species'][OH_IDX]}'")

    # ── 7.2  Dataloaders ──────────────────────────────────────
    train_loader, val_loader, t_span, all_trajs = prepare_dataloaders(
        data, train_split=0.8, batch_size=16
    )

    # ── 7.3  Build Model ──────────────────────────────────────
    model = NeuralODESurrogate(
        state_dim  = STATE_DIM,
        hidden_dim = 128,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parameters: {n_params:,}")

    # ── 7.4  Train ────────────────────────────────────────────
    history = train(
        model, train_loader, val_loader, t_span, all_trajs,
        n_epochs          = 300,
        lr                = 3e-3,
        physics_weight    = 1e-3,
        stiff_check_every = 50,
        solver            = "implicit_adams",
    )

    # ── 7.5  Plot ─────────────────────────────────────────────
    plot_results(
        model, data, t_span, history,
        n_profiles = 4,
        save_path  = "neural_ode_results.png",
    )

    # ── 7.6  Benchmark ────────────────────────────────────────
    benchmark(model, data, t_span, n_test_samples=20)

    # ── 7.7  Save checkpoint ──────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "history":     history,
    }, "neural_ode_checkpoint.pt")
    print("[SAVE] Checkpoint saved to 'neural_ode_checkpoint.pt'")


if __name__ == "__main__":
    main()
