#!/usr/bin/env python3
"""KL divergence of q-marginal vs UnderdampedLandauer target,
evaluated on the perturbative fine q-grid (~1910 points) for fair comparison."""

import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

JOINT_DIR = "joint1"
T_FINAL   = 5.0
N_NORM    = 100_000


def target_pdf(q):
    return np.exp(-((q**2 - 1)**2))


def make_target(q_grid):
    q_fine = np.linspace(-8, 8, N_NORM)
    norm = np.trapezoid(target_pdf(q_fine), q_fine)
    return target_pdf(q_grid) / norm


def kl_divergence(p, q):
    eps = 1e-300
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def extract_m(path):
    m = re.search(r"_[mM](\d+)", path)
    return int(m.group(1)) if m else 0


def load_pert_grid(pert_path, T=T_FINAL):
    """Return (q_fine, f_fine) from a perturbative CSV at time T."""
    data = np.genfromtxt(pert_path, delimiter=",", comments="#")
    ts   = np.unique(data[:, 0])
    t    = ts[np.argmin(np.abs(ts - T))]
    sub  = data[np.isclose(data[:, 0], t)]
    idx  = np.argsort(sub[:, 1])
    return sub[idx, 1], sub[idx, 2]


def joint_marginal_on_grid(joint_path, q_grid, T=T_FINAL):
    """Interpolate the joint q-marginal (coarse) onto q_grid."""
    data = np.genfromtxt(joint_path, delimiter=",", skip_header=1)
    ts   = np.unique(data[:, 0])
    t    = ts[np.argmin(np.abs(ts - T))]
    sub  = data[np.isclose(data[:, 0], t)]
    qs   = np.unique(sub[:, 1])
    ps   = np.unique(sub[:, 2])
    dp   = ps[1] - ps[0]
    marginal = sub[:, 3].reshape(len(qs), len(ps)).sum(axis=1) * dp
    interp = interp1d(qs, marginal, kind="linear", bounds_error=False, fill_value=0.0)
    return np.maximum(interp(q_grid), 0.0)


def compute_kls(joint_path, pert_path):
    """Return (kl_joint, kl_pert) both on the perturbative fine q-grid."""
    q_fine, f_pert = load_pert_grid(pert_path)
    target         = make_target(q_fine)
    marginal       = joint_marginal_on_grid(joint_path, q_fine)
    return kl_divergence(marginal, target), kl_divergence(f_pert, target)


def main():
    joint_files = sorted(
        glob.glob(f"{JOINT_DIR}/joint_distribution_m*.csv"),
        key=extract_m,
    )
    ms = [extract_m(p) for p in joint_files]

    results = [
        compute_kls(p, f"{JOINT_DIR}/perturbative_underdamped_m{m}.csv")
        for p, m in zip(joint_files, ms)
    ]
    kls_joint, kls_pert = zip(*results)

    trained_files = sorted(
        glob.glob(f"{JOINT_DIR}/joint_distribution_trained_[mM]*.csv"),
        key=extract_m,
    )
    ms_trained  = [extract_m(p) for p in trained_files]
    x_trained   = [ms.index(m) for m in ms_trained if m in ms]
    kls_trained = [
        kl_divergence(
            joint_marginal_on_grid(p, load_pert_grid(
                f"{JOINT_DIR}/perturbative_underdamped_m{m}.csv")[0]),
            make_target(load_pert_grid(
                f"{JOINT_DIR}/perturbative_underdamped_m{m}.csv")[0]),
        )
        for p, m in zip(trained_files, ms_trained)
    ]

    kl_joint_full, kl_pert_full = compute_kls(
        f"{JOINT_DIR}/joint_distribution_full.csv",
        f"{JOINT_DIR}/perturbative_underdamped.csv",
    )

    x_ticks = list(range(len(ms))) + [len(ms)]
    x_labels = [str(m) for m in ms] + ["full"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_ticks[:-1], kls_joint,   marker="o", color="steelblue",   label="joint marginal")
    ax.plot(x_ticks[-1],  kl_joint_full, marker="s", color="steelblue")
    ax.plot(x_ticks[:-1], kls_pert,    marker="o", color="tomato",      label="perturbative")
    ax.plot(x_ticks[-1],  kl_pert_full,  marker="s", color="tomato")
    ax.plot(x_trained,    kls_trained,  marker="o", color="forestgreen", label="trained")
    ax.set_yscale("log")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("M")
    ax.set_ylabel(r"$\mathrm{KL}(\rho_q^{T} \| \rho_\mathrm{target})$")
    ax.set_title(r"KL divergence at $t=T$ vs target (fine q-grid)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = "kl_target_finegrid.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
