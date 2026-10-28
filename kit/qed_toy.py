#!/usr/bin/env python3
"""Local analytical cavity-QED toy. No network, Tidy3D, FDTD or cloud operations.

Time is in ns. Internally every frequency/rate is in ns^-1. A value supplied as
g/(2 pi), kappa/(2 pi), gamma_phi/(2 pi), or Delta/(2 pi) in MHz is multiplied
by 2*pi*1e-3. gamma_1=1/lifetime_ns is a population rate, not gamma_1/(2*pi).

The retained Hilbert space contains one excitation: emitter and one cavity
state, or emitter plus CW and CCW. Loss jumps end in an omitted common ground
state; their probabilities are counted explicitly. Pure dephasing remains in
the retained block. This is exact for this undriven, Markovian one-excitation
model, and does not describe re-excitation, saturation, or multiphoton states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

REVISION = "qed-toy-r001"
COLORS = ["#4A1F7A", "#B2182B", "#007A50", "#C83E8C", "#0072B2", "#D1492E", "#007F86"]


def angular_ns(value_mhz: float) -> float:
    """Convert a quoted angular-rate/(2*pi) in MHz to angular ns^-1."""
    return 2 * np.pi * 1e-3 * value_mhz


@dataclass
class Model:
    hamiltonian: np.ndarray
    channels: dict[str, np.ndarray]
    gamma_phi: float
    labels: tuple[str, ...]

    @property
    def n(self):
        return self.hamiltonian.shape[0]

    @property
    def loss(self):
        return sum(self.channels.values(), np.zeros((self.n, self.n), dtype=complex))

    def liouvillian(self):
        """Column-major vec: vec(A rho B)=(B.T kron A) vec(rho)."""
        n = self.n
        eye = np.eye(n, dtype=complex)
        h = self.hamiltonian
        k = self.loss
        z = np.diag([1.0] + [-1.0] * (n - 1))
        return (-1j * (np.kron(eye, h) - np.kron(h.T, eye))
                - 0.5 * (np.kron(eye, k) + np.kron(k.T, eye))
                + 0.5 * self.gamma_phi * (np.kron(z.T, z) - np.eye(n*n)))

    def initial(self):
        rho = np.zeros((self.n, self.n), dtype=complex)
        rho[0, 0] = 1.0
        return rho


def make_model(g, gamma1, kappa_i, kappa_ex, gamma_phi=0.0, delta=0.0,
               q=0.8, s=0.3, ring=False, directionality=0.0,
               backscatter=0.0, relative_phase=0.0):
    """H/ħ in ns^-1; Delta=omega_c-omega_e; emitter diagonal is zero.

    For the ring, g is the bright coupling: |g_CW|^2+|g_CCW|^2=g^2.
    directionality D sets those fractions to (1+D)/2 and (1-D)/2.
    D is an explicit hypothetical coupling asymmetry, not a prediction for a
    real linearly polarized molecule. backscatter J couples the traveling modes.
    """
    values = (g, gamma1, kappa_i, kappa_ex, gamma_phi)
    if any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("Couplings and loss/dephasing rates must be finite and nonnegative.")
    if not (0 <= q <= 1 and 0 <= s <= 1 and -1 <= directionality <= 1):
        raise ValueError("q, s must be in [0,1]; directionality must be in [-1,1].")
    n = 3 if ring else 2
    h = np.diag([0.] + [delta] * (n-1)).astype(complex)
    if ring:
        gs = [g*np.sqrt((1+directionality)/2),
              g*np.sqrt((1-directionality)/2)*np.exp(1j*relative_phase)]
        h[1, 2] = backscatter
        h[2, 1] = np.conj(backscatter)
    else:
        gs = [g]
    for j, coupling in enumerate(gs, 1):
        h[j, 0] = coupling
        h[0, j] = np.conj(coupling)
    channels = {}
    for name, rate in [("background_zpl", gamma1*q*s),
                       ("background_sideband", gamma1*q*(1-s)),
                       ("nonradiative", gamma1*(1-q))]:
        v = np.zeros(n); v[0] = rate
        channels[name] = np.diag(v).astype(complex)
    v = np.array([0.] + [kappa_i] * (n-1))
    channels["intrinsic"] = np.diag(v).astype(complex)
    if ring:
        channels["bus_forward"] = np.diag([0., kappa_ex, 0.]).astype(complex)
        channels["bus_backward"] = np.diag([0., 0., kappa_ex]).astype(complex)
    else:
        channels["bus"] = np.diag([0., kappa_ex]).astype(complex)
    return Model(h, channels, gamma_phi,
                 ("emitter", "CW", "CCW") if ring else ("emitter", "cavity"))


def from_config(cfg, **overrides):
    args = dict(g=angular_ns(cfg["g_bright_over_2pi_MHz"]),
                gamma1=1/cfg["lifetime_ns"],
                kappa_i=angular_ns(cfg["kappa_intrinsic_over_2pi_MHz"]),
                kappa_ex=angular_ns(cfg["kappa_external_over_2pi_MHz"]),
                gamma_phi=angular_ns(cfg["gamma_phi_over_2pi_MHz"]),
                delta=angular_ns(cfg["detuning_over_2pi_MHz"]),
                q=cfg["reference_quantum_yield"],
                s=cfg["zpl_fraction_of_radiative_decay"])
    args.update(overrides)
    return make_model(**args)


def integrated(model):
    """Integrate rho to infinity by solving -L vec(X)=vec(rho_0)."""
    L = model.liouvillian()
    v = np.linalg.solve(-L, model.initial().reshape(-1, order="F"))
    X = v.reshape((model.n, model.n), order="F")
    probabilities = {name: float(np.trace(k @ X).real)
                     for name, k in model.channels.items()}
    return probabilities, X


def closed_single(g, gamma1, kappa_i, kappa_ex, gamma_phi=0., delta=0.):
    """Independent scalar result, including reabsorption and pure dephasing."""
    kappa = kappa_i + kappa_ex
    A = (kappa + gamma1)/2 + gamma_phi
    R = 2*g*g*A/(A*A + delta*delta)
    denominator = kappa*gamma1 + (kappa+gamma1)*R
    pcav = kappa*R/denominator if denominator else 0.0
    pbus = kappa_ex/kappa*pcav if kappa else 0.0
    return pbus, pcav


def dynamics(model, times):
    """DOP853 propagation, with one cumulative probability per loss channel."""
    L = model.liouvillian()
    channels = list(model.channels)
    n2 = model.n**2
    y0 = np.r_[model.initial().reshape(-1, order="F"), np.zeros(len(channels))].astype(complex)
    def rhs(t, y):
        rho = y[:n2].reshape((model.n, model.n), order="F")
        jumps = [np.trace(model.channels[key] @ rho).real for key in channels]
        return np.r_[L @ y[:n2], jumps]
    solution = solve_ivp(rhs, (times[0], times[-1]), y0, t_eval=times,
                         method="DOP853", rtol=2e-10, atol=2e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    rhos = solution.y[:n2].T.reshape((-1, model.n, model.n), order="C")
    # Each row was a column-major flattened matrix. Swap matrix indices back.
    rhos = rhos.transpose(0, 2, 1)
    counters = {key: solution.y[n2+j].real for j, key in enumerate(channels)}
    populations = np.diagonal(rhos, axis1=1, axis2=2).real
    return populations, counters, rhos


def csv_write(path, headers, columns):
    with path.open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(headers)
        writer.writerows(zip(*columns))


def style():
    font_path = font_manager.findfont("Trebuchet MS", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Trebuchet MS", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9,
        "mathtext.fontset": "cm", "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=COLORS), "lines.linewidth": 2.2,
        "axes.linewidth": .8, "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300})
    return font_path


def finish(fig, stem, figure_dir):
    fig.tight_layout(pad=1.0)
    result = {}
    for ext in ("png", "pdf", "svg"):
        path = figure_dir/f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        result[ext] = str(path)
    plt.close(fig)
    return result


def mark_line(ax, x, y, label, color=None, **kwargs):
    return ax.plot(x, y, label=label, color=color, marker="o", markersize=3.2,
                   markevery=max(1, len(x)//12), **kwargs)


def run_checks(cfg):
    rng = np.random.default_rng(cfg["random_seed"])
    rows = []
    finite_max = positivity_min = hermiticity_max = expm_max = 0.
    for i in range(cfg["random_check_cases"]):
        rates = dict(g=float(rng.uniform(.03, 3)), gamma1=float(rng.uniform(.05, 1.0)),
            kappa_i=float(rng.uniform(.05, 2)), kappa_ex=float(rng.uniform(.02, 4)),
            gamma_phi=float(rng.uniform(0, 2)), delta=float(rng.uniform(-4, 4)))
        model = make_model(**rates)
        p, X = integrated(model)
        pbus, pcav = closed_single(**rates)
        err = abs(p["bus"]-pbus)
        residual = abs(sum(p.values())-1)
        assert err < 2e-12 and residual < 2e-12
        assert min(p.values()) >= -2e-12 and max(p.values()) <= 1+2e-12
        finite_error = None
        if i < 8:
            t = np.linspace(0, 20, 121)
            pops, counters, rhos = dynamics(model, t)
            finite_error = float(np.max(np.abs(pops.sum(axis=1)+sum(counters.values())-1)))
            finite_max = max(finite_max, finite_error)
            hermiticity_max = max(hermiticity_max,
                float(np.max(np.abs(rhos-rhos.conj().transpose(0, 2, 1)))))
            positivity_min = min(positivity_min, float(np.linalg.eigvalsh(rhos).min()))
            exact = (expm(model.liouvillian()*t[-1]) @
                     model.initial().reshape(-1, order="F")).reshape((2, 2), order="F")
            expm_error = float(np.max(np.abs(exact-rhos[-1])))
            expm_max = max(expm_max, expm_error)
            assert finite_error < 2e-9 and expm_error < 2e-9
        rows.append({"case": i, **rates, "p_bus_liouville": p["bus"],
            "p_bus_scalar": pbus, "closed_formula_abs_error": err,
            "probability_conservation_error": residual,
            "finite_conservation_error": finite_error})
    base = dict(g=.4, gamma1=.2, kappa_i=.5, kappa_ex=1.5, gamma_phi=.1)
    p0, _ = integrated(make_model(**{**base, "g": 0.}))
    pex0, _ = integrated(make_model(**{**base, "kappa_ex": 0.}))
    assert abs(p0["bus"]) < 1e-14 and abs(pex0["bus"]) < 1e-14
    assert abs(sum(p0.values())-1) < 1e-12
    # Equal CW/CCW couplings at J=0 reduce to one bright cavity, with fixed total g^2.
    ps, _ = integrated(make_model(**base))
    pr, _ = integrated(make_model(**base, ring=True))
    bright_error = abs(pr["bus_forward"]+pr["bus_backward"]-ps["bus"])
    assert bright_error < 1e-12
    tr = np.linspace(0, 25, 201)
    rp, rc, rr = dynamics(make_model(**base, ring=True, directionality=.6,
                                  backscatter=.8, relative_phase=.3), tr)
    ring_error = float(np.max(np.abs(rp.sum(axis=1)+sum(rc.values())-1)))
    assert ring_error < 2e-9
    assert positivity_min > -2e-9 and hermiticity_max < 2e-9
    return {"status": "passed", "random_cases": rows,
        "max_closed_formula_abs_error": max(row["closed_formula_abs_error"] for row in rows),
        "max_integrated_conservation_error": max(row["probability_conservation_error"] for row in rows),
        "max_finite_counter_conservation_error": finite_max,
        "max_finite_vs_matrix_exponential_error": expm_max,
        "minimum_density_eigenvalue": positivity_min,
        "max_hermiticity_error": hermiticity_max,
        "zero_g_bus_probability": p0["bus"], "zero_external_bus_probability": pex0["bus"],
        "ring_bright_mode_reduction_error": bright_error,
        "ring_finite_counter_conservation_error": ring_error,
        "scope": "Algebraic/numerical consistency of this finite toy; no physical device validation."}


def build(cfg, out, figure_dir):
    out.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    font_path = style()
    manifest = []
    def save(fig, stem, caption, parameters):
        files = finish(fig, stem, figure_dir)
        manifest.append({"id": stem, "files": files, "caption": caption,
                         "parameters": parameters, "evidence_class": "illustrative analytical toy"})

    # 1. Spatial/dipolar coupling factors; amplitude decay length explicitly declared.
    angle = np.linspace(0, 180, 241); height = np.linspace(0, 400, 241)
    orient = np.cos(np.deg2rad(angle))**2
    ell = cfg["evanescent_field_amplitude_length_nm"]
    height_factor = np.exp(-2*height/ell)
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    mark_line(ax[0], angle, orient, r"$|g|^2/|g_{\max}|^2=\cos^2\vartheta$")
    ax[0].set(xlabel=r"Dipole–field angle $\vartheta$ (degrees)", ylabel="Normalized coupling squared",
              title="(a) Orientation", xlim=(0, 180), ylim=(0, 1.05))
    mark_line(ax[1], height, height_factor, r"$|g(z)|^2/|g(0)|^2=e^{-2z/\ell}$")
    ax[1].text(.96, .9, rf"Field-amplitude length $\ell={ell:g}$ nm", ha="right", transform=ax[1].transAxes)
    ax[1].set(xlabel="Height above the reference surface (nm)", ylabel="Normalized coupling squared",
              title="(b) Evanescent field", xlim=(0, 400), ylim=(0, 1.05))
    csv_write(out/"position_orientation.csv", ["angle_deg", "orientation_g_squared", "height_nm", "height_g_squared"],
              [angle, orient, height, height_factor])
    save(fig, "qed01-position-orientation",
         "A real dipole gives a cos² orientation factor. An assumed field amplitude exp(−z/ell), with ell=100 nm, gives coupling squared exp(−2z/ell). These factors do not come from a calculated LN field.",
         {"field_amplitude_decay_length_nm": ell})

    # 2. Branch capture in the irreversible additional-channel limit.
    F = np.logspace(-2, 3, 241)
    q = cfg["reference_quantum_yield"]; s = cfg["zpl_fraction_of_radiative_decay"]
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.5))
    columns = [F]; headers = ["additional_zpl_F"]
    for qi, si in [(1., 1.), (q, s), (.4, .1)]:
        beta = F*qi*si/(1+F*qi*si)
        mark_line(ax[0], F, beta, f"q={qi:g}, s={si:g}")
        columns.append(beta); headers.append(f"capture_q{qi:g}_s{si:g}")
    beta = F*q*s/(1+F*q*s)
    q_new = q*(1+s*F)/(1+q*s*F)
    zpl_rad = s*(1+F)/(1+s*F)
    for vals, label in [(beta, "Cavity capture"), (q_new, "Total radiative yield"), (zpl_rad, "ZPL fraction of fluorescence")]:
        mark_line(ax[1], F, vals, label)
    for a in ax:
        a.set(xscale="log", xlabel=r"Additional ZPL enhancement $F_{\rm extra}$", ylim=(0, 1.05))
        a.legend(loc="lower right")
    ax[0].set(ylabel="Probability per initial excitation", title="(a) Branching controls cavity capture")
    ax[1].set(ylabel="Probability or conditional fraction", title=f"(b) Different denominators: q={q:g}, s={s:g}")
    csv_write(out/"molecular_branching.csv", headers+["default_new_radiative_yield", "default_zpl_fraction_of_fluorescence"],
              columns+[q_new, zpl_rad])
    save(fig, "qed02-molecular-branching",
         "Irreversible additional-channel model: Gamma_c=F_extra s q gamma_1^0. Cavity capture, total radiative yield, and the ZPL fraction of fluorescence use different denominators. F_extra denotes added ZPL decay; it is not the total ZPL enhancement.",
         {"q": q, "s": s, "approximation": "irreversible added cavity channel; no coherent reabsorption"})

    # 3. Compare two declared dynamical regimes using the full excited-block Liouvillian.
    strong_cfg = {**cfg, **cfg["strong_scenario"]}
    scenarios = [("Default", cfg, 15.), ("Coherent exchange example", strong_cfg, 8.)]
    fig, ax = plt.subplots(2, 2, figsize=(8.5, 6.0), sharex="col")
    scenario_summary = {}
    for col, (name, pars, end) in enumerate(scenarios):
        model = from_config(pars); t = np.linspace(0, end, 1401)
        populations, counters, rhos = dynamics(model, t)
        background = sum(counters[key] for key in ("background_zpl", "background_sideband", "nonradiative"))
        for j, lab in enumerate([r"Emitter $p_e$", r"Cavity $p_c$"]):
            ax[0, col].plot(t, populations[:, j], label=lab, color=COLORS[j])
        for vals, lab, color in [(counters["bus"], "Bus", COLORS[0]),
                                  (counters["intrinsic"], "Intrinsic cavity loss", COLORS[1]),
                                  (background, "Other emitter decay", COLORS[2])]:
            ax[1, col].plot(t, vals, label=lab, color=color)
        ax[0, col].set(title=f"({'a' if col == 0 else 'b'}) {name}", ylabel="Excitation population", ylim=(-.02, 1.03))
        ax[1, col].set(xlabel="Time after excitation (ns)", ylabel="Cumulative channel probability", ylim=(-.02, 1.03))
        for row in (0, 1): ax[row, col].legend(loc="upper right")
        p_inf, _ = integrated(model)
        balance_error = float(np.max(np.abs(populations.sum(axis=1)+sum(counters.values())-1)))
        csv_write(out/f"dynamics_{'default' if col == 0 else 'coherent'}.csv",
                  ["time_ns", "p_emitter", "p_cavity"]+list(counters),
                  [t, populations[:, 0], populations[:, 1]]+list(counters.values()))
        scenario_summary[name] = {"parameters": pars, "probabilities_infinite_time": p_inf,
            "finite_time_ns": end, "finite_remaining_excitation": float(populations[-1].sum()),
            "finite_counter_conservation_error": balance_error}
    save(fig, "qed03-excitation-dynamics",
         "Exact dynamics within the Markovian one-excitation model, initially |e,0>. Top: emitter and cavity populations. Bottom: cumulative loss probabilities. Default: (g,kappa_i,kappa_ex,gamma_phi)/(2pi)=(100,100,500,30) MHz. Coherent example: (1000,30,70,1) MHz. Both have gamma_1=0.2 ns^-1 and Delta=0. Residual population plus all counters remains one.",
         {"scenarios": scenario_summary})

    # 4. Bus loading at fixed g, intrinsic loss, population decay and dephasing.
    load = np.logspace(-.3, 4.3, 281)
    p_bus = []; p_cav = []; escape = []
    for val in load:
        m = from_config(cfg, kappa_ex=angular_ns(val)); p, _ = integrated(m)
        p_bus.append(p["bus"]); p_cav.append(p["bus"]+p["intrinsic"])
        escape.append(val/(val+cfg["kappa_intrinsic_over_2pi_MHz"]))
    result = minimize_scalar(lambda x: -integrated(from_config(cfg, kappa_ex=angular_ns(np.exp(x))))[0]["bus"],
                             bounds=(np.log(load[0]), np.log(load[-1])), method="bounded")
    optimum = float(np.exp(result.x)); optimum_p = float(-result.fun)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for vals, lab in [(p_bus, "Emission into bus"), (p_cav, "Emission through either cavity channel"),
                      (escape, r"Conditional cavity escape $\kappa_{ex}/\kappa$")]:
        mark_line(ax, load, vals, lab)
    ax.axvline(cfg["kappa_intrinsic_over_2pi_MHz"], color="#555555", linestyle="--", linewidth=1.2,
               label="Empty-ring critical coupling")
    ax.plot(optimum, optimum_p, "o", color=COLORS[0], markersize=7)
    ax.annotate(f"Maximum bus probability {optimum_p:.3f}\nat {optimum:.0f} MHz", (optimum, optimum_p),
                xytext=(24, 18), textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": COLORS[0]})
    ax.set(xscale="log", xlabel=r"External energy-decay rate $\kappa_{ex}/(2\pi)$ (MHz)",
           ylabel="Probability per initial excitation", ylim=(0, 1.04))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.20), frameon=False)
    csv_write(out/"bus_loading.csv", ["kappa_ex_over_2pi_MHz", "P_bus", "P_cavity_all", "conditional_escape"],
              [load, p_bus, p_cav, escape])
    save(fig, "qed04-bus-loading",
         "Increasing external loading improves the probability that a cavity photon exits into the bus, while it broadens the cavity and reduces emitter capture. The maximum exact single-excitation bus probability need not coincide with empty-ring critical coupling. g, intrinsic loss, gamma_1 and gamma_phi are held at their default values.",
         {"optimum_kappa_ex_over_2pi_MHz": optimum, "optimum_P_bus": optimum_p,
          "critical_kappa_ex_over_2pi_MHz": cfg["kappa_intrinsic_over_2pi_MHz"]})

    # 5. Detuning/dephasing. No indistinguishability is inferred from these probabilities.
    detun = np.linspace(-3, 3, 241)
    dephasing = np.logspace(-2, 4, 241)
    fig, ax = plt.subplots(1, 2, figsize=(8.7, 3.6))
    cols = [detun]; heads = ["detuning_over_2pi_GHz"]
    for gp in [0., 30., 300., 3000.]:
        vals = [integrated(from_config(cfg, gamma_phi=angular_ns(gp), delta=angular_ns(d*1000)))[0]["bus"] for d in detun]
        mark_line(ax[0], detun, vals, f"{gp:g} MHz")
        cols.append(vals); heads.append(f"P_bus_gamma_phi_over_2pi_{gp:g}MHz")
    csv_write(out/"detuning.csv", heads, cols)
    cols = [dephasing]; heads = ["gamma_phi_over_2pi_MHz"]
    for d in [0., .6, 2.]:
        vals = [integrated(from_config(cfg, gamma_phi=angular_ns(gp), delta=angular_ns(d*1000)))[0]["bus"] for gp in dephasing]
        mark_line(ax[1], dephasing, vals, f"{d:g} GHz")
        cols.append(vals); heads.append(f"P_bus_detuning_over_2pi_{d:g}GHz")
    csv_write(out/"dephasing.csv", heads, cols)
    ax[0].set(xlabel=r"Detuning $\Delta/(2\pi)$ (GHz)", ylabel="Exact bus-emission probability",
              title=r"(a) Vary $\gamma_\phi/(2\pi)$", ylim=(0, .7))
    ax[1].set(xscale="log", xlabel=r"Pure-dephasing rate $\gamma_\phi/(2\pi)$ (MHz)",
              ylabel="Exact bus-emission probability", title=r"(b) Vary $\Delta/(2\pi)$", ylim=(0, .7))
    ax[0].legend(title=r"$\gamma_\phi/(2\pi)$", loc="upper right")
    ax[1].legend(title=r"$\Delta/(2\pi)$", loc="upper right")
    save(fig, "qed05-detuning-dephasing",
         "Exact integrated emission probability for the same one-excitation model. Pure dephasing reduces resonant transfer, but can increase capture for a detuned emitter by broadening its spectral access to the cavity. This plot calculates probabilities, not photon indistinguishability or a molecular temperature dependence.",
         {"detuning_range_over_2pi_GHz": [-3, 3], "gamma_phi_over_2pi_MHz": [0, 30, 300, 3000]})

    # 6. CW/CCW and coherent backscatter at fixed total coupling squared.
    D = np.linspace(-1, 1, 161); J = np.linspace(0, 600, 181)
    fig, ax = plt.subplots(1, 2, figsize=(8.7, 3.7))
    cols = [D]; heads = ["hypothetical_directionality_D"]
    dp = [integrated(from_config(cfg, ring=True, directionality=d))[0] for d in D]
    jp = [integrated(from_config(cfg, ring=True, directionality=1., backscatter=angular_ns(j)))[0] for j in J]
    for col, xx, rows in [(0, D, dp), (1, J, jp)]:
        forward = np.array([p["bus_forward"] for p in rows]); backward = np.array([p["bus_backward"] for p in rows])
        for vals, label in [(forward, "Forward bus"), (backward, "Backward bus"), (forward+backward, "Both bus directions")]:
            mark_line(ax[col], xx, vals, label)
        csv_write(out/("ring_directionality.csv" if col == 0 else "ring_backscatter.csv"),
                  ["directionality_D" if col == 0 else "J_over_2pi_MHz", "P_forward", "P_backward", "P_bus_total"],
                  [xx, forward, backward, forward+backward])
        ax[col].set(ylabel="Probability per initial excitation", ylim=(0, .8))
        ax[col].legend(loc="upper right")
    ax[0].set(xlabel=r"Assumed coupling asymmetry $D$", title="(a) J=0; total coupling fixed")
    ax[1].set(xlabel=r"Backscatter coupling $|J|/(2\pi)$ (MHz)", title="(b) Initially directional coupling: D=1")
    save(fig, "qed06-direction-backscatter",
         "Two traveling ring modes with |g_CW|²+|g_CCW|²=g_B². D sets the hypothetical directional coupling fractions (1±D)/2. The real linear-dipole reciprocal baseline is D=0. At D=1, coherent CW–CCW backscatter redistributes emission between bus directions and shifts the standing-wave resonances. Both cavity modes have the same default loss rates; Delta=0, J is real, and the relative dipole-coupling phase is zero.",
         {"g_bright_over_2pi_MHz": cfg["g_bright_over_2pi_MHz"], "backscatter_J_over_2pi_MHz_range": [0, 600],
          "D_for_backscatter": 1, "relative_coupling_phase_rad": 0})

    # 7. Keep emitter capture and downstream passive losses in a single probability budget.
    default_p, X = integrated(from_config(cfg))
    eta_prop = cfg["eta_propagation"]; eta_gc = cfg["eta_grating_to_selected_mode"]; eta_det = cfg["eta_detector"]
    stages = np.array([default_p["bus"], default_p["bus"]*eta_prop,
                       default_p["bus"]*eta_prop*eta_gc,
                       default_p["bus"]*eta_prop*eta_gc*eta_det])
    terminal = {"Other emitter decay": sum(default_p[k] for k in ["background_zpl", "background_sideband", "nonradiative"]),
                "Intrinsic cavity loss": default_p["intrinsic"],
                "Downstream optical loss": default_p["bus"]-stages[2],
                "Detector miss": stages[2]-stages[3], "Detected": stages[3]}
    fig, ax = plt.subplots(1, 2, figsize=(8.7, 3.7), gridspec_kw={"width_ratios": [1, 1.1]})
    ax[0].bar(np.arange(4), stages, color=[COLORS[0], COLORS[1], COLORS[2], COLORS[4]], width=.62)
    for j, value in enumerate(stages): ax[0].text(j, value+.012, f"{value:.3f}", ha="center")
    ax[0].set(xticks=np.arange(4), xticklabels=["Bus", "After\npropagation", "Selected\nfiber mode", "Detected"],
              ylabel="Probability per initial excitation", ylim=(0, .65), title="(a) Conditional factors multiply")
    bottom = 0
    for j, (label, probability) in enumerate(terminal.items()):
        ax[1].barh([0], probability, left=bottom, color=COLORS[j], label=f"{label}: {probability:.3f}", height=.5)
        bottom += probability
    ax[1].set(xlim=(0, 1), ylim=(-.5, 1), yticks=[], xlabel="Probability per initial excitation",
              title="(b) Mutually exclusive final channels")
    ax[1].legend(loc="upper center", frameon=False, bbox_to_anchor=(.5, .98))
    csv_write(out/"collection_stages.csv", ["stage", "probability_per_initial_excitation"],
              [["bus", "after_propagation", "selected_fiber_mode", "detected"], stages])
    save(fig, "qed07-collection-budget",
         "The default single-cavity-mode case with one bus output, followed by illustrative passive efficiencies eta_prop=0.9, eta_GC=0.35 into one selected fiber mode, and eta_det=0.7. Every bar is per initial emitter excitation. A one-way cascade is assumed, without optical feedback from the grating. For the equally coupled CW/CCW ring with only one collected arm, the bus and detected probabilities are each half these values. The final-channel probabilities sum to one; the chosen grating efficiency is an input, not an FDTD result.",
         {"eta_propagation": eta_prop, "eta_grating": eta_gc, "eta_detector": eta_det,
          "stage_probabilities": stages.tolist(), "terminal_channels": terminal})

    checks = run_checks(cfg)
    (out/"checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    csv_write(out/"random_scalar_checks.csv", list(checks["random_cases"][0]),
              [[row[key] for row in checks["random_cases"]] for key in checks["random_cases"][0]])
    (out/"parameters_used.json").write_text(json.dumps(cfg, indent=2)+"\n")
    summary = {"model_revision": REVISION, "config": cfg,
        "conventions": {"time": "ns", "internal_rates": "ns^-1", "detuning": "omega_c-omega_e",
            "hamiltonian_single": "H/hbar=[[0,g],[g,Delta]] for real g",
            "kappa": "energy / photon-number decay; amplitude decay kappa/2",
            "gamma1": "excited population decay", "gamma_phi": "pure optical coherence decay",
            "gamma2": "gamma1/2+gamma_phi", "ring_g": "|g_CW|^2+|g_CCW|^2=g_bright^2"},
        "default_probabilities": default_p, "default_mean_first_loss_time_ns": float(np.trace(X).real),
        "default_collection_stages": dict(zip(["bus", "after_propagation", "fiber", "detected"], stages.tolist())),
        "symmetric_ring_one_collected_arm": {"P_bus_one_arm": default_p["bus"]/2,
            "P_detected_one_arm": float(stages[-1])/2,
            "assumptions": "D=0, J=0, same total bright coupling and independent losses; only one bus arm collected"},
        "bus_loading_optimum": {"kappa_external_over_2pi_MHz": optimum, "probability": optimum_p},
        "dynamics_scenarios": scenario_summary, "figures": manifest,
        "checks": {key: value for key, value in checks.items() if key != "random_cases"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "font_path": font_path,
            "math_font": "Matplotlib Computer Modern mathtext"},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["No drive, re-excitation, multiphotons or g2 prediction.",
            "No actual electromagnetic field, molecular spectrum or device is fitted.",
            "Markovian independent loss channels and emitter pure dephasing.",
            "No direct-emitter/cavity common-bath interference or delayed grating feedback.",
            "One-way scalar downstream factors; grating efficiency is illustrative."]}
    for entry in manifest:
        entry["sha256"] = {ext: hashlib.sha256(Path(path).read_bytes()).hexdigest() for ext, path in entry["files"].items()}
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"default_P_bus": default_p["bus"], "default_P_detected": float(stages[-1]),
                      "loading_optimum_MHz": optimum, "loading_optimum_P_bus": optimum_p,
                      "checks": checks["status"], "figures": len(manifest), "output": str(out)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("qed_parameters.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("qed_toy_output"))
    parser.add_argument("--figures", type=Path, default=Path(__file__).resolve().parents[1]/"figures")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    build(cfg, args.output.resolve(), args.figures.resolve())


if __name__ == "__main__":
    main()
