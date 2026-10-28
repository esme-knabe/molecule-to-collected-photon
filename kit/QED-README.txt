This local toy follows one prepared molecular excitation into a cavity, a bus,
and a selected collection channel. It does not run electromagnetic simulations
or connect to any account. All parameters are illustrative; none are fitted to
a particular molecular species or a fabricated device.

Run with Python, NumPy, SciPy and Matplotlib:

    python qed_toy.py

Edit qed_parameters.json, then rerun. The output directory contains CSV data,
the exact parameters used, summary.json with figure captions, and checks.json.
Figures are exported to ../figures as PNG, PDF and editable SVG. Equations and
the plotting source remain in qed_toy.py. The output summary records the Python and library versions used for the
included results.

Units and conventions

Time is in ns. g/(2pi), kappa_i/(2pi), kappa_ex/(2pi), gamma_phi/(2pi), and
Delta/(2pi) are entered in MHz. gamma_1=1/lifetime_ns is the excited-population
loss rate. kappa is the cavity photon-number/energy decay rate, so the field
amplitude decays at kappa/2. gamma_phi is the pure coherence decay rate;
gamma_2=gamma_1/2+gamma_phi. Delta=omega_c-omega_e. In the emitter/cavity basis,
H/hbar=[[0,g],[g,Delta]] for real g. Consequently c=rho_(e,c) obeys
c'=-(A-i Delta)c+i g(p_e-p_c), with A=(kappa+gamma_1)/2+gamma_phi.

The initial condition is |e,0>. The retained density block is two by two for one
cavity mode, or three by three for emitter plus CW and CCW. It obeys the
Hamiltonian commutator, -{K,rho}/2 for loss, and
(gamma_phi/2)(Z rho Z-rho), with Z=diag(1,-1,...) for emitter dephasing.
Every omitted loss jump is accumulated in a separate channel counter. Thus
Tr(rho)+sum(counters)=1 at every time. Population loss is split into background
ZPL, sideband and nonradiative channels using q and s. The given g already
describes the selected transition: q or s is not multiplied into g again.

Infinite-time integration uses -L vec(X)=vec(rho_0). P_j=Tr(K_j X).
For one mode this is independently checked against

    R = 2 g^2 A / (A^2+Delta^2)
    P_cavity = kappa R / [kappa gamma_1+(kappa+gamma_1)R]
    P_bus = (kappa_ex/kappa) P_cavity.

On resonance without pure dephasing, P_cavity equals
[kappa/(kappa+gamma_1)] C/(1+C), where C=4g^2/(kappa gamma_1).
The familiar C/(1+C) is the bad-cavity limit of this initial-value problem.
The code's finite dynamics retain coherences and coherent reabsorption; R is
used for an integrated check, not as an unqualified instantaneous rate model.

The ring convention holds |g_CW|^2+|g_CCW|^2=g_B^2 fixed. Directionality D
sets the respective fractions to (1+D)/2 and (1-D)/2. A real linear transition
in an ideal reciprocal ring has the D=0 baseline. Nonzero D is a declared toy
asymmetry, not a result inferred from molecular position. A coherent coupling
J connects CW and CCW. Each traveling mode has its own external bus channel.

The separate molecular branching plot assumes an irreversible additional ZPL
channel, Gamma_c=F_extra s q gamma_1^0. It illustrates different denominators:
cavity capture, radiative quantum yield, and ZPL fraction of fluorescence.
It is not the coherent model reinterpreted as a total Purcell factor.

The final collection plot multiplies bus emission by eta_prop, eta_GC into one
selected optical mode, and eta_det. These are assumed passive efficiencies.
The grating value is an input, not a newly simulated grating performance.
This plot uses one retained cavity mode with one bus output. For the D=0, J=0
ring with equal CW/CCW coupling and only one collected arm, the one-arm bus and
detected probabilities are each half the single-output values. Do not apply a
two-arm emission probability to a single collected grating.

Useful boundaries of the calculation

No drive is present, so there is no re-excitation or multiphoton emission.
This toy cannot predict finite-pulse g2 or photon indistinguishability.
Re-excitation requires a driven ground/emitter/cavity model with enough Fock
states and photon-counting or two-time correlation calculations. Pure dephasing
can improve capture of a detuned emitter while reducing coherence; capture
probability does not measure photon indistinguishability. Shelving, molecular
spectral diffusion, common-bath interference, and delayed optical feedback are
not represented. The checks establish algebraic and numerical consistency of
this finite analytical model, not convergence or validation of an LN device.

The derivations connect to Cui and Raymer, Phys. Rev. A 73, 053807 (2006),
with erratum 78, 049904(E) (2008), https://arxiv.org/abs/quant-ph/0601108v3;
Grange et al., Phys. Rev. Lett. 114, 193601 (2015),
https://arxiv.org/abs/1501.00931v2; and the input-output framework of Gardiner
and Collett, Phys. Rev. A 31, 3761 (1985), doi:10.1103/PhysRevA.31.3761.
Read their rate definitions before comparing factors of two.
