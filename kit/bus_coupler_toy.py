"""Lossless evanescent coupler: spatial mode beating -> per-pass K -> kappa_ex.

Run locally: python kit/bus_coupler_toy.py
Edit the adjacent JSON. No electromagnetic or cloud solve is performed.
"""
from pathlib import Path
import json
import platform
import hashlib
import numpy as np
import scipy
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
C_UM_PER_NS=299792.458

def transfer(mu, mismatch, length):
    q=np.sqrt(np.abs(mu)**2+(np.asarray(mismatch)/2)**2)
    return np.abs(mu)**2*np.asarray(length)**2*np.sinc(q*np.asarray(length)/np.pi)**2

def main():
    cfg=json.loads((HERE/"bus_coupler_parameters.json").read_text())
    mu0=cfg["coupling_at_reference_gap_per_um"]
    ell=cfg["amplitude_overlap_length_nm"]
    length=cfg["interaction_length_um"]
    if mu0<0 or ell<=0 or length<0 or cfg["group_index"]<=0 or cfg["ring_radius_um"]<=0:
        raise ValueError("Coupling/length must be nonnegative; decay length, group index and radius must be positive.")
    out=HERE/"bus_coupler_toy_output";out.mkdir(exist_ok=True)
    figs=ROOT/"figures";figs.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":["Trebuchet MS","DejaVu Sans"],"mathtext.fontset":"cm",
        "font.size":10,"axes.spines.top":False,"axes.spines.right":False,
        "svg.fonttype":"none","pdf.fonttype":42,"axes.prop_cycle":matplotlib.cycler(color=["#4A1F7A","#B2182B","#007A50"])})
    fig,ax=plt.subplots(1,2,figsize=(10.2,3.7),layout="constrained")
    lengths=np.linspace(0,40,401)
    curves=[]
    for db in [0,.025,.05]:
        k=transfer(mu0,db,lengths);curves.append(k)
        ax[0].plot(lengths,100*k,label=rf"$\Delta\beta={db:g}\ \mu\mathrm{{m}}^{{-1}}$")
    ax[0].set(xlabel="Interaction length (um)",ylabel="Power transferred per pass (%)")
    ax[0].legend(frameon=False,fontsize=8)
    gaps=np.linspace(120,500,301)
    mus=mu0*np.exp(-(gaps-cfg["reference_gap_nm"])/ell)
    k=transfer(mus,cfg["propagation_mismatch_per_um"],length)
    trt=cfg["group_index"]*2*np.pi*cfg["ring_radius_um"]/C_UM_PER_NS
    kappa_ex=-np.log1p(-k)/trt
    kappa_MHz=kappa_ex/(2*np.pi)*1000
    ax[1].semilogy(gaps,kappa_MHz,color="#4A1F7A")
    ax[1].set(xlabel="Gap (nm)",ylabel=r"External energy decay $\kappa_{\rm ex}/2\pi$ (MHz)")
    ax[1].axvline(cfg["reference_gap_nm"],color="#666666",ls="--",lw=1)
    ax[1].set_title(f"Assumed overlap length {ell:g} nm; interaction {length:g} um",fontsize=9)
    for ext in ["png","pdf","svg"]:
        fig.savefig(figs/f"bus-coupler-toy.{ext}",dpi=220 if ext=="png" else None)
    plt.close(fig)
    np.savetxt(out/"length_sweep.csv",np.column_stack([lengths,*curves]),delimiter=",",
        header="length_um,K_dbeta0,K_dbeta0p025,K_dbeta0p05",comments="")
    np.savetxt(out/"gap_sweep.csv",np.column_stack([gaps,mus,k,kappa_MHz]),delimiter=",",
        header="gap_nm,mu_per_um,K,kappa_ex_over2pi_MHz",comments="")
    errors=[]
    for m in [0,.008,.03]:
        for db in [0,.025,-.05]:
            for z in [0,5,40]:
                H=np.array([[db/2,m],[m,-db/2]],complex)
                state=expm(1j*H*z)@np.array([1,0],complex)
                errors.append(abs(transfer(m,db,z)-abs(state[1])**2))
                if not np.isclose(np.vdot(state,state),1,atol=1e-12):
                    raise AssertionError("Lossless coupler did not conserve power.")
    maxerr=float(max(errors))
    if maxerr>1e-12 or np.any(k<0) or np.any(k>=1):raise AssertionError("Coupler checks failed")
    k0=float(transfer(mu0,cfg["propagation_mismatch_per_um"],length))
    summary={"model":"analytical two-mode lossless directional coupler","parameters":cfg,
        "default_power_transfer_K":k0,"round_trip_time_ns":trt,
        "default_kappa_ex_over_2pi_MHz":float(-np.log1p(-k0)/trt/(2*np.pi)*1000),
        "checks":{"independent_matrix_cases":len(errors),"max_formula_vs_matrix_error":maxerr,
                  "lossless_power_conserved":True},
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
        "caption":"Lossless spatial two-mode toy. Phase mismatch reduces power transfer; an assumed exponential overlap versus gap maps per-pass transfer to an external ring energy-decay rate. No gap dependence is obtained from an electromagnetic geometry.",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"K":k0,"kappa_ex_over2pi_MHz":summary["default_kappa_ex_over_2pi_MHz"],"checks":summary["checks"]}))

if __name__=="__main__":main()
