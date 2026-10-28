# From a molecular excitation to a collected photon

This guide follows a molecular excitation through a ring resonator, into a bus waveguide, and out through a grating coupler. It connects the field calculations to cavity QED, with an eventual design target of a 595 nm TM-like grating coupler on X-cut lithium niobate.

## Read the guide

- **[Open the edited guide PDF](docs/Molecule-to-Collected-Photon.pdf)**
- [Editable Pages source](docs/Molecule-to-Collected-Photon.pages)

The PDF and Pages file are the edited edition. To update the guide, edit the Pages document and export a new PDF.

## Try the toys

The three Python models run locally. They use NumPy, SciPy and Matplotlib, and do not use Tidy3D credits.

| Model | What you can explore | Inputs |
| --- | --- | --- |
| [Grating](kit/gc_toy.py) | Pitch, emitted angle, leakage, aperture length and Gaussian beam matching | [gc_parameters.json](kit/gc_parameters.json) |
| [Ring–bus coupler](kit/bus_coupler_toy.py) | Gap, interaction length, propagation mismatch and external cavity decay | [bus_coupler_parameters.json](kit/bus_coupler_parameters.json) |
| [Cavity QED](kit/qed_toy.py) | Dipole alignment, detuning, dephasing, ring directionality and collection probabilities | [qed_parameters.json](kit/qed_parameters.json) |

Use Python 3.10 or newer. Download the repository with **Code → Download ZIP**, or clone it:

~~~sh
git clone https://github.com/esme-knabe/molecule-to-collected-photon.git
cd molecule-to-collected-photon
python -m venv .venv
~~~

On macOS or Linux:

~~~sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python kit/gc_toy.py
.venv/bin/python kit/bus_coupler_toy.py
.venv/bin/python kit/qed_toy.py
~~~

On Windows, use <code>.venv\Scripts\python.exe</code> in place of <code>.venv/bin/python</code>.

Edit a model's JSON inputs and rerun its script. Each model saves plots, CSV data and a summary of its inputs and checks. Grating plots are in [kit/gc_toy_output](kit/gc_toy_output); ring–bus and cavity-QED plots are in [figures](figures). The output folders already contain example data, so you can read the guide and inspect the plots before installing anything.

The [grating notes](kit/GC-README.txt) and [cavity-QED notes](kit/QED-README.txt) explain the parameters and rate conventions in more detail. In the grating toy, set <code>target_angle_deg</code> to a fixed collection angle before comparing different pitches with the same optic.

## Using the models

These are analytical teaching models with adjustable inputs. They show how the couplings and losses fit together; they do not calculate a particular fabricated device. In particular, the grating's normalized aperture overlap is a shape comparison. The grating efficiency in the final collection budget is a separate assumed input.

The guide includes Tidy3D account setup and explains how later electromagnetic simulations supply the field profiles, coupling rates and collection efficiencies used by these models.
