Run an editable grating-coupler toy

This small model makes the grating's two jobs visible: each tooth removes some
guided power, and the fields from different teeth interfere. It uses NumPy and
Matplotlib locally. There is no Tidy3D account, API key, cloud job or credit use.

From this folder, with Python 3.10 or newer:
    python -m venv .venv
    .venv/bin/python -m pip install -r requirements-toys.txt
    .venv/bin/python gc_toy.py
On Windows, replace .venv/bin/python with .venv\Scripts\python.exe.

Open gc_toy_output/gc_toy_overview.png. The four panels compare guided-power
depletion, emitted-aperture shape, angular interference, and normalized overlap
with a target Gaussian aperture. gc_toy_leakage.png shows the leakage strength
needed along the grating. Editable source, CSV data and a JSON record accompany
the PNG/PDF figures.

Start by editing gc_parameters.json, or try:
    python gc_toy.py --pitch-um 0.33 --output comparison_pitch
    python gc_toy.py --duty 0.3 --output comparison_duty
    python gc_toy.py --periods 30 --output comparison_length
    python gc_toy.py --alpha-amp 0.04 --output comparison_leakage

The default pitch is 350 nm, the period count is 18, and the wavelength is 595 nm.
The effective index 2.05 and group index 2.2 are teaching inputs. They are not
computed from an LN stack. Likewise, alpha_ref=0.1 per micrometre is a chosen
amplitude-leakage rate. It is not a predicted scattering coefficient.

When target_angle_deg is null, each run chooses the first-order phase-matched
angle at the target wavelength. Set it to a fixed number (for example 20.49) to
keep the collection direction fixed while changing pitch or index. The displayed
angular range is from -80 to 80 degrees, measured from the surface normal.

Conventions

The complex guided amplitude obeys dA/dx=-alpha A, hence guided power obeys
dP/dx=-2 alpha P. A cell of length p emits q_j=P_j(1-exp(-2 alpha_j p)). The
remaining guided power plus all cell emission equals the unit input power.
This toy assigns every removed watt to the emitted array; it has no absorption,
reflection, substrate channel or upward/downward partition.

The phase mismatch is beta - k_clad sin(theta) - 2pi/p. We add the complex fields
sqrt(q_j) exp(i mismatch x_j). The result is an array shape normalized to its own
peak. A vector radiation pattern, interface Fresnel factors, vertical stack and
finite transverse aperture would be needed for an absolute radiation prediction.

The uniform grating uses alpha(d)=alpha_ref[sin(pi d)/sin(pi d_ref)]^2. This is a
deliberately simple first-harmonic strength law: d and 1-d give the same strength,
and alpha is largest near d=0.5. A real grating also changes effective index,
phase, reflection and directionality when duty or etch changes. Here those
effects are not included.

For apodization, prescribe a Gaussian emitted-power profile q_j proportional to
exp[-2((x_j-x_center)/waist)^2], using the same total extraction as the uniform
case. Then solve alpha_j=-log(1-q_j/P_j)/(2p). The parameter called waist is the
Gaussian AMPLITUDE waist: amplitude is exp[-((x-x_center)/waist)^2]. The required
alpha profile usually differs from the emitted envelope because the power still
available in the guide decreases along x. No map from this alpha profile to an
achievable tooth geometry is assumed.

The overlap compares two normalized discrete aperture fields. It includes phase
mismatch across the grating, so it changes with wavelength at a fixed angle.
An overlap of one for the apodized grating at phase match is imposed by this
inverse toy construction. It is not measured or simulated fiber efficiency.
The beta(lambda) model is linearized using the chosen group index; alpha and
target Gaussian shape are held constant over the wavelength range.

Useful changes

Change pitch with a fixed angle and watch the spectral peak move. Increase the
period count and inspect the narrower coherent angular pattern. Change leakage
and see how strongly front-loaded the uniform aperture becomes. Move or resize
the Gaussian target and compare the required alpha profile with its emitted
envelope. These are conceptual relationships to carry into a later Tidy3D model.

The script checks power conservation, equal total extraction, reconstruction of
the apodized leakage law, monotone guided power, phase matching and overlap
bounds. gc_toy_summary.json records those checks and all assumptions.
