#!/usr/bin/env python3
"""Offline grating-coupler toy: depletion, coherent array factor, aperture overlap.

Run: python gc_toy.py
Edit gc_parameters.json, or use --pitch-um, --duty, --periods, --alpha-amp.
Only NumPy and Matplotlib are needed. No Tidy3D import, account or network call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def validate(c):
    for key in ['target_wavelength_um', 'pitch_um', 'effective_index_at_target',
                'group_index_at_target', 'collection_medium_index',
                'reference_amplitude_leakage_per_um', 'gaussian_amplitude_waist_um']:
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not isinstance(c['periods'], int) or not 2 <= c['periods'] <= 500:
        raise ValueError('periods must be an integer from 2 to 500')
    for key in ['duty_cycle', 'reference_duty_cycle', 'gaussian_center_fraction']:
        if not 0 < c[key] < 1:
            raise ValueError(f'{key} must lie strictly between 0 and 1')
    if not 0 < c['wavelength_min_um'] < c['wavelength_max_um']:
        raise ValueError('Require 0 < wavelength_min_um < wavelength_max_um')
    if not -90 < c['angle_min_deg'] < c['angle_max_deg'] < 90:
        raise ValueError('Require -90 < angle_min_deg < angle_max_deg < 90')
    if (not isinstance(c['wavelength_samples'],int) or not isinstance(c['angle_samples'],int)
        or not 11 <= c['wavelength_samples'] <= 5001 or not 101 <= c['angle_samples'] <= 20001):
        raise ValueError('Use 11–5001 wavelength samples and 101–20001 angle samples')
    if c['target_angle_deg'] is not None and not -90 < c['target_angle_deg'] < 90:
        raise ValueError('target_angle_deg must be between -90 and 90, or null')


def beta(wavelength_um, c):
    """First-order dispersion of propagation constant about lambda0, units rad/um."""
    l0 = c['target_wavelength_um']
    b0 = 2*np.pi*c['effective_index_at_target']/l0
    return b0 - 2*np.pi*c['group_index_at_target']*(np.asarray(wavelength_um)-l0)/l0**2


def phase_mismatch(wavelength_um, theta_rad, c):
    """Delta k = beta - k_clad sin(theta) - 2 pi/pitch. theta is from normal."""
    return (beta(wavelength_um, c)
            - 2*np.pi*c['collection_medium_index']*np.sin(theta_rad)/wavelength_um
            - 2*np.pi/c['pitch_um'])


def uniform_grating(c):
    p = c['pitch_um']
    n = c['periods']
    # Illustrative first-Fourier-harmonic strength. Not a calibrated etch model.
    strength = (np.sin(np.pi*c['duty_cycle'])/np.sin(np.pi*c['reference_duty_cycle']))**2
    alpha = c['reference_amplitude_leakage_per_um']*strength
    edges = np.arange(n+1)*p
    power = np.exp(-2*alpha*edges)  # dA/dx=-alpha A; dP/dx=-2 alpha P.
    emitted = power[:-1]*(-np.expm1(-2*alpha*p))
    return (edges[:-1]+p/2, power, emitted, np.full(n, alpha))


def apodized_grating(c, emitted_total, x):
    """Inverse toy: prescribe Gaussian emitted power and solve cell leakage exactly."""
    center = c['gaussian_center_fraction']*c['periods']*c['pitch_um']
    waist = c['gaussian_amplitude_waist_um']
    target = np.exp(-2*((x-center)/waist)**2)
    target /= target.sum()
    emitted = emitted_total*target
    power = np.r_[1., 1.-np.cumsum(emitted)]
    fraction = emitted/power[:-1]
    if np.any(fraction >= 1) or power[-1] <= 0:
        raise ValueError('Chosen extraction is too close to unity for stable apodization')
    alpha = -np.log1p(-fraction)/(2*c['pitch_um'])
    return power, emitted, alpha, target


def overlap(emitted, target_weights, x, dk):
    """Discrete normalized aperture overlap; no absolute power or fiber efficiency."""
    phase = np.exp(1j*np.asarray(dk)[..., None]*x)
    numerator = np.abs(np.sum(np.sqrt(emitted*target_weights)*phase, axis=-1))**2
    return numerator/(emitted.sum()*target_weights.sum())


def angular_shape(emitted, x, theta, c):
    dk = phase_mismatch(c['target_wavelength_um'], theta, c)
    array = np.sum(np.sqrt(emitted)[None, :]*np.exp(1j*dk[:, None]*x), axis=1)
    # Shape only. Arbitrary radiative element pattern is set to unity.
    return np.abs(array)**2/np.max(np.abs(array)**2)


def phase_match_angle(wavelengths, c):
    u = (beta(wavelengths, c)-2*np.pi/c['pitch_um'])*wavelengths/(2*np.pi*c['collection_medium_index'])
    return np.rad2deg(np.arcsin(np.where(np.abs(u) <= 1, u, np.nan)))


def calculate(c):
    validate(c)
    x, pu, qu, au = uniform_grating(c)
    pa, qa, aa, target = apodized_grating(c, float(qu.sum()), x)
    match0 = float(phase_match_angle(np.asarray(c['target_wavelength_um']), c))
    target_angle = c['target_angle_deg']
    if target_angle is None:
        if not np.isfinite(match0):
            raise ValueError('No first-order phase-matched angle; set target_angle_deg or change pitch/index')
        target_angle = match0
    theta = np.deg2rad(np.linspace(c['angle_min_deg'],c['angle_max_deg'],c['angle_samples']))
    wavelengths = np.linspace(c['wavelength_min_um'],c['wavelength_max_um'],c['wavelength_samples'])
    dk = phase_mismatch(wavelengths, np.deg2rad(target_angle), c)
    ou = overlap(qu, target, x, dk)
    oa = overlap(qa, target, x, dk)
    checks = {
        'uniform_power_conservation_error': float(abs(pu[-1]+qu.sum()-1)),
        'apodized_power_conservation_error': float(abs(pa[-1]+qa.sum()-1)),
        'equal_total_extraction_error': float(abs(qu.sum()-qa.sum())),
        'inverse_leakage_reconstruction_error': float(np.max(abs(pa[:-1]*(-np.expm1(-2*aa*c['pitch_um']))-qa))),
        'exact_phase_match_mismatch_rad_per_um': (float(phase_mismatch(c['target_wavelength_um'],np.deg2rad(match0),c)) if np.isfinite(match0) else None),
        'matched_apodized_overlap': float(overlap(qa,target,x,0.)),
        'overlap_bounds_passed': bool(np.all(ou >= -1e-12) and np.all(oa >= -1e-12) and np.all(ou <= 1+1e-12) and np.all(oa <= 1+1e-12)),
        'monotone_guided_power_passed': bool(np.all(np.diff(pu) <= 0) and np.all(np.diff(pa) <= 0))
    }
    for name in ['uniform_power_conservation_error','apodized_power_conservation_error',
                 'equal_total_extraction_error','inverse_leakage_reconstruction_error']:
        if checks[name] > 1e-12:
            raise AssertionError(f'Bookkeeping failed: {name}')
    if not checks['overlap_bounds_passed'] or not checks['monotone_guided_power_passed']:
        raise AssertionError('Unphysical toy power or normalized overlap')
    return dict(x=x,pu=pu,qu=qu,au=au,pa=pa,qa=qa,aa=aa,target=target,theta=theta,
                wavelengths=wavelengths,ou=ou,oa=oa,match0=match0,target_angle=target_angle,
                shape_u=angular_shape(qu,x,theta,c),shape_a=angular_shape(qa,x,theta,c),checks=checks)


def save(c, d, output):
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Trebuchet MS','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
    purple,orange = '#4A1F7A','#C36C30'
    p = c['pitch_um']
    edges = np.arange(c['periods']+1)*p
    fig, ax = plt.subplots(2,2,figsize=(11.5,8.0))
    ax[0,0].plot(edges,d['pu'],color=purple,label='Uniform leakage')
    ax[0,0].plot(edges,d['pa'],color=orange,label='Gaussian emitted aperture')
    ax[0,0].set(xlabel='Distance along grating (µm)',ylabel='Remaining guided power / input power',ylim=(0,1.05))
    ax[0,0].legend(frameon=False,fontsize=9)
    ax[0,0].set_title('(a) Exact toy power depletion',loc='left')
    ax[0,1].plot(d['x'],d['qu']/p,color=purple)
    ax[0,1].plot(d['x'],d['qa']/p,color=orange)
    ax[0,1].set(xlabel='Distance along grating (µm)',ylabel='Emitted power per µm / input power')
    ax[0,1].set_title('(b) Same chosen total extraction',loc='left')
    ax[1,0].plot(np.rad2deg(d['theta']),d['shape_u'],color=purple)
    ax[1,0].plot(np.rad2deg(d['theta']),d['shape_a'],color=orange)
    ax[1,0].axvline(d['target_angle'],color='0.5',ls=':',lw=1)
    ax[1,0].set(xlabel='Angle from surface normal (degrees)',ylabel='Array intensity / its own peak',ylim=(0,1.05))
    ax[1,0].set_title(f"(c) Coherent phase matching at {1000*c['target_wavelength_um']:g} nm",loc='left')
    ax[1,1].plot(1000*d['wavelengths'],d['ou'],color=purple)
    ax[1,1].plot(1000*d['wavelengths'],d['oa'],color=orange)
    ax[1,1].axvline(1000*c['target_wavelength_um'],color='0.5',ls=':',lw=1)
    ax[1,1].set(xlabel='Wavelength (nm)',ylabel='Normalized aperture overlap',ylim=(0,1.05))
    ax[1,1].set_title(f"(d) Fixed target angle = {d['target_angle']:.2f}°",loc='left')
    fig.suptitle('An analytical grating toy: leakage sets the envelope; phase sets the beam',fontsize=14)
    fig.text(.5,.015,'Illustrative inputs. These are array/aperture calculations, not FDTD fields or predicted fiber efficiencies.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    for ext in ['png','pdf']:
        fig.savefig(output/f'gc_toy_overview.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.8,4.6))
    ax.step(d['x'],d['au'],where='mid',color=purple,label='Uniform')
    ax.step(d['x'],d['aa'],where='mid',color=orange,label='Gaussian target, inverse toy solution')
    ax.set(xlabel='Distance along grating (µm)',ylabel='Amplitude leakage α (1/µm)',
           title='Required leakage is not the same as the emitted envelope')
    ax.legend(frameon=False,fontsize=9)
    fig.text(.5,.02,'Convention: dA/dx = −αA and dP/dx = −2αP. No calibrated map from α to tooth geometry is assumed.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ['png','pdf']:
        fig.savefig(output/f'gc_toy_leakage.{ext}',dpi=180)
    plt.close(fig)
    cells=np.column_stack([np.arange(c['periods']),d['x'],d['pu'][:-1],d['qu'],d['au'],
                           d['pa'][:-1],d['qa'],d['aa'],d['target']])
    np.savetxt(output/'gc_cells.csv',cells,delimiter=',',comments='',
               header='cell,x_um,uniform_guided_power_in,uniform_emitted_power,uniform_alpha_amp_per_um,apodized_guided_power_in,apodized_emitted_power,apodized_alpha_amp_per_um,target_power_weight')
    np.savetxt(output/'gc_spectrum.csv',np.column_stack([d['wavelengths']*1000,d['ou'],d['oa'],phase_match_angle(d['wavelengths'],c)]),
               delimiter=',',comments='',header='wavelength_nm,uniform_normalized_aperture_overlap,apodized_normalized_aperture_overlap,first_order_phase_match_angle_deg')
    np.savetxt(output/'gc_angles.csv',np.column_stack([np.rad2deg(d['theta']),d['shape_u'],d['shape_a']]),
               delimiter=',',comments='',header='angle_deg,uniform_intensity_over_own_peak,apodized_intensity_over_own_peak')
    summary = {'model':'Offline scalar discrete leakage and array-factor toy',
               'parameters':c,'derived':{'length_um':c['periods']*p,'phase_match_angle_deg':d['match0'] if np.isfinite(d['match0']) else None,
               'target_angle_deg':d['target_angle'],'chosen_extracted_fraction':float(d['qu'].sum()),
               'remaining_guided_fraction':float(d['pu'][-1]),
               'uniform_aperture_overlap_at_target':float(overlap(d['qu'],d['target'],d['x'],phase_mismatch(c['target_wavelength_um'],np.deg2rad(d['target_angle']),c))),
               'apodized_aperture_overlap_at_target':float(overlap(d['qa'],d['target'],d['x'],phase_mismatch(c['target_wavelength_um'],np.deg2rad(d['target_angle']),c)))},
               'equations':{'alpha':'dA/dx=-alpha_amp A; dP/dx=-2 alpha_amp P',
               'cell_power':'q_j=P_j[1-exp(-2 alpha_amp,j pitch)]',
               'uniform_strength':'alpha_amp(d)=alpha_ref [sin(pi d)/sin(pi d_ref)]^2, an illustrative first-harmonic law',
               'phase':'Delta_k=beta(lambda)-2pi n_clad sin(theta)/lambda-2pi/pitch',
               'array':'F(theta)=sum_j sqrt(q_j) exp(i Delta_k x_j)',
               'target':'w_j proportional to exp[-2((x_j-xc)/waist)^2]',
               'overlap':'|sum_j sqrt(q_j w_j) exp(i Delta_k x_j)|^2 / [sum_j q_j sum_j w_j]'},
               'assumptions':['Unit guided input power; all prescribed guided depletion is assigned to toy emitted cells.',
               'No reflection, absorption, upward/downward partition, vector radiation element pattern, substrate interference or finite transverse width.',
               'No calibrated dependence on film thickness, etch, material tensor or real duty-cycle scattering.',
               'First-order beta dispersion; leakage and target aperture held independent of wavelength.',
               'Gaussian matching is imposed by construction; overlap near one is not independent validation.',
               'Array intensity is normalized to its own peak, and overlap is normalized within the discrete aperture. Neither is absolute fiber coupling.'],
               'checks':d['checks'],'fdtd_run':False,'network_used':False,
               'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'gc_toy_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (output/'gc_parameters_used.json').write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps({'output':str(output),'phase_match_angle_deg':d['match0'],
                      'chosen_extracted_fraction':float(d['qu'].sum()),'checks':d['checks']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',type=Path,default=ROOT/'gc_parameters.json')
    parser.add_argument('--output',type=Path,default=ROOT/'gc_toy_output')
    parser.add_argument('--pitch-um',type=float)
    parser.add_argument('--duty',type=float)
    parser.add_argument('--periods',type=int)
    parser.add_argument('--alpha-amp',type=float,help='Reference amplitude leakage in inverse micrometres')
    parser.add_argument('--target-angle-deg',type=float)
    args=parser.parse_args()
    c=json.loads(args.config.read_text())
    for arg,key in [('pitch_um','pitch_um'),('duty','duty_cycle'),('periods','periods'),
                    ('alpha_amp','reference_amplitude_leakage_per_um'),('target_angle_deg','target_angle_deg')]:
        if getattr(args,arg) is not None:
            c[key]=getattr(args,arg)
    save(c,calculate(c),args.output)


if __name__=='__main__':
    main()
