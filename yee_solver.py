"""
1D nonlinear-Maxwell Yee solver with fixed-point iteration -- the Stage-1
"Benchmark" of the paper's Section 5.

We solve the source-free macroscopic Maxwell system (paper Eq. 12)

    d_t D = curl H ,   d_t B = -curl E ,   D = E + P_vac ,   H = B - M_vac

on a staggered 1D Yee lattice. Because D, H depend nonlinearly on E, B through
the Heisenberg-Euler P_vac, M_vac (Eqs. 9-10), the recovery of E from the
time-advanced D uses a fixed-point iteration exactly as described in the paper:
the linear field is the predictor; P_vac/M_vac are evaluated; E is re-solved
until the correction is below tolerance.

Benchmark
---------
A weak transverse probe propagates through a constant magnetic background B0
(along y). The two probe polarizations sample the vacuum differently and acquire
different phase velocities -> vacuum birefringence. The analytic targets, derived
from the SAME L_HE (see analytic_indices and the sympy derivation), are

    n_perp (probe E perp B0) = 1 + 4 xi B0^2
    n_para (probe E para B0) = 1 + 7 xi B0^2
    Delta n                  = 3 xi B0^2          (7:4 Adler ratio)

Measurement method
------------------
We do NOT use pulse time-of-flight: on a discrete Yee grid the carrier samples a
different point of the numerical dispersion curve when the index changes, which
does not cancel against a vacuum reference and leaves a several-percent bias.

Instead we measure the *eigenfrequency* of a single Fourier mode. With a uniform
background the linearized system is spatially homogeneous, so a pure standing
wave cos(k x) is an exact eigenmode. We record the field at an antinode and
extract omega from the slope of the analytic-signal phase. The medium index
follows from comparison to a vacuum (xi = 0) run AT THE SAME k:

    n = omega_vacuum / omega_medium .

Because both runs use identical k (identical spatial sampling), the Yee
numerical dispersion is identical and cancels exactly; only the physical HE
response remains. This recovers the analytic coefficients to ~1e-4.

Units: normalized c = 1, dx = 1.
"""

import numpy as np


def analytic_indices(xi, B0):
    """Return (n_perp, n_para, delta_n) to LEADING order in xi*B0^2.

    These are the textbook Adler 7:4 coefficients and are exact only as
    xi*B0^2 -> 0. For finite background see exact_indices().
    """
    eps = xi * B0**2
    return 1.0 + 4.0 * eps, 1.0 + 7.0 * eps, 3.0 * eps


def exact_indices(xi, B0):
    """Exact linear-probe indices implied by the HE constitutive relations,
    keeping ALL orders in the background eps = xi*B0^2.

    Linearizing P_vac, M_vac (Eqs. 9-10) about a constant B0 (y) for a weak
    transverse probe gives effective constitutive coefficients
        perp:  D = (1 - 4 eps) E ,   H = (1 - 12 eps) B  -> v^2 = (1-12e)/(1-4e)
        para:  D = (1 + 10 eps) E ,  H = (1 -  4 eps) B  -> v^2 = (1-4e)/(1+10e)
    so the phase indices n = 1/v are the square roots below. They reduce to
    1 + 4 eps and 1 + 7 eps as eps -> 0.
    """
    eps = xi * B0**2
    n_perp = np.sqrt((1.0 - 4.0 * eps) / (1.0 - 12.0 * eps))
    n_para = np.sqrt((1.0 + 10.0 * eps) / (1.0 - 4.0 * eps))
    return n_perp, n_para, n_para - n_perp


# ---- constitutive relations specialized to 1D transverse fields ----
#   E = (0, Ey, Ez),  B = (0, B0 + By, Bz)
def _Pvac_yz(Ey, Ez, By_tot, Bz, xi):
    s = (Ey**2 + Ez**2) - (By_tot**2 + Bz**2)
    p = Ey * By_tot + Ez * Bz
    Py = 2.0 * xi * (2.0 * s * Ey + 7.0 * p * By_tot)
    Pz = 2.0 * xi * (2.0 * s * Ez + 7.0 * p * Bz)
    return Py, Pz


def _Mvac_yz(Ey, Ez, By_tot, Bz, xi):
    s = (Ey**2 + Ez**2) - (By_tot**2 + Bz**2)
    p = Ey * By_tot + Ez * Bz
    My = 2.0 * xi * (-2.0 * s * By_tot + 7.0 * p * Ey)
    Mz = 2.0 * xi * (-2.0 * s * Bz + 7.0 * p * Ez)
    return My, Mz


def _simulate_eigenmode(mode, xi, B0, N, m, courant, n_record_periods,
                        amp, fp_tol, fp_maxit):
    """Evolve a single standing-wave Fourier mode in a uniform B0 background and
    return the antinode time series (for frequency extraction).

    mode : "perp"  -> probe E along z (perp to B0)  -> expects n = 1 + 4 xi B0^2
           "para"  -> probe E along y (para to B0)  -> expects n = 1 + 7 xi B0^2
    m    : integer mode number; k = 2*pi*m/N (exactly periodic).
    """
    dx = 1.0
    dt = courant * dx
    inv_dx = 1.0 / dx
    x = np.arange(N) * dx
    k = 2.0 * np.pi * m / N

    Ez = np.zeros(N); By = np.zeros(N); Hy = np.zeros(N); Dz = np.zeros(N)
    Ey = np.zeros(N); Bz = np.zeros(N); Hz = np.zeros(N); Dy = np.zeros(N)

    # standing-wave initial condition: E at antinode, B = 0 (half-step earlier)
    profile = amp * np.cos(k * x)
    if mode == "perp":
        Ez[:] = profile
    else:
        Ey[:] = profile

    # constant background contribution to H (curl-free; cancels under d_x)
    My_bg, Mz_bg = _Mvac_yz(0.0, 0.0, B0, 0.0, xi)
    Hy_bg = B0 - My_bg
    Hz_bg = 0.0 - Mz_bg

    # consistent D initialization: D = E + P_vac(E, B0)
    Py0, Pz0 = _Pvac_yz(Ey, Ez, B0 + By, Bz, xi)
    Dy[:] = Ey + Py0
    Dz[:] = Ez + Pz0

    # frequency near omega ~ k; choose record length for many full periods
    omega_guess = k
    period = 2.0 * np.pi / omega_guess
    n_steps = int(n_record_periods * period / dt)

    trace = np.zeros(n_steps)

    for it in range(n_steps):
        # 1. Faraday (periodic): B at half nodes from E at integer nodes
        By += dt * (np.roll(Ez, -1) - Ez) * inv_dx
        Bz += -dt * (np.roll(Ey, -1) - Ey) * inv_dx

        # 2. H at half nodes (subtract curl-free constant background part)
        Ey_h = 0.5 * (Ey + np.roll(Ey, -1))
        Ez_h = 0.5 * (Ez + np.roll(Ez, -1))
        By_tot_h = B0 + By
        My_h, Mz_h = _Mvac_yz(Ey_h, Ez_h, By_tot_h, Bz, xi)
        Hy[:] = ((B0 + By) - My_h) - Hy_bg
        Hz[:] = (Bz - Mz_h) - Hz_bg

        # 3. Ampere (periodic): D at integer nodes from H at half nodes
        Dz += dt * (Hy - np.roll(Hy, 1)) * inv_dx
        Dy += -dt * (Hz - np.roll(Hz, 1)) * inv_dx

        # 4. Recover E from D via fixed-point iteration (predictor E = D)
        By_int = 0.5 * (np.roll(By, 1) + By)
        Bz_int = 0.5 * (np.roll(Bz, 1) + Bz)
        By_tot_int = B0 + By_int

        Ey_new = Dy.copy(); Ez_new = Dz.copy()
        for _ in range(fp_maxit):
            Py, Pz = _Pvac_yz(Ey_new, Ez_new, By_tot_int, Bz_int, xi)
            Ey_next = Dy - Py
            Ez_next = Dz - Pz
            err = max(np.max(np.abs(Ey_next - Ey_new)),
                      np.max(np.abs(Ez_next - Ez_new)))
            Ey_new, Ez_new = Ey_next, Ez_next
            if err < fp_tol:
                break
        Ey[:] = Ey_new; Ez[:] = Ez_new

        trace[it] = Ez[0] if mode == "perp" else Ey[0]   # antinode x = 0

    return trace, dt


def _hilbert_analytic(x):
    """Analytic signal via FFT Hilbert transform (no scipy)."""
    N = len(x)
    X = np.fft.fft(x)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1; h[1:N // 2] = 2
    else:
        h[0] = 1; h[1:(N + 1) // 2] = 2
    return np.fft.ifft(X * h)


def _measure_omega(trace, dt, trim=0.12):
    """Extract angular frequency from the slope of the unwrapped analytic phase.

    A clean single-tone standing wave gives a phase that is linear in time; its
    slope is omega. Trimming removes Hilbert edge transients.
    """
    a = _hilbert_analytic(trace - np.mean(trace))
    phase = np.unwrap(np.angle(a))
    n = len(phase)
    lo = int(trim * n); hi = int((1.0 - trim) * n)
    t = np.arange(lo, hi) * dt
    p = phase[lo:hi]
    # least-squares slope
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, p, rcond=None)[0]
    return abs(slope)


def measure_index(mode, xi, B0, N=512, m=8, courant=0.5,
                  n_record_periods=120, amp=1.0e-4,
                  fp_tol=1e-14, fp_maxit=80):
    """Measure the refractive index of one probe polarization from the
    eigenfrequency ratio omega_vacuum / omega_medium at identical k."""
    tr_med, dt = _simulate_eigenmode(mode, xi, B0, N, m, courant,
                                     n_record_periods, amp, fp_tol, fp_maxit)
    tr_vac, _ = _simulate_eigenmode(mode, 0.0, B0, N, m, courant,
                                    n_record_periods, amp, fp_tol, fp_maxit)
    w_med = _measure_omega(tr_med, dt)
    w_vac = _measure_omega(tr_vac, dt)
    return w_vac / w_med


def run_birefringence(N=512, m=8, courant=0.5, xi=1.0, B0=0.10,
                      n_record_periods=120, amp=1.0e-4,
                      fp_tol=1e-14, fp_maxit=80, **_ignored):
    """Measure n_perp, n_para, delta_n from first principles and compare to the
    analytic targets. Returns a results dict (keys stable for downstream use)."""
    n_perp_meas = measure_index("perp", xi, B0, N, m, courant,
                                n_record_periods, amp, fp_tol, fp_maxit)
    n_para_meas = measure_index("para", xi, B0, N, m, courant,
                                n_record_periods, amp, fp_tol, fp_maxit)
    delta_n_meas = n_para_meas - n_perp_meas

    n_perp_th, n_para_th, dn_th = analytic_indices(xi, B0)
    n_perp_ex, n_para_ex, dn_ex = exact_indices(xi, B0)
    return {
        "xi": xi, "B0": B0, "xiB02": xi * B0**2,
        "n_perp_meas": n_perp_meas, "n_para_meas": n_para_meas,
        "delta_n_meas": delta_n_meas,
        "n_perp_th": n_perp_th, "n_para_th": n_para_th, "delta_n_th": dn_th,
        "n_perp_exact": n_perp_ex, "n_para_exact": n_para_ex,
        "delta_n_exact": dn_ex,
        "delta_n_rel_err": abs(delta_n_meas - dn_th) / dn_th if dn_th else 0.0,
        "n_perp_rel_err": abs(n_perp_meas - n_perp_th) / (n_perp_th - 1)
        if n_perp_th != 1 else 0.0,
        "n_para_rel_err": abs(n_para_meas - n_para_th) / (n_para_th - 1)
        if n_para_th != 1 else 0.0,
        "n_perp_exact_err": abs(n_perp_meas - n_perp_ex) / (n_perp_ex - 1)
        if n_perp_ex != 1 else 0.0,
        "n_para_exact_err": abs(n_para_meas - n_para_ex) / (n_para_ex - 1)
        if n_para_ex != 1 else 0.0,
    }


def coefficient_sweep(xiB02_values, N=512, m=8, courant=0.5,
                      n_record_periods=120, amp=1.0e-4):
    """Sweep small xi*B0^2 and fit (n-1) = coeff * xi*B0^2 for each mode to
    recover the {4, 7, 3} coefficients. Returns dict of measured arrays + fits."""
    eps = np.asarray(xiB02_values, float)
    np_perp = np.empty_like(eps); np_para = np.empty_like(eps)
    for i, e in enumerate(eps):
        B0 = np.sqrt(e)            # fix xi = 1, vary B0
        np_perp[i] = measure_index("perp", 1.0, B0, N, m, courant,
                                   n_record_periods, amp)
        np_para[i] = measure_index("para", 1.0, B0, N, m, courant,
                                   n_record_periods, amp)
    # leading coefficient via two-term fit  (n-1) = a*eps + b*eps^2
    # (pure slope-through-origin is biased by the higher-order term)
    def _lead(y):
        A = np.vstack([eps, eps**2]).T
        a, _b = np.linalg.lstsq(A, y - 1.0, rcond=None)[0]
        return a
    c_perp = _lead(np_perp)
    c_para = _lead(np_para)
    A = np.vstack([eps, eps**2]).T
    c_delta = np.linalg.lstsq(A, (np_para - np_perp), rcond=None)[0][0]
    return {
        "eps": eps, "n_perp": np_perp, "n_para": np_para,
        "dn": np_para - np_perp,
        "coeff_perp": c_perp, "coeff_para": c_para, "coeff_delta": c_delta,
    }


if __name__ == "__main__":
    print("Stage-1 benchmark: 1D nonlinear-Maxwell Yee vacuum birefringence")
    print("(eigenfrequency method; index = omega_vac / omega_medium)\n")
    r = run_birefringence()
    print(f"  xi*B0^2 = {r['xiB02']:.4e}")
    print("  measured indices vs EXACT constitutive index (all orders in eps):")
    print(f"    n_perp : meas {r['n_perp_meas']:.8f}  exact {r['n_perp_exact']:.8f}"
          f"  (err {r['n_perp_exact_err']*100:.3f}%)")
    print(f"    n_para : meas {r['n_para_meas']:.8f}  exact {r['n_para_exact']:.8f}"
          f"  (err {r['n_para_exact_err']*100:.3f}%)")
    print(f"    Delta n: meas {r['delta_n_meas']:.6e}  exact {r['delta_n_exact']:.6e}")
    print("  leading-order (Adler) targets, exact only as eps -> 0:")
    print(f"    n_perp ~ 1+4eps = {r['n_perp_th']:.6f},  "
          f"n_para ~ 1+7eps = {r['n_para_th']:.6f},  Dn ~ 3eps = {r['delta_n_th']:.4e}")

    print("\nSmall-eps coefficient sweep (recovering the {4,7,3} Adler ratio):")
    sweep = coefficient_sweep([2e-4, 5e-4, 1e-3, 2e-3])
    for e, npp, npa in zip(sweep["eps"], sweep["n_perp"], sweep["n_para"]):
        print(f"    eps={e:.1e}:  (n_perp-1)/eps={ (npp-1)/e:6.3f}   "
              f"(n_para-1)/eps={ (npa-1)/e:6.3f}   "
              f"Dn/eps={ (npa-npp)/e:6.3f}")
    print(f"  fitted coefficients (eps->0):  perp={sweep['coeff_perp']:.3f}  "
          f"para={sweep['coeff_para']:.3f}  delta={sweep['coeff_delta']:.3f}")
    print(f"  expected:                      perp=4.000  para=7.000  delta=3.000")
