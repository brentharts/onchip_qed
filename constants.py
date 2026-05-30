"""
Physical constants, QED scales, and unit conversions for the on-chip QED project.

All SI unless explicitly noted. The Heisenberg-Euler coefficient `xi` is provided
in BOTH the natural Heaviside-Lorentz (HL, hbar=c=1) form used in the paper's
Section 4 and in an SI-usable normalized form, with the relationship documented.

Reference equations (paper):
    Eq. (1)  Schwinger field / intensity
    Eq. (7)  xi = 2 alpha^2 / (45 m_e^4)  [HL units, hbar=c=1]
"""

import numpy as np

# ----------------------------------------------------------------------
# Fundamental SI constants (CODATA-style values)
# ----------------------------------------------------------------------
c = 2.99792458e8            # speed of light [m/s]
hbar = 1.054571817e-34      # reduced Planck [J s]
h = 6.62607015e-34          # Planck [J s]
e = 1.602176634e-19         # elementary charge [C]
m_e = 9.1093837015e-31      # electron mass [kg]
m_p = 1.67262192369e-27     # proton mass [kg]
eps0 = 8.8541878128e-12     # vacuum permittivity [F/m]
mu0 = 1.25663706212e-6      # vacuum permeability [N/A^2]
alpha = 7.2973525693e-3     # fine-structure constant (~1/137)

# Electron rest energy
m_e_c2_J = m_e * c**2                 # [J]
m_e_c2_eV = m_e_c2_J / e             # [eV]  ~ 5.11e5


# ----------------------------------------------------------------------
# Schwinger scales  --  Eq. (1)
# ----------------------------------------------------------------------
# Critical (Schwinger) electric field  E_S = m_e^2 c^3 / (e hbar)   [V/m]
E_S = (m_e**2 * c**3) / (e * hbar)            # [V/m]
E_S_Vcm = E_S * 1e-2                          # [V/cm]

# Critical magnetic field B_cr = m_e^2 c^2 / (e hbar) = E_S / c     [T]
B_cr = E_S / c                                # [T]

# --- Schwinger intensity: convention note (IMPORTANT) -----------------
# There are two conventions in the literature differing by a factor of 2:
#
#   (a) time-averaged plane wave:  I = (1/2) eps0 c E^2  -> 2.32e29 W/cm^2
#   (b) Di Piazza et al. RMP 2012: I_S = eps0 c E_S^2    -> 4.65e29 W/cm^2
#
# The paper's Eq. (1) WRITES form (a) but QUOTES the value 4.6e29 of form (b),
# and Table 1 (sec 6) uses 4.6e29. They are inconsistent by x2. We adopt the
# Di Piazza convention (b) so that the code matches the paper's stated value
# and Table 1; the recommended manuscript fix is to drop the 1/2 in Eq. (1).
I_S_planewave_Wcm2 = 0.5 * eps0 * c * E_S**2 * 1e-4   # convention (a)
I_S_SI = eps0 * c * E_S**2                            # convention (b) [W/m^2]
I_S_Wcm2 = I_S_SI * 1e-4                              # [W/cm^2]  ~ 4.6e29


# ----------------------------------------------------------------------
# Heisenberg-Euler coupling  --  Eq. (7)
# ----------------------------------------------------------------------
# In HL natural units (hbar = c = 1):  xi = 2 alpha^2 / (45 m_e^4).
# Here m_e is the electron mass in energy units. We expose the numeric
# prefactor (2 alpha^2 / 45) which multiplies 1/m_e^4 and which sets the
# 1 : 7 invariant structure used throughout Section 4.
XI_PREFACTOR = 2.0 * alpha**2 / 45.0          # dimensionless coefficient

# Equivalent written with e^4:  xi = e^4 / (360 pi^2 m_e^4)  (alpha=e^2/4pi in HL)
XI_PREFACTOR_e4 = 1.0 / (360.0 * np.pi**2)    # multiplies e^4 / m_e^4


def xi_HL():
    """xi in HL units with m_e set to 1 (energy units). Returns the prefactor.

    The physically meaningful, dimensionless ratios (1:7 in the Lagrangian,
    2:7 in P_vac/M_vac) are independent of the choice of m_e normalization;
    this function returns the leading coefficient 2 alpha^2 / 45.
    """
    return XI_PREFACTOR


# ----------------------------------------------------------------------
# Dimensionless field amplitude a0
# ----------------------------------------------------------------------
def a0_from_intensity(I_Wcm2, wavelength_um=0.8):
    """Normalized vector potential a0 = e E / (m_e c omega).

    Uses the standard relation a0 ~= 0.85 * sqrt(I[1e18 W/cm^2]) * lambda[um]
    (peak, linear polarization).
    """
    return 0.85 * np.sqrt(I_Wcm2 / 1e18) * wavelength_um


def intensity_from_a0(a0, wavelength_um=0.8):
    """Inverse of a0_from_intensity (linear polarization)."""
    return ((a0 / (0.85 * wavelength_um))**2) * 1e18


# ----------------------------------------------------------------------
# Field <-> intensity helpers (SI)
# ----------------------------------------------------------------------
def E_from_intensity(I_Wcm2):
    """Peak E field [V/m] from intensity [W/cm^2], I = (1/2) eps0 c E^2."""
    I_SI = I_Wcm2 * 1e4
    return np.sqrt(2.0 * I_SI / (eps0 * c))


def intensity_from_E(E_Vm):
    """Intensity [W/cm^2] from peak E field [V/m]."""
    I_SI = 0.5 * eps0 * c * E_Vm**2
    return I_SI * 1e-4


if __name__ == "__main__":
    print("Schwinger / QED scales (computed, not hard-coded)")
    print(f"  E_S   = {E_S_Vcm:.3e} V/cm   (paper: 1.32e16)")
    print(f"  B_cr  = {B_cr:.3e} T")
    print(f"  I_S   = {I_S_Wcm2:.3e} W/cm^2 (paper value, Di Piazza conv.: 4.6e29)")
    print(f"  I_S(1/2 conv) = {I_S_planewave_Wcm2:.3e} W/cm^2 (Eq.1 formula as written)")
    print(f"  m_e c^2 = {m_e_c2_eV:.4e} eV")
    print(f"  xi prefactor (2 alpha^2/45) = {XI_PREFACTOR:.6e}")
    print(f"  xi prefactor (1/360 pi^2)   = {XI_PREFACTOR_e4:.6e}")
