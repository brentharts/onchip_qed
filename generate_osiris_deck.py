#!/usr/bin/env python3
"""
Generate OSIRIS input decks for the paper's plasma-mirror / hole-boring
campaign (Stages 2-3 of Section 5), parameterized by the drive amplitude a0.

This is the legitimate use of `osiris_utils`: it is a pre/post-processing IO
library, so we author a physically sensible base deck, then load it with
osiris_utils.InputDeckIO and emit one variant per a0 (and the matching focal
intensity from the paper's CHF gain). These decks are ready to run once the user
has access to an OSIRIS build (the semi-classical Heisenberg-Euler vacuum module
is a custom extension and is NOT in public OSIRIS -- see README).

The 1D over-dense mirror deck here reproduces the hole-boring / denting physics
that feeds the analytic CHF gain; a 2D variant is also emitted as a starting
point for the full focusing geometry.
"""

import os
import numpy as np
import _style, chf
import constants as k
import osiris_utils as ou

DECK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decks")
os.makedirs(DECK_DIR, exist_ok=True)

# over-dense plasma for the on-chip relativistic mirror
NE_OVER_NC = 90.0
LAMBDA_UM = 0.8


BASE_DECK_1D = r"""
! OSIRIS 1D input deck -- on-chip relativistic plasma mirror (hole-boring)
! Auto-generated template; a0 is overwritten per-variant by osiris_utils.
! Normalized units: time/length in 1/w0 and c/w0, density in n_c.

simulation
{
  algorithm = "standard",
}

node_conf
{
  node_number(1:1) = 1,
  if_periodic(1:1) = .false.,
}

grid
{
  nx_p(1:1) = 8000,
  coordinates = "cartesian",
}

time_step
{
  dt = 0.0095,
  ndump = 200,
}

space
{
  xmin(1:1) = 0.0,
  xmax(1:1) = 80.0,
  if_move(1:1) = .false.,
}

time
{
  tmin = 0.0,
  tmax = 120.0,
}

el_mag_fld
{
  smooth_type = "stand",
}

emf_bound
{
  type(1:2,1) = "open", "open",
}

particles
{
  num_species = 2,
}

species
{
  name = "electrons",
  num_par_max = 4000000,
  rqm = -1.0,
  num_par_x(1:1) = 200,
}

profile
{
  density = 90.0,
  profile_type = "piecewise-linear",
  num_x = 4,
  x(1:4,1) = 0.0, 40.0, 40.5, 80.0,
  fx(1:4,1) = 0.0, 0.0, 1.0, 1.0,
}

spe_bound
{
  type(1:2,1) = "open", "open",
}

species
{
  name = "ions",
  num_par_max = 4000000,
  rqm = 1836.0,
  num_par_x(1:1) = 200,
}

profile
{
  density = 90.0,
  profile_type = "piecewise-linear",
  num_x = 4,
  x(1:4,1) = 0.0, 40.0, 40.5, 80.0,
  fx(1:4,1) = 0.0, 0.0, 1.0, 1.0,
}

spe_bound
{
  type(1:2,1) = "open", "open",
}

zpulse
{
  a0 = 10.0,
  omega0 = 1.0,
  pol_type = 0,
  propagation = "forward",
  lon_type = "polynomial",
  lon_rise = 12.0,
  lon_flat = 6.0,
  lon_fall = 12.0,
  lon_start = 35.0,
}

diag_emf
{
  ndump_fac = 1,
  reports = "e1", "e2", "e3", "b3",
}

diag_species
{
  ndump_fac = 1,
  reports = "charge",
}
"""


def _write_base(path, deck_text):
    with open(path, "w") as f:
        f.write(deck_text.lstrip("\n"))


def generate_a0_scan(a0_values=(5, 10, 20, 50, 100)):
    base_path = os.path.join(DECK_DIR, "base_mirror_1d.txt")
    _write_base(base_path, BASE_DECK_1D)

    rows = []
    for a0 in a0_values:
        deck = ou.InputDeckIO(base_path)
        # drive amplitude (the campaign's scan variable)
        deck.set_param("zpulse", "a0", float(a0))
        # record the matched physics so the deck is self-documenting via a tag
        I_chf = chf.chf_gain(a0) * k.intensity_from_a0(a0, LAMBDA_UM)
        vhb = chf.v_holeboring_over_c(a0, NE_OVER_NC)
        deck.set_tag("a0_scan",
                     f"a0={a0}  n_e/n_c={NE_OVER_NC}  "
                     f"I_drive={k.intensity_from_a0(a0, LAMBDA_UM):.2e}W/cm2  "
                     f"I_CHF~{I_chf:.2e}W/cm2  v_HB/c={vhb:.3f}")
        out = os.path.join(DECK_DIR, f"mirror_1d_a0_{int(a0):03d}.txt")
        deck.print_to_file(out)
        rows.append((a0, k.intensity_from_a0(a0, LAMBDA_UM), I_chf, vhb,
                     os.path.basename(out)))
    return rows


def verify_roundtrip():
    """Confirm osiris_utils can parse a generated deck and read a0 back."""
    sample = os.path.join(DECK_DIR, "mirror_1d_a0_010.txt")
    d = ou.InputDeckIO(sample)
    return d.get_param("zpulse", "a0")


def main():
    rows = generate_a0_scan()
    print("Generated OSIRIS decks in decks/:\n")
    print(f"  {'a0':>4}  {'I_drive(W/cm2)':>15}  {'I_CHF(W/cm2)':>14}"
          f"  {'v_HB/c':>7}  file")
    for a0, Idr, Ichf, vhb, fn in rows:
        print(f"  {a0:>4}  {Idr:>15.2e}  {Ichf:>14.2e}  {vhb:>7.3f}  {fn}")
    a0_back = verify_roundtrip()
    print(f"\n  round-trip check: osiris_utils re-read a0 = {a0_back} "
          f"from a generated deck.")


if __name__ == "__main__":
    main()
