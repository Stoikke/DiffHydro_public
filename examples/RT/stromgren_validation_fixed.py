"""Static, isothermal Stromgren-sphere validation (Iliev et al. Test 1).

This is an independent corrected copy of ``stromgren_validation_updated.py``.
It validates precisely the analytic R-type solution

    R_I(t) = R_S [1 - exp(-t / t_rec)]^(1/3).

The reference assumes *both* a static, uniform gas density and a prescribed
temperature.  Consequently, this benchmark transports photons but deliberately
does not apply ``hydro_flux``: an isothermal ionized gas has twice as many
particles as neutral hydrogen and would otherwise develop a pressure gradient,
expand, lower its recombination rate, and no longer obey this analytic curve.

For a physical radiation-hydrodynamic expansion, retain ``hydro_flux``, turn
on physical heating/cooling, and compare against an RHD (e.g. Spitzer) solution
instead -- not against the Test-1 curve used here.

Usage (from the repository root):

    conda run -n jax-gpu python examples/RT/stromgren_validation_fixed.py
    N=96 TEND=5 conda run -n jax-gpu python examples/RT/stromgren_validation_fixed.py

Outputs are isolated in ``examples/RT/Images/stromgren_validation_fixed``.
"""

import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "0"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import diffhydro as dh
from diffhydro.equationmanager_radiative_transf_no_chat_copy import (
    EquationManager as EquationManagerRT,
)
from diffhydro.physics import hydrogen_chemistry as hchem
from diffhydro.physics.fraction_xHII import HydrogenPhotoChemistryForce
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.units import CodeUnits


# ---------------------------------------------------------------------------
# Iliev et al. (2006), Test 1: pure H, static, isothermal, case B.
# ---------------------------------------------------------------------------
# N=96 resolves R_S with 24 cells in the deliberately wide 4 R_S box.  It is
# the minimum default that gives a meaningful 3-D shape plot; use N=128 for a
# radius/convergence measurement intended for a figure or report.
N = int(os.environ.get("N", 96))
n_H_cgs = 1.0e-3
T_K = 1.0e4
Q_phot = 5.0e48
GAMMA = 5.0 / 3.0
TEND_REC = float(os.environ.get("TEND", 5.0))
# This is the preferred value, not an unconditional value.  The actual RSLA
# below is raised automatically when the first resolved I-front would outrun
# it at the selected N.
RSLA_REQUESTED = float(os.environ.get("RSLA", 2.0e-2))
CFL = float(os.environ.get("CFL", 0.30))
N_SNAP = int(os.environ.get("N_SNAP", 40))

# The old box ended at 1.43 R_S from the source.  The numerical solver is
# periodic internally, so keep the front two R_S away from each boundary.  At
# this density the neutral buffer is very optically thick, making wrap-around
# photons a measurable zero rather than an unspoken source of extra ionization.
BOX_HALF_SIZE_RS = float(os.environ.get("BOX_HALF_SIZE_RS", 2.0))
F_MAX = float(os.environ.get("F_MAX", 0.95))
SOURCE_CENTER_CODE = 0.5 * (N - 1)

if N < 32:
    raise ValueError("Use N >= 32; otherwise R_S is resolved by too few cells.")
if not 0.0 < F_MAX <= 1.0:
    raise ValueError("F_MAX must lie in (0, 1].")

alpha_B = float(hchem.alpha_B_HII_cgs(T_K))
R_S = (3.0 * Q_phot / (4.0 * np.pi * alpha_B * n_H_cgs**2)) ** (1.0 / 3.0)
t_rec = 1.0 / (alpha_B * n_H_cgs)
box_cgs = 2.0 * BOX_HALF_SIZE_RS * R_S
dx_cgs = box_cgs / N
t_end = TEND_REC * t_rec

# For an initially neutral one-cell sphere, dR/dt = Q/(4 pi r^2 n_H).
# A fixed RSLA=0.02 is adequate at N=64 in this 4 R_S box, but it is too low
# at N=96 because dx is smaller.  Use a small safety factor above that speed.
v_front_at_dx = Q_phot / (4.0 * np.pi * dx_cgs**2 * n_H_cgs)
RSLA_MIN = 1.20 * v_front_at_dx / hchem.C_LIGHT_CGS
RSLA = max(RSLA_REQUESTED, RSLA_MIN)
c_red_cgs = RSLA * hchem.C_LIGHT_CGS

# These units make rho_code == 1 in the initially uniform medium, keeping all
# hydro and chemistry quantities well scaled in float64.
M_unit_cgs = n_H_cgs * hchem.MH_CGS * dx_cgs**3
cu = CodeUnits.from_config(
    {
        "length": f"{dx_cgs} cm",
        "mass": f"{M_unit_cgs} g",
        "velocity": f"{c_red_cgs} cm/s",
    },
    {"gamma": GAMMA, "mu": 1.0},
)

dx_code = 1.0
c_code = 1.0
dt_code = CFL / (3.0 * c_code / dx_code)
time_code = t_end / cu.T_cgs
n_steps = int(math.ceil(time_code / dt_code))
n_super_step = n_steps + 10

print("Backend:", jax.default_backend())
print("=" * 76)
print("Iliev Test 1: static gas + prescribed T + case-B chemistry")
print(f"N={N}, R_S/dx={R_S / dx_cgs:.2f}, box half-size={BOX_HALF_SIZE_RS:.2f} R_S")
print(f"Q={Q_phot:.3e} s^-1, n_H={n_H_cgs:.3e} cm^-3, T={T_K:.0f} K")
print(f"alpha_B={alpha_B:.6e} cm^3 s^-1, R_S={R_S:.6e} cm, t_rec={t_rec:.6e} s")
print(f"RSLA requested/used={RSLA_REQUESTED:.3e}/{RSLA:.3e}, "
      f"c_red/v_front(dx)={c_red_cgs / v_front_at_dx:.2f}")
if RSLA > RSLA_REQUESTED:
    print("RSLA was raised automatically so the early resolved front is not "
          "artificially limited by the speed of light.")
print(f"dt={dt_code * cu.T_cgs:.6e} s, steps={n_steps}")
if R_S / dx_cgs < 24.0:
    print("WARNING: R_S is resolved by fewer than 24 cells; the x_HII=0.5 "
          "contour will visibly follow the Cartesian grid. Use N>=96 for a "
          "shape diagnostic, and N>=128 for a convergence result.")
print("=" * 76)


# ---------------------------------------------------------------------------
# Transport.  RT is evolved; gas fluxes are intentionally absent (see module
# docstring).  The hydro block remains in the combined state for rho*x_HII and
# E_tot, which the chemistry force owns.
# ---------------------------------------------------------------------------
eq_rt = EquationManagerRT(light_speed=c_code, mesh_shape=(N, N, N), eps=1e-30, debug=False)
eq_hydro = dh.EquationManager(
    gamma=GAMMA,
    n_cons=6,
    passive_names=("x_HII",),
    mesh_shape=(N, N, N),
    eps=1e-30,
)


class StateBlockFlux:
    """Embed a flux that operates on one contiguous block of the state."""

    def __init__(self, base_flux, state_slice):
        self.base_flux = base_flux
        self.state_slice = state_slice
        self.dx_o = base_flux.dx_o

    def flux(self, sol, ax, params, accumulated_flux):
        local = self.base_flux.flux(sol[self.state_slice], ax, params, accumulated_flux)
        return jnp.zeros_like(sol).at[self.state_slice].set(local)

    def timestep(self, sol):
        return self.base_flux.timestep(sol[self.state_slice])


rt_flux = StateBlockFlux(
    dh.ConvectiveFlux_Radiative_transfer(
        eq_rt,
        dh.HLL_Radiative_transfer_Local(eq_rt, dh.signal_speed_Rusanov),
        dh.PLM(limiter="VANLEER"),
        dx=dx_code,
    ),
    slice(0, eq_rt.n_cons),
)


class CellCenteredStromgrenSource(StellarRadiationForce):
    """An exactly reflection-symmetric point source on an even Cartesian grid.

    A physical source at the geometric centre of an even-size mesh lies on the
    intersection of eight cells.  Injecting it in only ``N//2,N//2,N//2``
    shifts the source by half a cell in every direction and seeds the visible
    four-/six-lobed pattern in a 2-D cut.  This source distributes the photon
    rate equally over the eight central cells and assigns each one the radial
    M1 flux appropriate to its octant.  The summed photon rate remains Q.
    """

    def force(self, i, sol, params, dt):
        # N_gamma is a number density in code volume.  ``dx_code == 1``, but
        # retain the base helper so the dimensional relation remains explicit.
        dN_total = self.get_N_gamma_stromgen_sphere() * dt
        dN_cell = dN_total / 8.0
        flux_component = self.beam_reduced_flux * self.light_speed * dN_cell / math.sqrt(3.0)

        lo = N // 2 - 1
        hi = N // 2
        for ix, sx in ((lo, -1.0), (hi, 1.0)):
            for iy, sy in ((lo, -1.0), (hi, 1.0)):
                for iz, sz in ((lo, -1.0), (hi, 1.0)):
                    sol = sol.at[0, ix, iy, iz].add(dN_cell)
                    sol = sol.at[1, ix, iy, iz].add(sx * flux_component)
                    sol = sol.at[2, ix, iy, iz].add(sy * flux_component)
                    sol = sol.at[3, ix, iy, iz].add(sz * flux_component)

        # The sum of an old radiation field and a new source can otherwise
        # leave the M1 cone even though the injected component itself is valid.
        return self._clip_to_m1_cone(sol), params


stellar = CellCenteredStromgrenSource(
    dx=dx_code,
    injection_mode="stromgren",
    stromgren_rate=Q_phot * cu.T_cgs,
    injection_momentum=True,
    injection_geometry="radial_3D",
    gaussian_star=True,
    beam_momentum_scaling="legacy_c2_source2",
    beam_reduced_flux=F_MAX,
    eq=eq_rt,
    hydro_eq=eq_hydro,
    cu=cu,
    chemistry=False,  # sole owner of injection; coupled chemistry is below
)
chemistry = HydrogenPhotoChemistryForce(
    stellar,
    case="B",
    collisional=False,
    max_frac=0.9,
    include_heating=False,
    include_cooling=False,
    # This applies the benchmark thermostat *after* each x_HII update.
    fixed_temperature_K=T_K,
)

sim = dh.hydro(
    n_super_step=n_super_step,
    fluxes=[rt_flux],  # Deliberately no hydro_flux: analytic reference is static.
    forces=[stellar, chemistry],
    dx=dx_code,
    max_dt=dt_code,
)


# ---------------------------------------------------------------------------
# Conservative initial state: RT [N, Fx, Fy, Fz], hydro
# [rho, rho vx, rho vy, rho vz, E_tot, rho*x_HII].
# ---------------------------------------------------------------------------
rho_code = n_H_cgs * hchem.MH_CGS / cu.rho_cgs
p_code = n_H_cgs * hchem.KB_CGS * T_K / cu.P_cgs
state = jnp.zeros((eq_rt.n_cons + eq_hydro.n_cons, N, N, N), dtype=jnp.float64)
state = state.at[4].set(rho_code)
state = state.at[8].set(p_code / (GAMMA - 1.0))

# ``center_code`` is the physical source position.  ``center_slice`` chooses
# one of the two central planes for an image; both are mirror-equivalent.
center_slice = N // 2
params = {
    "star_masses": jnp.array([1.0]),
    "star_ages": jnp.array([0.0]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions": jnp.array([[center_slice] * 3], dtype=jnp.int32),
}


def x_hii(state_):
    return np.asarray(chemistry.view.xHII(state_), dtype=np.float64)


def ionized_radius(x_):
    """Volume-equivalent radius.  This is the diagnostic matching Test 1."""
    v_ion = float(np.sum(x_, dtype=np.float64)) * dx_cgs**3
    return (3.0 * v_ion / (4.0 * np.pi)) ** (1.0 / 3.0)


def radial_profile(x_):
    grid = np.indices((N, N, N), dtype=np.float64)
    radius = np.sqrt(sum((grid[a] - SOURCE_CENTER_CODE) ** 2 for a in range(3)))
    shell = np.floor(radius).astype(np.int32)
    counts = np.bincount(shell.ravel())
    sums = np.bincount(shell.ravel(), weights=x_.ravel())
    return np.arange(len(counts)), sums / np.maximum(counts, 1)


@jax.jit
def recombination_rate_case_b(state_):
    """Total case-B recombination rate [s^-1] at the prescribed T."""
    x_ = chemistry.view.xHII(state_)
    n_h = chemistry.view.number_densities_cgs(state_, x_)[0]
    return alpha_B * jnp.sum((n_h * x_) ** 2) * dx_cgs**3


snap_every = max(1, n_steps // N_SNAP)


@jax.jit
def run_chunk(state_, params_, recombinations_, step0):
    """Advance a fixed number of steps and integrate recombinations per step."""

    def body(j, carry):
        s, p, n_rec = carry
        (s, p) = sim._hydrostep(step0 + j, (s, p), dt_code)
        n_rec = n_rec + recombination_rate_case_b(s) * dt_code * cu.T_cgs
        return s, p, n_rec

    return jax.lax.fori_loop(0, snap_every, body, (state_, params_, recombinations_))


# ---------------------------------------------------------------------------
# Evolution and conservation diagnostics.
# ---------------------------------------------------------------------------
time_code = 0.0
step = 0
recombinations = jnp.array(0.0, dtype=jnp.float64)
times = [0.0]
radii = [0.0]
closures = [1.0]
temperature_error = [0.0]
boundary_x = [0.0]

while step < n_steps:
    steps_this = min(snap_every, n_steps - step)
    if steps_this == snap_every:
        state, params, recombinations = run_chunk(state, params, recombinations, step)
    else:
        # The final chunk has a distinct static loop length, so compile it only once.
        def last_body(j, carry):
            s, p, n_rec = carry
            s, p = sim._hydrostep(step + j, (s, p), dt_code)
            n_rec = n_rec + recombination_rate_case_b(s) * dt_code * cu.T_cgs
            return s, p, n_rec

        state, params, recombinations = jax.lax.fori_loop(
            0, steps_this, last_body, (state, params, recombinations)
        )
    step += steps_this
    time_code += steps_this * dt_code

    x = x_hii(state)
    t_s = time_code * cu.T_cgs
    radius = ionized_radius(x)
    n_hii = n_H_cgs * float(np.sum(x, dtype=np.float64)) * dx_cgs**3
    n_gamma = float(np.sum(np.asarray(state[0], dtype=np.float64))) * dx_code**3
    n_emitted = Q_phot * t_s
    closure = (n_hii + n_gamma + float(recombinations)) / max(n_emitted, 1e-300)
    temp = np.asarray(chemistry.view.temperature_K(state), dtype=np.float64)

    # Any ionization at the physical edges exposes the periodic numerical grid.
    edge = np.concatenate(
        [x[0].ravel(), x[-1].ravel(), x[:, 0].ravel(), x[:, -1].ravel(),
         x[:, :, 0].ravel(), x[:, :, -1].ravel()]
    )
    edge_x = float(np.max(edge))

    times.append(t_s)
    radii.append(radius)
    closures.append(closure)
    temperature_error.append(float(np.max(np.abs(temp / T_K - 1.0))))
    boundary_x.append(edge_x)
    print(
        f"step {step:6d}/{n_steps} t/t_rec={t_s/t_rec:7.4f} "
        f"R/R_S={radius/R_S:8.5f} analytic={(1.0-np.exp(-t_s/t_rec))**(1.0/3.0):8.5f} "
        f"budget={closure:8.5f} dT/T={temperature_error[-1]:.2e} edge_x={edge_x:.2e}"
    )

times = np.asarray(times)
radii = np.asarray(radii)
closures = np.asarray(closures)
analytic = R_S * (1.0 - np.exp(-times / t_rec)) ** (1.0 / 3.0)
ratio = np.divide(radii, analytic, out=np.ones_like(radii), where=analytic > 0.0)
relative_error = np.abs(radii[1:] - analytic[1:]) / analytic[1:]

rho_final = np.asarray(state[4], dtype=np.float64)
speed_final = np.sqrt(sum(np.asarray(state[k], dtype=np.float64) ** 2 for k in (5, 6, 7)))
print("=" * 76)
print(f"final R_num/R_analytic = {ratio[-1]:.6f}")
print(f"mean |relative radius error| = {100.0 * relative_error.mean():.3f}%")
print(f"photon-budget closure final = {closures[-1]:.6f}")
print(f"rho/rho0 min/max = {rho_final.min()/rho_code:.12f} / {rho_final.max()/rho_code:.12f}")
print(f"max |rho v| = {speed_final.max():.3e} (must remain zero in Test 1)")
print(f"max boundary x_HII = {max(boundary_x):.3e}")
print("=" * 76)


# ---------------------------------------------------------------------------
# Outputs: curve, final shape, radial profile, and conservation history.
# ---------------------------------------------------------------------------
out_dir = os.path.join(REPO_ROOT, "examples/RT/Images/stromgren_validation_fixed")
os.makedirs(out_dir, exist_ok=True)
np.savetxt(
    os.path.join(out_dir, f"history_N{N}.csv"),
    np.column_stack([times, radii, analytic, ratio, closures, temperature_error, boundary_x]),
    delimiter=",",
    header="time_s,radius_cm,analytic_radius_cm,ratio,photon_budget_closure,max_abs_dT_over_T,max_boundary_xHII",
    comments="",
)

x_final = x_hii(state)
r_shell, x_shell = radial_profile(x_final)
np.savetxt(
    os.path.join(out_dir, f"radial_profile_N{N}.csv"),
    np.column_stack([r_shell, r_shell * dx_cgs, x_shell]),
    delimiter=",",
    header="radius_cells,radius_cm,xHII_shell_mean",
    comments="",
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
t_fine = np.linspace(0.0, times[-1], 400)
axes[0].plot(t_fine / t_rec, (1.0 - np.exp(-t_fine / t_rec)) ** (1.0 / 3.0), "k-", lw=2, label="analytic")
axes[0].plot(times / t_rec, radii / R_S, "o-", color="tab:red", ms=3, label=f"DiffHydro N={N}")
axes[0].set(xlabel=r"$t/t_{rec}$", ylabel=r"$R_I/R_S$", title="Static isothermal Test 1")
axes[0].grid(alpha=0.3)
axes[0].legend()

extent_pc = [-BOX_HALF_SIZE_RS * R_S / 3.0857e18, BOX_HALF_SIZE_RS * R_S / 3.0857e18] * 2
image = axes[1].imshow(x_final[:, :, center_slice].T, origin="lower", cmap="magma", vmin=0.0, vmax=1.0, extent=extent_pc)
angle = np.linspace(0.0, 2.0 * np.pi, 300)
axes[1].plot(R_S / 3.0857e18 * np.cos(angle), R_S / 3.0857e18 * np.sin(angle), "c--", lw=1.5, label=r"$R_S$")
axes[1].set(xlabel="x [pc]", ylabel="y [pc]", title=fr"$x_{{HII}}$ at ${times[-1]/t_rec:.1f},t_{{rec}}$")
axes[1].legend()
fig.colorbar(image, ax=axes[1], label=r"$x_{HII}$")

axes[2].plot(r_shell * dx_cgs / R_S, x_shell, color="tab:purple")
axes[2].axvline(1.0, ls="--", color="k", lw=1, label=r"$R_S$")
axes[2].set(xlabel=r"$r/R_S$", ylabel=r"shell mean $x_{HII}$", ylim=(-0.02, 1.02), title="Final radial ionization profile")
axes[2].grid(alpha=0.3)
axes[2].legend()

fig.tight_layout()
figure_path = os.path.join(out_dir, f"stromgren_static_isothermal_N{N}.png")
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Outputs written to {out_dir}")
