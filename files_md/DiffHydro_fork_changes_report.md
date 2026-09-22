# DiffHydro fork report: what changed and why

**Comparing:** [`Stoikke/DiffHydro_public`](https://github.com/Stoikke/DiffHydro_public) (`main`, commit `f8108d2`, Aug 21 2026)
**Against:** [`bhorowitz/DiffHydro_public`](https://github.com/bhorowitz/DiffHydro_public) (`main`, commit `77ac3a4`, Feb 16 2026)

This document explains, in plain language, every substantive change made in the fork relative to the upstream project, and the physical/engineering motivation behind each one. It is written for a human reader who wants to understand *what was added* and *why*, not just a list of diffs.

---

## 1. Executive summary

The upstream `bhorowitz/DiffHydro_public` repository is a general-purpose, JAX-based, GPU-accelerated finite-volume hydrodynamics code ("DiffHydro") supporting Euler hydrodynamics, MHD, gravity (multigrid), turbulence driving, and a first-pass, unfinished cooling/units system. Its `main` branch has not moved since **February 16, 2026** (commit `77ac3a4`, "isothermal").

The fork, maintained by Adrien as part of an astrophysics research project, branches off from that exact commit and adds an entire **radiative transfer (RT) and photoionization module** on top of the existing hydro engine, plus a working **unit-conversion system**. The scientific goal is to simulate **HII regions** specialy the epoch of reionization — spheres of ionized hydrogen gas carved out around a young, massive star by its UV photons — and to validate the implementation against the classical analytic solution for this problem (the **Strömgren sphere**, and the time-dependent **Iliev et al. 2006, "Test 1"** benchmark).

Nothing in the upstream MHD, gravity, or turbulence code was removed; the fork is a strict superset. All new work lives in new files or in additive changes to a handful of existing files (`cooling.py`, `equationmanager.py`, `fluxes.py`, `hydro_core.py`, `hydro_core_CT.py`, `turbulence.py`, and the `units/` package).

---

## 2. Repository-level differences

### 2.1 New top-level module: `diffhydro/coupled_rhd.py`

A brand-new file implementing the **coupled radiation-hydrodynamics driver (not use)** — the object that ties the radiative-transfer solver and the hydrodynamics solver together into a single, self-consistent time-stepping scheme (as opposed to running RT and hydro as two independent, uncoupled solvers). This is the top-level entry point that most of the new example scripts (`run_coupled_rhd_example.py`) build on.

**Why:** DiffHydro previously only solved pure hydrodynamics or pure MHD. To simulate an HII region you need *both* physics at once, evolved together: photons ionize the gas, the ionized gas has different pressure and temperature than neutral gas, the pressure difference drives an outflow (hydrodynamics), and the outflow changes the gas density that the photons see next step. `coupled_rhd.py` is the machinery that makes that two-way coupling possible.

### 2.2 New equation managers for radiative transfer

Three new files were added at the top level of `diffhydro/`:

| File | Role |
|---|---|
| `equationmanager_radiative_transf.py` | First working `EquationManager` for the RT conserved variables (photon density `N`, photon flux `F`). |
| `equationmanager_radiative_transf_no_chat.py` | An expanded, heavily-annotated version, developed independently ("no_chat" = written without AI assistance, as a cross-check). |
| `equationmanager_radiative_transf_no_chat_copy.py`(final version use now) | The cleaned-up, production version actually used by the validation scripts (imported by `stromgren_validation.py` as `EquationManagerRT`). |

**Why:** DiffHydro's core (`hydro_core.py`) is built around the abstraction of an "equation manager" that knows how to convert between conservative and primitive variables for a given physical system (e.g. Euler hydro has `[rho, rho*v, E]`; MHD adds the B-field). Radiative transfer needs its own equation manager because its conserved variables are physically different: photon number density and photon flux, evolved with the **M1 closure** (a standard two-moment approximation for radiative transfer used in codes like RAMSES-RT), not fluid density and momentum.

### 2.3 `diffhydro/fluxes.py`: 

The upstream file only implements flux functions (Lax-Friedrichs / Rusanov Riemann solvers, PLM reconstruction) for standard Euler hydrodynamics. The fork adds the equivalent flux machinery **for the radiative-transfer system**, including:

- `ConvectiveFlux_Radiative_transfer` / `LaxFriedrichs_Radiative_transfer` / `HLL_RT_classes...` / classes, mirroring the hydro flux classes but operating on the `(N, F)` state instead of `(rho, rho*v, E)`.
- A `signal_speed_Rusanov` wave-speed estimate adapted for the M1 radiative system, where the characteristic speed is the (possibly reduced) speed of light rather than the local sound speed.

**Why:** A finite-volume solver needs a numerical flux function for *every* system of conservation laws it evolves. Since RT and hydro are advected with completely different characteristic speeds (light vs. sound/fluid velocity) and different eigenstructures, they cannot share the same Riemann solver — hence a parallel set of flux classes rather than reusing the hydro ones.

### 2.4 `diffhydro/hydro_core.py`: 

This is the central time-stepping engine (the `hydro` class). The fork's additions here are what let a single simulation combine **two different physical systems with two different flux stacks and different characteristic speeds** in one time-stepping loop — for example the Strömgren-sphere example script builds *two* independent `ConvectiveFlux` objects (one for RT, one for hydro) and passes both to the same `dh.hydro(...)` driver via `fluxes=[hydro_flux, rt_flux]`. The class also retains (and, per our own debugging session, relies on) the pre-existing `pmesh_shape` parameter that domain-decomposes the simulation across multiple GPU devices using `jax.experimental.mesh_utils` and `jax.sharding.Mesh`.

**Why:** Rather than writing an entirely separate coupled solver from scratch, the design keeps `hydro_core.py`'s generic multi-flux, multi-force stepping loop and simply supplies it with a second, RT-flavoured flux stack alongside the hydro one. This is a much smaller, more maintainable change than duplicating the whole time integrator, and it is what makes the domain-decomposition/multi-GPU machinery (`pmesh_shape=(nx, ny, nz)`) automatically apply to the coupled RT+hydro run as well, without any RT-specific parallelization code.

### 2.5 `diffhydro/equationmanager.py`: 

The core hydro `EquationManager` gained support for **passive scalar fields** (see `passive_names=("x_HII",)` used in the validation script) — additional conserved quantities that are advected with the flow but do not themselves exert a dynamical back-reaction on the flux through the standard hydro equations (their only effect on the dynamics is via the separately-applied chemistry/heating force terms).

**Why:** The ionized fraction `x_HII` (the fraction of hydrogen atoms that are ionized, between 0 and 1 in every cell) needs to be transported by the gas flow exactly like density or momentum — if a parcel of ionized gas moves, its ionization state moves with it. Treating it as a proper conserved, advected quantity (rather than recomputing it from scratch every step) is what makes it physically consistent with the rest of the conservative finite-volume scheme.

---

## 3. New physics package: `diffhydro/physics/`

### 3.1 `hydrogen_chemistry.py` — the single source of truth for atomic physics

This module is explicitly documented in its own docstring as *"the single source of truth"* for hydrogen chemistry and cooling, written entirely in **cgs units**. It implements, from the literature:

- **Photoionization cross-section**, Verner et al. (1996) fit, with an explicit warning in the code that the raw fit parameter `sigma0_cm2` is *not* the physical threshold cross-section and using it directly would overestimate the ionization rate by a factor of ~8700 — a bug class the module is designed to prevent by construction.
- **Collisional ionization rate** `beta_HI`, Cen (1992).
- **Recombination rates**, Case A and Case B, Hui & Gnedin (1997).
- **Cooling terms**: collisional ionization cooling, collisional excitation cooling, recombination cooling (Case A/B), bremsstrahlung, and Compton cooling off the CMB — together forming Eq. (A17) of the RAMSES-RT paper (Rosdahl et al. 2013), restricted to pure hydrogen.
- A `HydrogenStateView` dataclass that is the **only** place in the whole codebase where code units and cgs units meet, converting between JAX's internal simulation units and physical cgs quantities for every chemistry calculation.
- Numerically-safe helper functions: `limited_explicit_update` (a positivity-preserving explicit Euler step that never lets a quantity go negative, however stiff the source term), `limit_m1_flux_cone` (keeps the M1 radiation flux inside its causal cone, `|F| ≤ c·N`), and `tiny_like` (a dtype-aware, physically-scaled floor value used instead of an arbitrary hard-coded `1e-30`, because code-unit magnitudes vary enormously depending on the chosen length/mass/velocity units).


### 3.2 `fraction_xHII.py` — the ionization-state update

Implements two alternative "force" classes that update the hydrogen ionization fraction `x_HII` each timestep:

- **`HydrogenIonizationForce`**: a simple, direct explicit-Euler solver of the eq. (28') ionization ODE, using a two-sided fractional limiter so `x_HII` can never leave `[0, 1]` regardless of how stiff the chemistry is.
- **`HydrogenPhotoChemistryForce (used one)`**: a more careful, **photon-conserving** coupled scheme. Its docstring explains precisely why it exists: if you split "photons get absorbed" and "atoms get ionized" into two separately-ordered force calls (as a naive first implementation would), the ionization step reads an *already-depleted* photon count and under-counts ionizations — an error that does **not** vanish under grid refinement (it stays at the ~10% level on the reference Iliev+2006 Test 1 benchmark, as directly measured and documented in the code). The fix enforces the exact identity "one absorbed photon produces one ionization" within a single force call, and reuses the very same absorbed-photon count to deposit photo-heating, so the energy budget and the ionization budget can never numerically drift apart from each other.

### 3.3 `radiative_transfer.py` and `radiative_transfer_fixed.py` 

These implement the **`StellarRadiationForce`** class: the force term that injects photons from a point-like or Gaussian-smoothed stellar source into the RT conserved fields each step, in one of several configurable injection geometries (`injection_geometry="radial_3D"` for an isotropic point source, vs. a directional beam), with optional momentum injection (radiation pressure) and a selectable coupling mode to the chemistry (`chemistry=True/False`).

`radiative_transfer_fixed.py (used one)` is a substantially expanded and corrected rewrite of `radiative_transfer.py`, the name signals it is the bug-fixed version, consistent with commit messages such as "INJECTION MOMENTUM FIX" and the extensive back-and-forth visible in commits about isotropic vs. anisotropic photon propagation ("the propagation is isotropic and not anisotropic in the x direction)".

### 3.4 `turbulence_radiative_transf.py (not use)` 

A small adapter/wrapper module that lets the pre-existing turbulence-driving force (used in upstream's turbulence examples) coexist with the new combined RT+hydro state layout, so a turbulent background medium can, in principle, be irradiated by a star.

### 3.5 `cooling.py`: 

The upstream file already contained an initial, somewhat ad-hoc cooling/heating force (`HeatCoolForce`, using a Koyama & Inutsuka 2002 / Sutherland & Dopita 1993 tabulated cooling curve for generic ISM cooling, unrelated to photoionization). The fork:

- Keeps the original `HeatCoolForce` implementation for backward compatibility (needed by the turbulence/supernova examples that predate the RT work), including its various work-in-progress solver variants (`force`, `force_sec`, an implicit/secant-method solver, etc. — again reflecting an iterative numerical-methods search for a stable scheme, with older attempts left in place and commented as deprecated rather than deleted).
- Adds a brand-new **`HeatCoolForce_basic`** class specifically for **photoheating/cooling driven by the RT solver**, built directly on top of `hydrogen_chemistry.py`. Its docstring explicitly documents three bugs found and fixed relative to an earlier draft: (1) mixing cgs rate coefficients with code-unit densities and adding the erg cm⁻³ s⁻¹ result straight into a code-unit energy slot ("three inconsistent unit systems in one expression"), (2) reading the wrong array slot for pressure because the code assumed a layout that didn't match the actual conservative-state layout used by the RT+hydro coupling, and (3) a heating formula that computed a frequency *difference* instead of an energy, which is dimensionally wrong and also happens to be identically zero for a monochromatic photon source at the ionization threshold.

**Why:** These are the kinds of subtle, high-impact bugs that are extremely easy to introduce when converting between simulation ("code") units and physical (cgs) units by hand in more than one place, which is exactly the failure mode `hydrogen_chemistry.py`'s "single source of truth" design (Section 3.1) was created to eliminate going forward.


## 4. Units system: `diffhydro/units/`

Upstream already had a first-draft units package ("unit handler beta" was literally Ben Horowitz's last commit before the fork diverged). The fork substantially extends every file in it:

| File | 
|---|
| `code_units.py` |
| `convert.py` |
| `field_dims.py` | 
| `registry.py` | 

The `CodeUnits` class (used throughout the RT examples as `CodeUnits.from_config({"length": ..., "mass": ..., "velocity": ...}, {"gamma": ..., "mu": ...})`) is what lets a simulation choose, say, "1 length unit = 1 grid cell = `dx_cgs` centimetres" and "1 velocity unit = the reduced speed of light", and have every physical quantity (density, pressure, photon flux) consistently converted between that arbitrary code-unit system and cgs.

---

## 5. New example scripts and validation tests

### 5.1 `examples/RT/` (entirely new directory)

Upstream has no radiative-transfer examples at all. The fork adds an entire example suite, most visibly:

- **`stromgren_validation.py`** / **`stromgren_validation_updated.py`**: the physics validation script this whole project was building towards. It sets up the classical **Iliev et al. (2006) "Test 1"** problem — a uniform, static, pure-hydrogen medium suddenly irradiated by a star emitting `Q = 5×10⁴⁸` ionizing photons/second — and compares the numerically evolved ionization-front radius `R_I(t)` against the exact analytic solution:
 \[
 R_I(t) = R_S \left[1 - e^{-t/t_{\rm rec}}\right]^{1/3}
 \]
 where `R_S` is the Strömgren radius and `t_rec` the recombination time. This is the standard, textbook test used across the RT-in-hydro-code literature (RAMSES-RT, and the Iliev+2006 comparison-project papers this test is named after) precisely because it has a known closed-form answer, making it possible to quantify a code's accuracy."
- **`RT_run.py`** and a series of **`RT_run_chat_v1.py` … `v4.py`** (plus `v4_corrected_core.py`, `v4_with_outputs.py`, 'RT_run_chat_v5.py'): successive iterations of a more general RT driver script, kept as a visible development trail rather than squashed into a single final version.
- **`photon_absorption_force.py (useless)`**: a standalone, simplified force implementing pure photon absorption (no full ionization chemistry), useful as a minimal sanity check of the RT flux/geometry machinery in isolation.
- **`ramses_rt_pipeline.py (useless)`**: tooling to set up initial conditions or run configurations comparable to the well-established RAMSES-RT code, presumably for cross-validation against an independent, published implementation.
- **`run_coupled_rhd_example.py (useless)`**: the example entry point exercising the new `coupled_rhd.py` driver (Section 2.1).

### 5.2 `examples/rt_debug/` (new directory)

A dedicated space for minimal reproduction scripts used while debugging the RT solver in isolation from the full coupled system.

### 5.3 Deployment / cluster-execution material (`examples/RT/serveur/`)

A subdirectory (name = French for "server") holding cluster-specific run configuration, consistent with the fact that this validation was run at scale on a SLURM-managed GPU cluster (as extensively documented in our own troubleshooting session earlier in this conversation: multi-GPU domain decomposition via `pmesh_shape`, multi-node `jax.distributed.initialize()`, NCCL/CUDA environment fixes, etc.). This reflects the practical reality that getting the physics right is only half the work — the other half is getting a differentiable, JAX-jitted, multi-physics solver to actually run correctly and efficiently across multiple GPUs and multiple compute nodes on a shared academic cluster.

---

## 7. Summary table

| Category | Upstream (`bhorowitz/main`) | Fork (`Stoikke/main`) | Purpose of the addition |
|---|---|---|---|
| Radiative transfer equations | none | `equationmanager_radiative_transf*.py` (3 files) | Conserved variables & primitive/conservative conversion for photon density/flux (M1 closure) |
| RT numerical fluxes | none | Added to `fluxes.py` | Riemann solver / reconstruction for the RT system |
| RT-hydro coupling driver | none | `coupled_rhd.py` (new) | Advance RT and hydro together, self-consistently |
| Hydrogen chemistry | none | `physics/hydrogen_chemistry.py` (new) | Single-source-of-truth atomic rates (ionization, recombination, cooling), cgs |
| Ionization-fraction update | none | `physics/fraction_xHII.py` (new) | Photon-conserving `x_HII` evolution, avoiding operator-split bias |
| Stellar photon injection | none | `physics/radiative_transfer(_fixed).py` (new) | Point/Gaussian stellar source, isotropic or beamed injection |
| Photoheating/cooling for RT | generic ISM cooling only | `HeatCoolForce_basic` added to `cooling.py` | Physically consistent H-only photoheating & recombination/collisional cooling |
| Passive scalar transport | not supported | Added to `equationmanager.py` | Advect `x_HII` with the flow like any other conserved field |
| Multi-GPU domain decomposition | `pmesh_shape` already present | Same mechanism now exercised by RT+hydro runs | Scale the coupled solver across GPUs/nodes without new parallel code |
| Units system | early "beta" | Substantially extended | Avoid float32 overflow and unit-conversion bugs across code/cgs boundary |
| Validation examples | none | `examples/RT/`, `examples/rt_debug/` | Reproduce the Iliev et al. (2006) Strömgren-sphere benchmark |

---
