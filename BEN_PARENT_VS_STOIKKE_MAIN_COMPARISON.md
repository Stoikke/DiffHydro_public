# Comparison Report: Ben Parent Repository vs Stoikke Main

This document compares:

- **Ben parent repository main**: `upstream/main` (`77ac3a4`)
- **Stoikke main branch**: `origin/main` (`f8108d2`)

Date of comparison: 2026-08-26.

## 1) Repository relationship

- The two branches currently have **no merge base** (`git merge-base upstream/main origin/main` returns none).
- This means they are treated as **unrelated histories** by Git in this clone.
- Divergence count (`git rev-list --left-right --count upstream/main...origin/main`):
  - `upstream/main` only: **76 commits**
  - `origin/main` only: **3 commits**

## 2) Global change volume (tree-to-tree diff)

From `git diff upstream/main..origin/main`:

- **799 changed files**
- **776 added**, **23 modified**, **0 deleted**
- **+20,735,055 / -898 lines**

## 3) Where the changes are concentrated

### By top-level directory (file count)

- `examples/`: **682 files**
- `Images/`: **67 files**
- `diffhydro/`: **27 files**
- `files_md/`: **15 files**
- `tests/`: **4 files**
- repository root: **4 files**

### By changed lines (additions + deletions)

- `examples/`: **20,721,366 lines**
- `diffhydro/`: **7,776 lines**
- `files_md/`: **4,168 lines**
- root: **1,313 lines**
- `tests/`: **883 lines**

### By file extension (line volume)

- `.csv`: **20,000,020 lines**
- `.txt`: **547,872 lines**
- `.ipynb`: **159,124 lines**
- `.py`: **22,679 lines**
- `.md`: **4,502 lines**

## 4) Functional/code-level differences (main points)

Compared with Ben’s parent `main`, Stoikke `main` adds a large radiative-transfer and coupled hydro/radiation workstream, including:

- New core module: `diffhydro/coupled_rhd.py`
- New radiative-transfer equation managers:
  - `diffhydro/equationmanager_radiative_transf.py`
  - `diffhydro/equationmanager_radiative_transf_no_chat.py`
  - `diffhydro/equationmanager_radiative_transf_no_chat_copy.py`
- New RT/chemistry physics modules:
  - `diffhydro/physics/radiative_transfer.py`
  - `diffhydro/physics/radiative_transfer_fixed.py`
  - `diffhydro/physics/hydrogen_chemistry.py`
  - `diffhydro/physics/fraction_xHII.py`
  - `diffhydro/physics/turbulence_radiative_transf.py`
- Updates to existing numerical/core files:
  - `diffhydro/fluxes.py`
  - `diffhydro/hydro_core.py`
  - `diffhydro/hydro_core_CT.py`
  - `diffhydro/solver/riemann_solver.py`
  - `diffhydro/solver/recon.py`
  - `diffhydro/solver/limiter.py`
  - `diffhydro/equationmanager.py`
  - `diffhydro/physics/cooling.py`
- Unit-system updates in `diffhydro/units/*`
- New debugging helper: `diffhydro/utils/debug_checks.py`

## 5) Test and documentation differences

- New tests:
  - `tests/test_flux_block_alignment.py`
  - `tests/test_hydrogen_chemistry.py`
  - `tests/test_radiative_transfer_units.py`
- Updated test:
  - `tests/test_unit_handler_frontend.py`
- Documentation additions are significant under `files_md/` (mostly French-language internal/project docs), plus `README.md` changes.

## 6) Data and artifact-heavy additions

The majority of the line volume comes from generated or artifact-like files, especially:

- Large RT field CSV cubes under `examples/RT/Images/...`
- Many PNG images (`.png`)
- Execution logs (`run_log.txt`)
- Notebook outputs (`.ipynb`)

So, the comparison is dominated numerically by data/output assets, while the substantive software changes are concentrated in `diffhydro/`, `tests/`, and selected `examples/RT` scripts.

## 7) Commit-level difference summary

### Commits present in Stoikke `main` and not in Ben parent `main`

1. `f8108d2` Merge pull request #17 from Stoikke/Coupling_gas_photons
2. `30ce019` synch serv ready
3. `358b040` Merge pull request #16 from Stoikke/Coupling_gas_photons

### Commits present in Ben parent `main` and not in Stoikke `main`

- **76 commits** (not listed in full here), including recent upstream items such as:
  - `77ac3a4` isothermal
  - `f6e5702` unit handler beta
  - `4f9a0fe` small additions related to MHD
  - `526a3d4` updated MHD code + working example...
  - `f494300` minor tweaks to cooling

---

If you want, I can generate a second version of this report with:

1) the full 76-commit upstream-only list, and  
2) a filtered comparison that excludes generated assets (`*.csv`, images, notebook outputs) to focus only on source code and tests.
