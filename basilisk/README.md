# Multicomponent sharp-interface solidification on Basilisk

A sharp-interface solver for **N-component** alloy solidification built on Basilisk's embedded boundaries and the
hybrid level-set / embedded-boundary method of Limare, Popinet, Josserand, Xue & Ghigo (JCP 2022). The interface
is a true jump surface: each phase has its own fields (two-fluid representation), the level set is only the
geometry, and every interface condition is imposed on the cut cells, not smeared.

## What it solves

- NC = N-1 independent solute components in **both** phases, with (matrix) interdiffusion, `dc/dt = div(D grad c)`.
- Interface thermodynamics from user-supplied **quadratic phase free energies** g_s(c,T), g_l(c,T): tie lines from
  equal diffusion potentials, grand-potential driving force.
- Interface conditions at every cut cell (solved by a small local Newton problem):
  - solute balance `V [c] = D_l g_l + D_s g_s`,
  - Gibbs-Thomson with kinetics: `omega_l - omega_s = (gamma + gamma'') kappa + beta(theta) V`,
  - Stefan balance `rho L V = k_s dT_s/dN - k_l dT_l/dN` (evolving temperature only).
- Surface energy `gamma(theta,T) = gamma0(T) (1 + eps cos(n (theta - theta0)))` with `gamma0 = gamma_m + gamma_T (T - T_ref)`,
  anisotropic kinetic coefficient, 2D.
- Temperature: **prescribed** (`Tprescribed(x,y,z,t)`) or **evolving** (heat equations in both phases + Stefan balance).
- Optional continuous-growth solute trapping (`MC.trapping`, `MC.VD[]`).

## Files

| file | role |
|---|---|
| `mc_physics.h` | plain C, no Basilisk: free energies, tie lines, anisotropy, diffusion modes, **local interface Newton problem**. Checked against the Python reference (`tests/mc_physics_check.c`). |
| `mc_solver.h` | Basilisk glue: fields, gradient responses, interface solve, level-set advection, field fill, implicit modal diffusion, adaptation, output. |
| `examples/` | `planar_binary.c`, `ternary_planar.c` (exact similarity solutions built in), `crystal_frozenT.c` (anisotropic crystal, isothermal or `-DDIRECTIONAL`), `thermosolutal_planar.c` (evolving T), `crystal3d.c` (octree). |
| `fetch_deps.sh`, `patches/` | download of the third-party sandbox files + one-line compatibility patch. |

## Conventions

- `dist > 0` is liquid. `cs` is the **liquid** volume fraction (Basilisk's "fluid"), `1 - cs` the solid.
- N is the unit normal from solid to liquid; V > 0 is growth of the solid; `kappa = div N` (> 0 for a convex solid).
  `curvature_LS` returns `-div N`; the solver negates it.
- Basilisk's embedded normal points out of the phase being solved: gradients for the liquid are `d/dn_e = -d/dN`.
- `dv()` already contains the fluid fraction (metric `cm = cs`); integrals over both phases use the full cell volume.

## Method (per time step)

1. `mc_stability`: gradient responses of every field -> local interface Newton solve -> V; continuous extended speed; time step.
2. `mc_advance`: RK3-WENO5 advection and reinitialisation of `dist`, fractions and curvature, extrapolation into
   cells uncovered by the motion, interface solve on the **new** geometry (Dirichlet data), implicit diffusion of
   all modes in both phases.

The interdiffusion matrix of each phase is diagonalised once (`modes_from_D`: NC = 1, NC = 2 any real-eigenvalue
matrix, symmetric D for any NC; `modes_from_onsager` for the free-energy form `D = L H`, any NC), so the coupled
components become independent modes solved with Basilisk's implicit embedded diffusion.
The embedded gradient is affine in the Dirichlet value, so the local Newton problem is exact in the interface unknowns.

## Build

```
./fetch_deps.sh                      # sandbox dependencies -> ./deps/basilisk-sandbox (needs basilisk.fr)
make BASILISK=/path/to/basilisk/src planar_binary
./examples/planar_binary
```
Developed against the Basilisk release of 2025-02-26 (`NEWS`: "macros"). qcc needs `config` (copy `config.gcc`) and the
`gl` libraries built (`make -k qcc` in `src/`, plus `src/gl`). `NC` is a compile-time constant (`#define NC 2` before the include).

## Status and known limitations

**Unit-level checks done:** thermodynamics, tie lines, anisotropy and mode decomposition match the Python reference to
round-off. **Smoke runs done** (no crashes, no Newton failures): binary and ternary planar fronts (front position within
~0.5% of the exact similarity solution at 256^2; composition profiles in both phases match the exact solution, liquid error < 1e-3), thermosolutal planar front, anisotropic 2D crystal, 3D octree crystal.

**Not validated yet:** convergence orders, solid diffusion (`D_s > 0`) against the exact solution, curved fronts
(Frank/radial, Mullins-Sekerka growth rate), dendrite tip velocity, the evolving-temperature path against an analytic
Stefan solution, trapping, and everything in 3D beyond "it runs".

Known limitations / approximations:
- Solute conservation: in the planar runs the total drifts by less than +0.1% (binary +0.09%, ternary +0.06%/+0.04%). It is not exact by construction (emerged cells are filled by extrapolation, not a conservative update), and the curved-front (crystal) runs lose 7-8% of the solute by t = 0.08 (level 7). Diagnosis so far: with the interface frozen the liquid volume still creeps by ~2e-5 per step (the level-set reinitialisation/fraction update shrinks a small seed), and the solid volume gained per step is 83-97% of V*A*dt. Adaptation, the number of reinitialisation iterations, and the diffusion solver are not the cause (uniform grid and 0/2/10 reinit iterations give the same drift). Planar fronts, which are axis-aligned, are unaffected. The level-set volume error on small/curved interfaces needs a fix (e.g. volume-correcting reinitialisation, or deriving the solute exchange from the actual change of `cs`).
- First-order in time; the front is advanced with a lagged speed and a small time step is needed for the explicit curvature
  (`mc_dt_max`), as in the sandbox solver.
- Constant interdiffusion matrices per phase; no composition-dependent D.
- Anisotropy is 2D only (`mc_setup` refuses it in 3D). The `(dgamma/dT) d_n T` term of a temperature-dependent surface energy is not included.
- Trapping uses the continuous-growth interpolation per component with the *equilibrium* driving force (no kinetic liquidus shift).
- **Adaptation must include every transported field**: with an under-resolved thermal layer, adapting on the solute alone let the
  evolving-temperature case run away; `mc_adapt` therefore blends and adapts all fields (set `mc_adapt_tol_field[]`).
- Cells swept by the front are filled with `fill_value` (weighted plane fit over neighbours that held the phase at the previous step). `embed_extrapolate_ls` is deliberately not used: with `emerged = false` its quadratic fit is singular for a front sweeping a whole row of cells and silently returns 0; with `emerged = true` it reads stale values of cells emerging in the same step.
- The user's `init` event must run before the solver events; `mc_solver.h` declares an empty `init` first to guarantee it.
- Third-party code (Limare, Ghigo sandboxes) is downloaded, not redistributed; check their licences on basilisk.fr.
