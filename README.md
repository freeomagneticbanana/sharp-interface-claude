# sharp-interface-claude

## Layout

| directory | contents |
|---|---|
| `sharp_interface/`, `tests/` | Python reference implementation (ghost-fluid level set, N components) and its exact-solution tests |
| `model/` | our Basilisk (C) implementation: `mc_physics.h`, `mc_solver.h`, examples, checks |
| `shared/` | downloaded third-party code (Limare/Ghigo sandboxes) + fetch script and patch; never edited by hand |
