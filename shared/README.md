# shared/ — third-party code (not ours)

Everything in this directory that is not `fetch_deps.sh`, `patches/` or this file is **downloaded, unmodified
(apart from the listed patch) third-party code**, kept apart from our own code in `model/`.

| path | origin | used for |
|---|---|---|
| `alimare/` | A. Limare, `basilisk.fr/sandbox/alimare` | hybrid level-set / embedded-boundary method (advection, reinitialisation, reconstruction, curvature) |
| `ghigo/src/` | B. Ghigo, `basilisk.fr/sandbox/ghigo/src` | embedded-boundary extensions (`myembed.h`, quadratic extrapolation, implicit diffusion) |

Basilisk itself (the `qcc` compiler and `src/`) is *not* here: point `BASILISK=` at your own installation
(developed against the release of 2025-02-26).

## Fetching

```
./fetch_deps.sh        # downloads the files listed in the script into this directory and applies patches/
```

`patches/LS_curvature_dirty.patch` is the single compatibility fix for the 2025 Basilisk release (the removed `.dirty`
attribute). The downloaded directories (`alimare/`, `ghigo/`) are git-ignored; the script and patch are tracked, so a fresh
clone is restored with one command. Licences: see the pages on basilisk.fr (Basilisk is GPL).

`model/` reaches these files through `BASILISK_INCLUDE_PATH=shared` (set by `model/Makefile`); our code never edits them.
