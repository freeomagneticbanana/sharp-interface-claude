#!/bin/sh
# Download the third-party Basilisk sandbox files this solver builds on (A. Limare's level-set/embedded
# boundary code and B. Ghigo's embed extensions), and apply the one-line compatibility patch needed for
# the Feb-2025 Basilisk release.  They are not redistributed here; see their pages on basilisk.fr for licences.
#
#   usage: ./fetch_deps.sh [destination]      (default: this directory, shared/)
#
# Files land at shared/alimare/... and shared/ghigo/src/...; model/ finds them through BASILISK_INCLUDE_PATH=shared.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HERE}"
FILES="
alimare/LS_curvature.h
alimare/LS_recons.h
alimare/LS_reinit.h
alimare/alex_functions.h
alimare/double_embed-tree.h
alimare/simple_discretization.h
alimare/weno2.h
alimare/weno5.h
ghigo/src/mydiffusion.h
ghigo/src/myembed-tree-moving.h
ghigo/src/myembed.h
ghigo/src/mypoisson.h
ghigo/src/myquadratic.h
"
for f in $FILES; do
  mkdir -p "$DEST/$(dirname "$f")"
  curl -sSfL -o "$DEST/$f" "https://basilisk.fr/sandbox/$f?raw"
  echo "fetched $f"
done
( cd "$DEST" && patch -p1 < "$HERE/patches/LS_curvature_dirty.patch" )
echo "dependencies in $DEST"
