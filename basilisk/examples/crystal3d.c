/**
# 3D isotropic solutal crystal in a supersaturated liquid (frozen temperature, octree)

Dimension-generic code path: octree grid, 3D normals/curvature, isotropic surface energy (anisotropy is
implemented in 2D only; `mc_setup` refuses a nonzero anisotropy when `dimension == 3`).

    usage: ./crystal3d [MAXLEVEL] [t_end] */

#include "grid/octree.h"
#define NC 1
#include "../mc_solver.h"

#define TM 1000.
#define TISO 990.
#define CINF 0.12
#define R0 0.06

double tend = 0.02;

static double Tfun (double x, double y, double z, double tt) { return TISO; }

int main (int argc, char ** argv)
{
  MAXLEVEL = argc > 1 ? atoi (argv[1]) : 5;
  if (argc > 2) tend = atof (argv[2]);
  MINLEVEL = 3;
  size (1.);
  init_grid (1 << MAXLEVEL);
  TOLERANCE = 1.e-8; NITERMIN = 4;

  memset (&MC, 0, sizeof MC);
  MC.S.Tref = MC.L.Tref = TM; MC.S.g0T = 0.002; MC.S.H[0][0] = 4.; MC.L.H[0][0] = 2.; MC.L.c0[0] = 0.05;
  MC.Dl[0][0] = 1.; MC.Ds[0][0] = 0.;
  MC.gam.gamma_m = 2e-4;
  MC.Vscale = 5.; MC.Escale = 0.01;
  if (mc_setup (&MC)) return 1;
  Tprescribed = Tfun;

  NB_width = 8.*L0/(1 << MAXLEVEL);
  DT = 0.3*sq (L0/(1 << MAXLEVEL)); DT_LS = 0.45*L0/(1 << MAXLEVEL);
  mc_dt_max = DT;
  CFL = 0.5;
  run();
}

event init (i = 0)
{
  double cl[NC] = {CINF}, cs[NC], dO;
  mc_tie (&MC.S, &MC.L, cl, TISO, cs, &dO);
  foreach()
    dist[] = sqrt (sq (x - 0.5) + sq (y - 0.5) + sq (z - 0.5)) - R0;
  mc_init_geometry();
  double cL[NC] = {CINF}, cS[NC] = {cs[0]};
  mc_set_uniform (cL, true);
  mc_set_uniform (cS, false);
}

event report (i += 20)
{
  double inv[NC]; mc_inventory (inv);
  fprintf (stderr, "i=%d t=%g solute=%.6e newton-failures=%d\n", i, t, inv[0], mc_nfail);
}

event end (t = tend) { return 1; }
