/**
# Planar front with evolving temperature (thermosolutal, NC = 1)

Exercises the coupled path: unknowns (c_l, V, T) at the interface from the tie line, Gibbs-Thomson, the
solute balance and the Stefan balance; heat diffuses in both phases. Latent heat rho L = T_m * S (entropy
of fusion density of the free energy). No exact solution is used: the output reports the front position,
interface temperature and solute inventory. */

#define NC 1
#include "../mc_solver.h"

#define TM 1000.
#define TINIT 990.
#define CINF 0.12
#define S0 0.002

double c_star, c_sG;

int main()
{
  size (2.);
  MAXLEVEL = 7; MINLEVEL = 4;
  init_grid (1 << MAXLEVEL);
  TOLERANCE = 1.e-8; NITERMIN = 4;

  memset (&MC, 0, sizeof MC);
  MC.S.Tref = MC.L.Tref = TM; MC.S.g0T = S0; MC.S.H[0][0] = 4.; MC.L.H[0][0] = 2.; MC.L.c0[0] = 0.05;
  MC.Dl[0][0] = 1.; MC.Ds[0][0] = 0.;
  MC.gam.gamma_m = 1e-4;
  MC.evolveT = 1;
  MC.kL = MC.kS = 1.;                  // conductivities
  double Cvol = 100.;                  // volumetric heat capacity -> alpha = k / C
  MC.alphaL = MC.alphaS = MC.kL/Cvol;
  MC.rhoLat = TM*S0;                   // rho L = T_m * entropy of fusion
  MC.Vscale = 5.; MC.Escale = 0.01; MC.Tscale = 1.;
  if (mc_setup (&MC)) return 1;

  NB_width = 8.*L0/(1 << MAXLEVEL);
  DT = 0.3*sq (L0/(1 << MAXLEVEL)); DT_LS = 0.45*L0/(1 << MAXLEVEL);
  mc_dt_max = DT;
  CFL = 0.5;
  run();
}

static double front_x (void)
{
  double a = 0.02, b = L0 - 0.02, y0 = L0/2.;
  for (int it = 0; it < 60; it++) {
    double m = 0.5*(a + b);
    if (interpolate (dist, m, y0) < 0.) a = m; else b = m;
  }
  return 0.5*(a + b);
}

event init (i = 0)
{
  foreach()
    dist[] = x - 0.15;
  mc_init_geometry();
  double cL[NC] = {CINF}, cS[NC] = {0.05};
  mc_set_uniform (cL, true);
  mc_set_uniform (cS, false);
  scalar TL = WL[NC].s, TS = WS[NC].s;       // temperature fields follow the NC solute modes
  foreach() { TL[] = TINIT; TS[] = TINIT; }
  boundary (liquid_fields); boundary (solid_fields);
  restriction (liquid_fields); restriction (solid_fields);
}

event report (t += 0.005; t <= 0.04)
{
  double inv[NC]; mc_inventory (inv);
  /* interface temperature: mean over interface cells */
  double Tsum = 0., n = 0.;
  foreach (serial) if (interfacial (point, cs)) { Tsum += Tint[]; n += 1.; }
  fprintf (stdout, "%g %.6f %.6f %.6e %d\n", t, front_x(), n > 0. ? Tsum/n : 0., inv[0], mc_nfail);
  fflush (stdout);
}

event end (t = 0.04) { return 1; }
