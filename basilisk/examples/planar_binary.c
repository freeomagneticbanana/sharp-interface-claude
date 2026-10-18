/**
# Isothermal planar binary front (frozen temperature), D_s = 0

Same set-up as the Python reference test (`tests/planar.py`): quadratic free energies
g_s = S (T - Tm) + 1/2 Hs c^2,  g_l = 1/2 Hl (c - 0.05)^2, T = 990, D_l = 1, far-field c = 0.1.
The exact similarity solution X = 2 lambda sqrt(D_l t) is used to initialise the fields at t0 and to compare. */

#define NC 1
#include "../mc_solver.h"

#define TM 1000.
#define TISO 990.
#define CINF 0.10
#define T0 0.01
#define TEND 0.05

double c_star, c_sG, lam_ex;

static double Tfun (double x, double y, double z, double tt) { return TISO; }

static void exact_setup (void)
{
  /* liquidus composition at TISO: Newton on the driving force */
  double cl[NC] = {0.1}, cs[NC], dO;
  for (int it = 0; it < 50; it++) {
    mc_tie (&MC.S, &MC.L, cl, TISO, cs, &dO);
    double cp[NC] = {cl[0] + 1e-7}, cs2[NC], dO2;
    mc_tie (&MC.S, &MC.L, cp, TISO, cs2, &dO2);
    cl[0] -= dO/((dO2 - dO)/1e-7);
  }
  mc_tie (&MC.S, &MC.L, cl, TISO, cs, &dO);
  c_star = cl[0]; c_sG = cs[0];
  /* lambda: lam sqrt(pi) (c* - c_sG) = (c* - c_inf)/erfcx(lam) */
  double lo = 1e-6, hi = 3.;
  for (int it = 0; it < 200; it++) {
    double m = 0.5*(lo + hi);
    double f = m*sqrt (M_PI)*(c_star - c_sG) - (c_star - CINF)/(exp (m*m)*erfc (m));
    if (f > 0.) hi = m; else lo = m;
  }
  lam_ex = 0.5*(lo + hi);
}

static double front_x (void)
{
  double a = 0.05, b = L0 - 0.05, y0 = L0/2.;
  for (int it = 0; it < 60; it++) {
    double m = 0.5*(a + b);
    if (interpolate (dist, m, y0) < 0.) a = m; else b = m;
  }
  return 0.5*(a + b);
}

int main()
{
  size (2.);
  MAXLEVEL = 7; MINLEVEL = 4;
  init_grid (1 << MAXLEVEL);
  TOLERANCE = 1.e-8; NITERMIN = 4;

  memset (&MC, 0, sizeof MC);
  /* solid: g = S (T - Tm) + 1/2 Hs c^2 ;  liquid: g = 1/2 Hl (c - 0.05)^2 */
  MC.S.Tref = TM; MC.S.g0T = 0.002; MC.S.H[0][0] = 4.; MC.S.c0[0] = 0.;
  MC.L.Tref = TM; MC.L.H[0][0] = 2.; MC.L.c0[0] = 0.05;
  MC.Dl[0][0] = 1.; MC.Ds[0][0] = 0.;
  MC.gam.gamma_m = 1e-4;               // small capillarity: removes the Mullins-Sekerka grid-scale instability
  MC.Vscale = 5.; MC.Escale = 0.01;
  if (mc_setup (&MC)) return 1;
  Tprescribed = Tfun;
  exact_setup();
  fprintf (stderr, "# c* = %.8f c_sG = %.8f lambda = %.8f\n", c_star, c_sG, lam_ex);

  nb_cell_NB = 8;
  NB_width = nb_cell_NB*L0/(1 << MAXLEVEL);
  DT = 0.3*sq (L0/(1 << MAXLEVEL)); DT_LS = 0.45*L0/(1 << MAXLEVEL);
  mc_dt_max = DT;
  CFL = 0.5;
  run();
}

event init (i = 0)
{
  double s0 = 2.*lam_ex*sqrt (MC.Dl[0][0]*T0);
  foreach()
    dist[] = x - s0;
  mc_init_geometry();
  scalar wl = WL[0].s, ws = WS[0].s;
  foreach() {
    wl[] = CINF + (c_star - CINF)*erfc (x/(2.*sqrt (MC.Dl[0][0]*T0)))/erfc (lam_ex);
    ws[] = c_sG;
  }
  boundary (liquid_fields); boundary (solid_fields);
  restriction (liquid_fields); restriction (solid_fields);
}

/* run() starts at t = 0, so the similarity time is tau = t + T0 */
event report (t += 0.005; t <= TEND - T0)
{
  double X = front_x(), Xe = 2.*lam_ex*sqrt (MC.Dl[0][0]*(t + T0));
  double inv[NC]; mc_inventory (inv);
  fprintf (stdout, "%g %.8f %.8f %.3e %.8e %d\n", t, X, Xe, (X - Xe)/Xe, inv[0], mc_nfail);
  fflush (stdout);
}

event stop (t = TEND - T0) { return 1; }
