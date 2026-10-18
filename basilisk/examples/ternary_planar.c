/**
# Isothermal planar ternary front (NC = 2, frozen temperature), D_s = 0

Same set-up as `tests/test_ternary_planar.py`: unequal liquid diffusivities D = (1, 0.4), full 2x2 free-energy
Hessians with off-diagonal coupling. The exact similarity solution X = 2 lambda sqrt(D_1 t) with
liquid c_i = c_inf_i + (c*_i - c_inf_i) erfc(x / 2 sqrt(D_i t)) / erfc(lambda_i), lambda_i = lambda sqrt(D_1/D_i),
is obtained here by a small Newton solve for (lambda, mu*_1, mu*_2). */

#define NC 2
#include "../mc_solver.h"

#define TM 1000.
#define TISO 990.
#define T0 0.01
#define TEND 0.05

double cinf[NC] = {0.08, 0.05}, Dd[NC] = {1.0, 0.4};
double lam_ex, c_star[NC], c_sG[NC];

static double Tfun (double x, double y, double z, double tt) { return TISO; }

static double erfcx_ (double z) { return exp (z*z)*erfc (z); }

static void exact_residual (const double * z, double * F)
{
  double lam = z[0], mu[NC] = {z[1], z[2]}, cl[NC], cs[NC];
  qp_c_of_mu (&MC.L, mu, TISO, cl);
  qp_c_of_mu (&MC.S, mu, TISO, cs);
  F[0] = qp_omega (&MC.L, mu, TISO) - qp_omega (&MC.S, mu, TISO);
  for (int i = 0; i < NC; i++) {
    double li = lam*sqrt (Dd[0]/Dd[i]);
    F[i + 1] = lam*sqrt (M_PI)*sqrt (Dd[0]/Dd[i])*(cl[i] - cs[i]) - (cl[i] - cinf[i])/erfcx_ (li);
  }
}

static void exact_setup (void)
{
  double z[3] = {0.5, 0.2, 0.2}, F[3];
  for (int it = 0; it < 60; it++) {
    exact_residual (z, F);
    double J[9], b[3], dz[3];
    for (int j = 0; j < 3; j++) {
      double zp[3] = {z[0], z[1], z[2]}, Fp[3], h = 1e-7;
      zp[j] += h; exact_residual (zp, Fp);
      for (int i = 0; i < 3; i++) J[i*3 + j] = (Fp[i] - F[i])/h;
    }
    for (int i = 0; i < 3; i++) b[i] = -F[i];
    mc_gauss (3, J, b, dz);
    for (int i = 0; i < 3; i++) z[i] += dz[i];
  }
  lam_ex = z[0];
  double mu[NC] = {z[1], z[2]};
  qp_c_of_mu (&MC.L, mu, TISO, c_star);
  qp_c_of_mu (&MC.S, mu, TISO, c_sG);
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
  double Hs[2][2] = {{4., .5}, {.5, 3.}}, Hl[2][2] = {{2., .3}, {.3, 1.5}};
  for (int i = 0; i < NC; i++)
    for (int j = 0; j < NC; j++) { MC.S.H[i][j] = Hs[i][j]; MC.L.H[i][j] = Hl[i][j]; }
  MC.S.Tref = MC.L.Tref = TM; MC.S.g0T = 0.002;
  MC.L.c0[0] = 0.05; MC.L.c0[1] = 0.03;
  for (int i = 0; i < NC; i++) MC.Dl[i][i] = Dd[i];     // Ds = 0
  MC.gam.gamma_m = 1e-4;
  MC.Vscale = 5.; MC.Escale = 0.01;
  if (mc_setup (&MC)) return 1;
  Tprescribed = Tfun;
  exact_setup();
  fprintf (stderr, "# lambda = %.8f  c* = (%.6f, %.6f)  c_s = (%.6f, %.6f)\n", lam_ex, c_star[0], c_star[1], c_sG[0], c_sG[1]);

  NB_width = 8.*L0/(1 << MAXLEVEL);
  DT = 0.3*sq (L0/(1 << MAXLEVEL)); DT_LS = 0.45*L0/(1 << MAXLEVEL);
  mc_dt_max = DT;
  CFL = 0.5;
  run();
}

event init (i = 0)
{
  double s0 = 2.*lam_ex*sqrt (Dd[0]*T0);
  foreach()
    dist[] = x - s0;
  mc_init_geometry();
  foreach() {
    double cl[NC], cs[NC];
    for (int i = 0; i < NC; i++) {
      double li = lam_ex*sqrt (Dd[0]/Dd[i]);
      cl[i] = cinf[i] + (c_star[i] - cinf[i])*erfc (x/(2.*sqrt (Dd[i]*T0)))/erfc (li);
      cs[i] = c_sG[i];
    }
    mc_set_cell (point, cl, true);
    mc_set_cell (point, cs, false);
  }
  boundary (liquid_fields); boundary (solid_fields);
  restriction (liquid_fields); restriction (solid_fields);
}

event report (t += 0.005; t <= TEND - T0)
{
  double X = front_x(), Xe = 2.*lam_ex*sqrt (Dd[0]*(t + T0));
  double inv[NC]; mc_inventory (inv);
  fprintf (stdout, "%g %.8f %.8f %.3e %.8e %.8e %d\n", t, X, Xe, (X - Xe)/Xe, inv[0], inv[1], mc_nfail);
  fflush (stdout);
}

event stop (t = TEND - T0) { return 1; }
