/* Plain-C check of mc_physics.h against the Python reference (sharp_interface/thermo). Build:
   gcc -O2 -DNC=2 mc_physics_check.c -o check -lm */
#include "../mc_physics.h"

static void fill (QPhase * p, double g0, double g0T, const double * a, const double * aT,
                  const double H[NC][NC], const double * c0)
{
  p->g0 = g0; p->g0T = g0T; p->Tref = 1000.;
  for (int i = 0; i < NC; i++) { p->a[i] = a[i]; p->aT[i] = aT[i]; p->c0[i] = c0[i]; for (int j = 0; j < NC; j++) p->H[i][j] = H[i][j]; }
  qp_finalize (p);
}

int main (void)
{
#if NC == 2
  double Hs[2][2] = {{4.,.5},{.5,3.}}, Hl[2][2] = {{2.,.3},{.3,1.5}};
  double a0[2] = {0,0}, aT0[2] = {0,0}, c0s[2] = {0,0}, c0l[2] = {.05,.03};
  McParams P; memset (&P, 0, sizeof P);
  fill (&P.S, 0., 0.002, a0, aT0, Hs, c0s);
  fill (&P.L, 0., 0., a0, aT0, Hl, c0l);
  /* Onsager mobilities -> diffusivities */
  double Ll[2][2] = {{.5,.1},{.1,.4}}, Ls[2][2] = {{0,0},{0,0}};
  modes_from_onsager (Ll, Hl, P.Dl, &P.ml);
  double Dtest[2][2];
  /* check D = V diag(lam) Vinv */
  double err = 0.;
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
    double s = 0.; for (int k = 0; k < 2; k++) s += P.ml.V[i][k]*P.ml.lam[k]*P.ml.Vinv[k][j];
    err = fmax (err, fabs (s - P.Dl[i][j])); Dtest[i][j] = s; }
  printf ("onsager modes: |V lam Vinv - D| = %.2e   lam = %.8f %.8f\n", err, P.ml.lam[0], P.ml.lam[1]);
  /* closed-form 2x2 non-symmetric */
  double Dn[2][2] = {{1.0,.2},{.05,.4}}; Modes m2; modes_from_D (Dn, &m2);
  err = 0.; for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) { double s = 0.; for (int k = 0; k < 2; k++) s += m2.V[i][k]*m2.lam[k]*m2.Vinv[k][j]; err = fmax (err, fabs (s - Dn[i][j])); }
  printf ("2x2 closed form: err = %.2e lam = %.8f %.8f\n", err, m2.lam[0], m2.lam[1]);
  double cl[2] = {.138254, .097391}, cs[2], dO;
  mc_tie (&P.S, &P.L, cl, 990., cs, &dO);
  printf ("tie: cs = %.12f %.12f   dOmega = %.12e\n", cs[0], cs[1], dO);
  printf ("L = %.12e\n", ({ double mu[2]; qp_mu (&P.L, cl, 990., mu); qp_omega (&P.L, mu, 990.); }));
  SurfEnergy se = {0.3, 0.02, 0.2, -1e-3, 1000., 4};
  printf ("gamma = %.12f stiff = %.12f\n", se_gamma (&se, 0.7, 990.), se_stiffness (&se, 0.7, 990.));
  (void) Ls;
#endif
  return 0;
}
