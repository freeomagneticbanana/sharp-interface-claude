/**
# Anisotropic solutal crystal in a supersaturated liquid, frozen temperature (2D)

A small seed grows in an undercooled binary liquid. The temperature is prescribed: isothermal by default, or
a moving gradient T = T_iso + G (y - V_p t) when compiled with `-DDIRECTIONAL`. The surface energy has
four-fold anisotropy (stiffness gamma + gamma''), plus optional kinetic undercooling.

    usage: ./crystal_frozenT [MAXLEVEL] [t_end] [eps4] [reinit iterations]

Output: `interface-NNNN.dat` (x y V T c_l c_s per interface cell) every `dt_out`. */

#define NC 1
#include "../mc_solver.h"

#define TM 1000.
#define TISO 990.
#define CINF 0.12
#define R0 0.05
#ifdef DIRECTIONAL
 #define GRAD 20.          // temperature gradient dT/dy
 #define VPULL 0.5         // pulling speed
#endif

double c_star, c_sG, tend = 0.2, dt_out = 0.02;
int nout = 0;

static double Tfun (double x, double y, double z, double tt)
{
#ifdef DIRECTIONAL
  return TISO + GRAD*(y - 0.1 - VPULL*tt);
#else
  return TISO;
#endif
}

int main (int argc, char ** argv)
{
  if (argc > 1) MAXLEVEL = atoi (argv[1]); else MAXLEVEL = 7;
  if (argc > 2) tend = atof (argv[2]);
  double eps4 = argc > 3 ? atof (argv[3]) : 0.02;
  if (argc > 4) itredist = atoi (argv[4]);       // reinitialisation iterations per step (default 10)
  MINLEVEL = 4;
  size (1.);
  init_grid (1 << MAXLEVEL);
  TOLERANCE = 1.e-8; NITERMIN = 4;

  memset (&MC, 0, sizeof MC);
  MC.S.Tref = MC.L.Tref = TM; MC.S.g0T = 0.002; MC.S.H[0][0] = 4.; MC.L.H[0][0] = 2.; MC.L.c0[0] = 0.05;
  MC.Dl[0][0] = 1.; MC.Ds[0][0] = 0.;
  MC.gam.gamma_m = 2e-4; MC.gam.eps = eps4; MC.gam.nfold = 4; MC.gam.theta0 = 0.;
  MC.kin.beta0 = 0.;       // set > 0 for kinetic undercooling
  MC.kin.nfold = 4;
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
  /* tie line at the initial temperature, for the solid seed composition */
  double cl[NC] = {CINF}, cs[NC], dO;
  mc_tie (&MC.S, &MC.L, cl, TISO, cs, &dO);
  c_sG = cs[0];
  foreach()
    dist[] = sqrt (sq (x - 0.5) + sq (y - 0.5)) - R0;
  mc_init_geometry();
  double cL[NC] = {CINF}, cS[NC] = {c_sG};
  mc_set_uniform (cL, true);
  mc_set_uniform (cS, false);
}

event output (t += dt_out; t <= tend)
{
  char name[64];
  sprintf (name, "interface-%04d.dat", nout++);
  FILE * fp = fopen (name, "w");
  mc_output_interface (fp);
  fclose (fp);
  double inv[NC]; mc_inventory (inv);
  fprintf (stderr, "t=%g step=%d interface-file=%s solute=%.6e newton-failures=%d\n", t, i, name, inv[0], mc_nfail);
}

/* liquid composition field (component 0) on a regular grid at the end, for plotting */
event fields (t = tend)
{
  FILE * fp = fopen ("field.dat", "w");
  scalar wl = WL[0].s, ws = WS[0].s;
  int n = 256;
  for (int a = 0; a < n; a++) {
    for (int b = 0; b < n; b++) {
      double xx = (a + 0.5)*L0/n, yy = (b + 0.5)*L0/n;
      double d = interpolate (dist, xx, yy);
      double c = d > 0. ? interpolate (wl, xx, yy) : interpolate (ws, xx, yy);
      fprintf (fp, "%g ", c);
    }
    fprintf (fp, "\n");
  }
  fclose (fp);
}

event end (t = tend) { return 1; }
