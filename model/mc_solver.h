/**
# Multicomponent sharp-interface solidification on Basilisk (level set + embedded boundaries)

Extends the hybrid level-set/embedded-boundary method of Limare et al. (JCP 2022, `sandbox/alimare`) from a
single scalar pair (TL, TS) with a Stefan velocity to

- NC = N-1 independent solute components in *both* phases with (matrix) interdiffusion,
- interface compositions from phase free energies (tie lines, grand-potential driving force),
- orientation- and temperature-dependent surface energy and kinetic undercooling,
- either a prescribed ("frozen") temperature, or evolving temperatures with the Stefan balance.

## Method in one paragraph

Each phase has its own fields on the full grid (two-fluid representation): `cs` is the liquid volume
fraction (`dist > 0` is liquid), `1 - cs` the solid. The interdiffusion matrix of each phase is diagonalised
(`mc_physics.h`) so the NC coupled components become NC independent *modes* `w = Vinv c`, each solved with
Basilisk's implicit embedded-boundary diffusion with a Dirichlet value on the interface. At every interface
cell a small nonlinear system for (c_l, V[, T]) is solved: tie line + Gibbs-Thomson with kinetics + solute
balance [+ Stefan balance]. The one-sided embedded gradients depend *affinely* on the Dirichlet value, so the
local Newton problem is exact in the interface unknowns. The speed is extended to a continuous field and
the level set is advected (RK3-WENO5), reinitialised, and emerged cells are filled by extrapolation.

## Per-step order (events)

1. `mc_stability`: interface solve on the current geometry -> V; continuous speed; time step.
2. `mc_advance`: advect/reinitialise the level set, update fractions and curvature, fill emerged cells,
   interface solve on the *new* geometry (Dirichlet data), implicit diffusion of all modes in both phases.

Geometry first, then implicit transport on the new geometry (as in Bochkov, Pollock & Gibou, JCP 2023).

## Usage

```
#define NC 2                       // number of independent components, before the include
#include "mc_solver.h"
// in main(): fill MC (free energies, D, surface energy, ...), call mc_setup(&MC), set Tprescribed (frozen T)
// in event init: set dist[], then mc_init_geometry(); set WL/WS (mc_set_uniform), then run().
```
*/

#define QUADRATIC 1
#define GHIGO 1
#define CURVE_LS 1

#include "mc_physics.h"

#include "ghigo/src/myembed.h"
#include "alimare/double_embed-tree.h"
#include "ghigo/src/mydiffusion.h"
#include "run.h"
#include "fractions.h"
#include "curvature.h"
#include "alimare/alex_functions.h"
#include "alimare/simple_discretization.h"
#include "alimare/LS_reinit.h"
#include "ghigo/src/myquadratic.h"
#include "alimare/LS_recons.h"
#include "alimare/LS_curvature.h"

/**
## Model parameters and global fields */

McParams MC;                                              // set by the user, then mc_setup (&MC)
double (* Tprescribed) (double x, double y, double z, double t) = NULL;   // temperature field when !MC.evolveT (z = 0 in 2D)

attribute { scalar Gam; }       // embedded Dirichlet value of a transported field (set by the interface solve)

/* Basilisk's qcc treats `scalar a[N];` as a field declaration, so arrays of scalars are wrapped. */
typedef struct { scalar s; } SF;

SF WL[NF], WS[NF];              // transported fields: modes 0..NC-1 then (if MC.evolveT) the temperature
SF WLg[NF], WSg[NF];            // their embedded Dirichlet values
SF CLg[NC], CSg[NC];            // interface compositions in component space (diagnostics, next guess)
SF GL0[NF], GL1[NF], GS0[NF], GS1[NF];   // affine gradient responses, liquid and solid
scalar * liquid_fields = NULL, * solid_fields = NULL;

scalar dist[], curve[], Vn[], Tint[];
vector Ngeo[], vpc[], vpcf[];
face vector muv[];

/**
Level-set / narrow-band controls (same meaning as in the sandbox code). */

double DT_LS;
int itrecons = 30, itredist = 10;
double tolrecons = 1.e-10, mc_sclean = 1.e-10;
int nb_cell_NB = 8;
double NB_width;
double mc_dt_max = 1e30;        // extra user cap on the time step (explicit curvature stiffness etc.)

int mc_nfail = 0;               // number of interface cells whose local Newton did not converge
int mc_nfields (void) { return NC + (MC.evolveT != 0); }

event defaults (i = 0) {
  int nf = mc_nfields();
  for (int q = 0; q < nf; q++) {
    scalar a = new scalar, b = new scalar, ag = new scalar, bg = new scalar;
    WL[q].s = a; WS[q].s = b; WLg[q].s = ag; WSg[q].s = bg;
    a.Gam = ag; b.Gam = bg;
    a.third = b.third = true;
    liquid_fields = list_append (liquid_fields, a);
    solid_fields  = list_append (solid_fields, b);
    scalar gl0 = new scalar, gl1 = new scalar, gs0 = new scalar, gs1 = new scalar;
    GL0[q].s = gl0; GL1[q].s = gl1; GS0[q].s = gs0; GS1[q].s = gs1;
  }
  for (int i = 0; i < NC; i++) {
    scalar a = new scalar, b = new scalar;
    CLg[i].s = a; CSg[i].s = b;
  }
#if TREE
  for (scalar s in liquid_fields) { s.refine = s.prolongation = refine_embed_linear; s.restriction = restriction_embed_linear; }
  for (scalar s in solid_fields)  { s.refine = s.prolongation = refine_embed_linear2; s.restriction = restriction_embed_linear2; }
#endif
  for (scalar s in liquid_fields) s[embed] = dirichlet (s.Gam[]);
  for (scalar s in solid_fields)  s[embed] = dirichlet (s.Gam[]);
}

/**
## Event order

Basilisk orders events by the position of the first definition of each event *name*. The empty `init` below
is declared before the solver events so that the user's own `init` event (set `dist`, then
`mc_init_geometry()`, then the fields) runs before `mc_stability` at step 0. */

event init (i = 0) { }

/**
## Phase swap

Basilisk's embed solves for the fluid `cs`. To work on the solid we swap `cs <-> 1 - cs` (and `fs`) in place,
exactly as the sandbox code does, then swap back. */

static void flip_phase (void)
{
  foreach()
    cs[] = 1. - cs[];
  foreach_face()
    fs.x[] = 1. - fs.x[];
  boundary ({cs});
  restriction ({cs});
  boundary ({cs, fs});
  restriction ({cs, fs});
}

/**
## Level set -> volume/face fractions (liquid = dist > 0), as in the sandbox `level_set.h` */

static void LS2fractions (scalar dist, scalar cs, face vector fs, double s_clean)
{
  vertex scalar distn[];
  cell2node (dist, distn);
  fractions (distn, cs, fs);
  fractions_cleanup (cs, fs, smin = s_clean, opposite = true);
  boundary ({cs});
  restriction ({cs});
}

/**
## Gradient responses at interface cells

For each transported field the embedded gradient along the phase's outward embedded normal is affine in the
Dirichlet value: g(bc) = g0 + g1*bc. The two evaluations (bc = 0, 1) give the exact local response used by the
interface Newton solve. Must be called with `cs` set to the phase being solved. */

static void gradient_responses (SF * F, SF * G0, SF * G1, scalar * fields, bool store_geometry)
{
  int nf = mc_nfields();
  boundary (fields);
  foreach() {
    if (interfacial (point, cs)) {
      coord n, b;
      embed_geometry (point, &b, &n);
      if (store_geometry) {              // N points from solid into liquid = minus the liquid's embedded normal
        Ngeo.x[] = -n.x; Ngeo.y[] = -n.y;
#if dimension == 3
        Ngeo.z[] = -n.z;
#endif
        if (!MC.evolveT) {
          Tint[] = Tprescribed (x + b.x*Delta, y + b.y*Delta, z + b.z*Delta, t);
        }
      }
      for (int q = 0; q < nf; q++) {
        scalar f = F[q].s, g0 = G0[q].s, g1 = G1[q].s;
        double c0 = 0., c1 = 0.;
        double ga = dirichlet_gradient (point, f, cs, n, b, 0., &c0) + f[]*c0;
        double gb = dirichlet_gradient (point, f, cs, n, b, 1., &c1) + f[]*c1;
        g0[] = ga; g1[] = gb - ga;
      }
    }
    else {
      if (store_geometry) { foreach_dimension() Ngeo.x[] = 0.; Tint[] = nodata; }
      for (int q = 0; q < nf; q++) { scalar g0 = G0[q].s, g1 = G1[q].s; g0[] = 0.; g1[] = 0.; }
    }
  }
}

/**
## Interface solve

Fills, at every interface cell: interface compositions `CLg/CSg`, their modal Dirichlet values `WLg/WSg`,
the interface temperature `Tint` (and `WLg/WSg` of the temperature field), the normal speed `Vn` and the
velocity `vpc = Vn N`. */

void mc_interface_solve (void)
{
  int nf = mc_nfields();
  gradient_responses (WL, GL0, GL1, liquid_fields, true);
  flip_phase();
  gradient_responses (WS, GS0, GS1, solid_fields, false);
  flip_phase();

  int nfail = 0;
  foreach (reduction(+:nfail)) {
    if (interfacial (point, cs)) {
      McLocalIn in; McLocalOut out;
      memset (&in, 0, sizeof in);
      in.Nx = Ngeo.x[]; in.Ny = Ngeo.y[];
#if dimension == 3
      in.Nz = Ngeo.z[];
#endif
      in.kappa = - curve[];            // curvature_LS returns -div N (negative for a convex solid)
      in.Tfrozen = Tint[];
      for (int q = 0; q < nf; q++) {
        scalar a = GL0[q].s, b = GL1[q].s, c = GS0[q].s, d = GS1[q].s;
        in.gl0[q] = a[]; in.gl1[q] = b[];
        in.gs0[q] = c[]; in.gs1[q] = d[];
      }
      /* initial guess: previous interface values if the cell was already an interface cell, else the
         liquid cell-centre composition */
      scalar c0s = CLg[0].s;
      if (c0s[] != nodata && Vn[] != nodata) {
        for (int i = 0; i < NC; i++) { scalar ci = CLg[i].s; in.cl_guess[i] = ci[]; }
        in.V_guess = Vn[]; in.T_guess = MC.evolveT ? Tint[] : in.Tfrozen;
      }
      else {
        for (int i = 0; i < NC; i++) {
          in.cl_guess[i] = 0.;
          for (int k = 0; k < NC; k++) { scalar w = WL[k].s; in.cl_guess[i] += MC.ml.V[i][k]*w[]; }
        }
        in.V_guess = 0.;
        if (MC.evolveT) { scalar TL = WL[NC].s; in.T_guess = TL[]; }
        else in.T_guess = in.Tfrozen;
      }
      mc_local_solve (&MC, &in, &out);
      if (!out.ok) {
        nfail++;
#ifdef MC_DEBUG
        fprintf (stderr, "mc: local solve failed at (%g,%g): res %.3e iters %d  N=(%g,%g) kappa=%g\n"
                 "    guess cl=(%g", x, y, out.res, out.iters, in.Nx, in.Ny, in.kappa, in.cl_guess[0]);
        for (int i = 1; i < NC; i++) fprintf (stderr, ",%g", in.cl_guess[i]);
        fprintf (stderr, ") V=%g  gl0=(%g..) gl1=(%g..) gs0=(%g..)\n    out cl=(%g..) V=%g\n",
                 in.V_guess, in.gl0[0], in.gl1[0], in.gs0[0], out.cl[0], out.V);
#endif
      }
      for (int i = 0; i < NC; i++) { scalar a = CLg[i].s, b = CSg[i].s; a[] = out.cl[i]; b[] = out.cs[i]; }
      for (int k = 0; k < NC; k++) {
        double wl = 0., ws = 0.;
        for (int j = 0; j < NC; j++) { wl += MC.ml.Vinv[k][j]*out.cl[j]; ws += MC.ms.Vinv[k][j]*out.cs[j]; }
        scalar a = WLg[k].s, b = WSg[k].s;
        a[] = wl; b[] = ws;
      }
      if (MC.evolveT) { scalar a = WLg[NC].s, b = WSg[NC].s; a[] = out.T; b[] = out.T; Tint[] = out.T; }
      Vn[] = out.V;
      foreach_dimension() vpc.x[] = out.V*Ngeo.x[];
    }
    else {
      for (int i = 0; i < NC; i++) { scalar a = CLg[i].s, b = CSg[i].s; a[] = nodata; b[] = nodata; }
      Vn[] = nodata;
      foreach_dimension() vpc.x[] = 0.;
    }
  }
  mc_nfail = nfail;
  boundary ((scalar *){vpc});
  boundary (liquid_fields); boundary (solid_fields);
}

/**
## Continuous speed field for the level-set advection (as in the sandbox `LS_speed`) */

static void mc_speed (void)
{
  vector vpcr[];
  foreach()
    foreach_dimension()
      vpcr.x[] = interfacial (point, cs) ? vpc.x[] : 0.;
  boundary ((scalar *){vpcr});
  restriction ((scalar *){vpcr});
#if dimension == 2
  scalar * speed_recons = {vpcr.x, vpcr.y};
#else
  scalar * speed_recons = {vpcr.x, vpcr.y, vpcr.z};
#endif
  double err = 0.;
  recons_speed (dist, DT_LS, speed_recons, tolrecons, &err, itrecons, cs, fs, NB_width);
  foreach()
    foreach_dimension() {
      if (fabs (dist[]) < 0.99*NB_width) vpcf.x[] = vpcr.x[];
      else vpcf.x[] = 0.;
    }
  boundary ((scalar *){vpcf});
  restriction ((scalar *){vpcf});
}

#ifdef MC_DEBUG
/** Report non-finite values in the transported fields (and the fractions) at a named stage. */
static void mc_check_finite (const char * where)
{
  int nf = mc_nfields();
  for (int q = 0; q < nf; q++) {
    for (int ph = 0; ph < 2; ph++) {
      scalar f = (ph ? WS : WL)[q].s;
      int bad = 0; double bx = 0., by = 0.;
      foreach (serial)
        if (!isfinite (f[])) { if (!bad) { bx = x; by = y; } bad++; }
      if (bad) fprintf (stderr, "mc_check_finite[%s]: %s field %d: %d non-finite cells, first at (%g,%g)\n",
                        where, ph ? "solid" : "liquid", q, bad, bx, by);
    }
  }
  int badc = 0;
  foreach (serial) if (!isfinite (cs[]) || !isfinite (dist[])) badc++;
  if (badc) fprintf (stderr, "mc_check_finite[%s]: %d cells with non-finite cs/dist\n", where, badc);
}
#else
#define mc_check_finite(w)
#endif

event mc_stability (i++) {
  mc_check_finite ("stability start");
  mc_interface_solve();
  mc_speed();
  double DT3 = timestep_LS (vpcf, min (DT, mc_dt_max), dist, NB_width);
  tnext = t + DT3;
  dt = DT3;
}

/**
## Advection of the interface, field fill, and implicit transport on the new geometry */

/**
Cells that did not contain the phase at the previous step ("emerged") get a value extrapolated from the
neighbouring cells that *did* (`csm1 > 0`) and still do (`cs > 0`): a distance-weighted least-squares plane
f = a0 + g.r over the 5x5(x5) stencil, evaluated at the cell centre. A light ridge penalty on the slope g keeps the
fit well posed when the valid neighbours are one-sided and narrow (a front sweeping a whole row of cells leaves
only two usable columns, for which the quadratic fit of `embed_extrapolate_ls` is singular and silently returns 0).
`cs` must be the fraction of the phase being filled and `csm1` its previous value. */

static double fill_value (Point point, scalar f)
{
  const int nd = dimension + 1;
  double A[16] = {0.}, rhs[4] = {0.}, wsum = 0.;
  int nvalid = 0;
#if dimension == 2
  for (int i = -2; i <= 2; i++)
    for (int j = -2; j <= 2; j++) {
      if ((i || j) && cs[i,j] > 0. && csm1[i,j] > 0.) {
        double r[3] = {1., i, j}, w = 1./(i*i + j*j), v = f[i,j];
#else
  for (int i = -2; i <= 2; i++)
    for (int j = -2; j <= 2; j++)
      for (int k = -2; k <= 2; k++) {
      if ((i || j || k) && cs[i,j,k] > 0. && csm1[i,j,k] > 0.) {
        double r[4] = {1., i, j, k}, w = 1./(i*i + j*j + k*k), v = f[i,j,k];
#endif
        for (int a_ = 0; a_ < nd; a_++) {
          rhs[a_] += w*r[a_]*v;
          for (int b_ = 0; b_ < nd; b_++) A[a_*nd + b_] += w*r[a_]*r[b_];
        }
        wsum += w;
        nvalid++;
      }
    }
  if (!nvalid)
    return f[];                                   // nothing to extrapolate from: keep the current value
  for (int a_ = 1; a_ < nd; a_++)
    A[a_*nd + a_] += 0.05*wsum;                   // ridge on the slope components only
  double sol[4];
  if (mc_gauss (nd, A, rhs, sol))
    return f[];
  return sol[0];
}

static void fill_emerged (scalar * fields)
{
  foreach() {
    if (cs[] > 0. && csm1[] <= 0.) {
      /* compute every field before storing, so neighbours are never other emerged cells' fresh values */
      for (scalar f in fields)
        f[] = fill_value (point, f);
    }
  }
  boundary (fields);
  restriction (fields);
}

static void diffuse_phase (SF * F, const Modes * md, double alpha_th)
{
  int nf = mc_nfields();
  for (int q = 0; q < nf; q++) {
    double D = q < NC ? md->lam[q] : alpha_th;
    foreach_face()
      muv.x[] = D*fs.x[];
    boundary ((scalar *){muv});
    scalar f = F[q].s;
    mgstats s = diffusion (f, dt, muv);
    (void) s;
  }
}

event mc_advance (i++) {
  /* previous geometry */
  foreach()
    csm1[] = cs[];
  boundary ({csm1});
  restriction ({csm1});

  /* level set: advect, reinitialise, fractions, curvature */
#ifndef MC_FREEZE_GEOMETRY                       // debug switch: keep the interface fixed
  RK3_WENO5 (dist, vpcf, dt, NB_width);
#endif
  boundary ({dist});
  restriction ({dist});
  LS_reinit (dist = dist, dt = 0.2*L0/(1 << grid->maxdepth), it_max = itredist);
  foreach()
    dist[] = clamp (dist[], -1.05*NB_width, 1.05*NB_width);
  boundary ({dist});
  restriction ({dist});
  LS2fractions (dist, cs, fs, mc_sclean);
  curvature_LS (dist, curve);
  boundary ({curve});
  restriction ({curve});

  /* cells uncovered by the motion: extrapolate each phase's fields into them */
  fill_emerged (liquid_fields);
  flip_phase();                       // cs is now the solid fraction ...
  foreach()
    csm1[] = 1. - csm1[];             // ... and csm1 the previous solid fraction
  boundary ({csm1});
  fill_emerged (solid_fields);
  foreach()
    csm1[] = 1. - csm1[];
  boundary ({csm1});
  flip_phase();

  mc_check_finite ("after fill");
  /* Dirichlet data on the new geometry, then implicit diffusion of every mode in both phases */
  mc_interface_solve();
  mc_check_finite ("after interface solve");
  diffuse_phase (WL, &MC.ml, MC.alphaL);
  flip_phase();
  diffuse_phase (WS, &MC.ms, MC.alphaS);
  flip_phase();
}

/**
## Solid fractions for the solid-phase refine/restriction operators (needed on adaptive grids) */

void mc_update_solid_fractions (void)
{
  foreach_cell()
    cs2[] = 1. - cs[];
  foreach_face()
    fs2.x[] = 1. - fs.x[];
  boundary ({cs, cs2});
  fractions_cleanup (cs, fs, smin = 1.e-10);
  fractions_cleanup (cs2, fs2, smin = 1.e-10);
  restriction ({cs, cs2, fs, fs2});
}

/**
## Initialisation helpers

`mc_init_geometry()` must be called in the user's `init` event after `dist[]` is set. */

void mc_init_geometry (void)
{
  foreach()
    dist[] = clamp (dist[], -1.5*NB_width, 1.5*NB_width);
  boundary ({dist});
  restriction ({dist});
  LS2fractions (dist, cs, fs, 1e-10);
  curvature_LS (dist, curve);
  boundary ({curve});
  foreach()
    csm1[] = cs[];
  boundary ({csm1});
  restriction ({csm1});
  mc_update_solid_fractions();
  scalar c0 = CLg[0].s;
  foreach() { Vn[] = nodata; c0[] = nodata; }
  boundary ({Vn, c0});
}

/** Set uniform compositions (component space) in a phase: w = Vinv c. `c` has NC entries. */
void mc_set_uniform (const double * c, bool liquid)
{
  const Modes * md = liquid ? &MC.ml : &MC.ms;
  SF * F = liquid ? WL : WS;
  foreach()
    for (int k = 0; k < NC; k++) {
      double w = 0.;
      for (int j = 0; j < NC; j++) w += md->Vinv[k][j]*c[j];
      scalar f = F[k].s;
      f[] = w;
    }
  boundary (liquid ? liquid_fields : solid_fields);
  restriction (liquid ? liquid_fields : solid_fields);
}

/** Set the modal fields of the current cell from component-space values `c` (use inside foreach). */
void mc_set_cell (Point point, const double * c, bool liquid)
{
  const Modes * md = liquid ? &MC.ml : &MC.ms;
  SF * F = liquid ? WL : WS;
  for (int k = 0; k < NC; k++) {
    double w = 0.;
    for (int j = 0; j < NC; j++) w += md->Vinv[k][j]*c[j];
    scalar f = F[k].s;
    f[] = w;
  }
}

/** Component-space value of component `i` in a phase at the current cell (use inside foreach). */
#define mc_comp(i, liquid) ({ double _c = 0.; \
  for (int _k = 0; _k < NC; _k++) { scalar _w = (liquid ? WL : WS)[_k].s; \
    _c += (liquid ? MC.ml : MC.ms).V[i][_k]*_w[]; } _c; })

/** Total amount of each component (volume integral of cs*c_l + (1-cs)*c_s). Note: with embed, `dv()`
includes the fluid fraction (the metric cm is aliased to cs), so the full cell volume is used here. */
void mc_inventory (double * out)
{
  for (int i = 0; i < NC; i++) {
    double tot = 0.;
    foreach (reduction(+:tot)) {
      double cl = 0., cs_ = 0.;
      for (int k = 0; k < NC; k++) {
        scalar wl = WL[k].s, ws = WS[k].s;
        cl += MC.ml.V[i][k]*wl[]; cs_ += MC.ms.V[i][k]*ws[];
      }
      tot += (cs[]*cl + (1. - cs[])*cs_)*pow (Delta, dimension);   // dv() already contains cs (cm = cs): use the full cell volume
    }
    out[i] = tot;
  }
}

/**
## Adaptation

Wavelet adaptation on the geometry (`cs`), the interface speed, and a phase-blended copy of *every*
transported field, visu_q = cs w_l,q + (1 - cs) w_s,q (the same blend the sandbox uses for its single scalar).
`mc_adapt_tol_field[q]` are absolute tolerances per field (modes: composition units; temperature: Kelvin;
0 means the default 1e-3). */

int MAXLEVEL = 7, MINLEVEL = 4;
double mc_adapt_tol_cs = 1.e-3, mc_adapt_tol_speed = 5.e-3;
double mc_adapt_tol_field[NF];                   // 0 (default) means 1e-3

event mc_adapt (i++, last) {
  mc_update_solid_fractions();
#if TREE
  int nf = mc_nfields();
  scalar * list = list_concat ({cs}, (scalar *){vpcf});
  double tol[3 + NF];
  tol[0] = mc_adapt_tol_cs;
  for (int k = 0; k < dimension; k++) tol[1 + k] = mc_adapt_tol_speed;
  int ntol = 1 + dimension;
  for (int q = 0; q < nf; q++) {
    scalar v = new scalar;
    scalar wl = WL[q].s, ws = WS[q].s;
    foreach()
      v[] = cs[]*wl[] + (1. - cs[])*ws[];
    boundary ({v});
    restriction ({v});
    list = list_append (list, v);
    tol[ntol++] = mc_adapt_tol_field[q] > 0. ? mc_adapt_tol_field[q] : 1.e-3;
  }
  adapt_wavelet (list, tol, MAXLEVEL, MINLEVEL);
  /* the temporary blended fields are the last nf entries */
  scalar * tmp = NULL;
  int idx = 0;
  for (scalar s in list) { if (idx++ >= 1 + dimension) tmp = list_append (tmp, s); }
  delete (tmp);
  free (tmp);
  free (list);
  curvature_LS (dist, curve);
  boundary ({curve});
#endif
}

/**
## Output

One line per interface cell: x y [z] V T then (c_l, c_s) for each component. */

void mc_output_interface (FILE * fp)
{
  foreach (serial) {
    if (interfacial (point, cs)) {
      coord n, b;
      embed_geometry (point, &b, &n);
#if dimension == 2
      fprintf (fp, "%g %g", x + b.x*Delta, y + b.y*Delta);
#else
      fprintf (fp, "%g %g %g", x + b.x*Delta, y + b.y*Delta, z + b.z*Delta);
#endif
      fprintf (fp, " %g %g", Vn[], Tint[]);
      for (int i = 0; i < NC; i++) {
        scalar a = CLg[i].s, c = CSg[i].s;
        fprintf (fp, " %g %g", a[], c[]);
      }
      fputc ('\n', fp);
    }
  }
  fflush (fp);
}
