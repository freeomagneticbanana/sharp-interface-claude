/**
# Multicomponent sharp-interface physics (plain C, no Basilisk dependency)

Everything that does not touch the grid lives here so that it can be compiled and unit-tested on its own
(and cross-checked against the Python reference in `sharp_interface/`):

- quadratic phase free energies g(c,T) for NC independent components,
- tie lines (equal diffusion potentials) and the grand-potential driving force,
- surface-energy anisotropy gamma(theta,T), its stiffness, and the kinetic coefficient beta(theta),
- diagonalisation of the interdiffusion matrix into decoupled diffusion modes,
- the *local interface problem*: given one-sided gradient responses, solve for the interface
  composition, normal speed (and temperature when it evolves) at a single interface cell.

Conventions (identical to the Python code):

- N is the unit normal pointing from the solid into the liquid; V > 0 is growth of the solid along N.
- [a] = a_l - a_s.  kappa = div N  (> 0 for a convex solid).
- Basilisk's embedded normal points *out of the phase being solved*, so the gradient returned for the
  liquid is d/dn_e = -d/dN, and for the solid d/dn_e = +d/dN.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NC
#define NC 1                 // number of independent components (N - 1)
#endif
#define NF (NC + 1)          // transported fields per phase: NC solute modes + temperature
#define NU (NC + 2)          // maximum number of interface unknowns: NC c_l, V, T

/* ------------------------------------------------------------------------------------------------
   Dense linear algebra for tiny systems (row-major, n <= NU)
   ---------------------------------------------------------------------------------------------- */

/** Solve A x = b by Gaussian elimination with partial pivoting. A and b are destroyed.
Returns 0 on success, 1 if singular. */
static int mc_gauss (int n, double * A, double * b, double * x)
{
  for (int c = 0; c < n; c++) {
    int p = c;
    for (int r = c + 1; r < n; r++)
      if (fabs (A[r*n + c]) > fabs (A[p*n + c])) p = r;
    if (fabs (A[p*n + c]) < 1e-300) return 1;
    if (p != c) {
      for (int k = 0; k < n; k++) { double t = A[c*n + k]; A[c*n + k] = A[p*n + k]; A[p*n + k] = t; }
      double t = b[c]; b[c] = b[p]; b[p] = t;
    }
    for (int r = c + 1; r < n; r++) {
      double f = A[r*n + c]/A[c*n + c];
      for (int k = c; k < n; k++) A[r*n + k] -= f*A[c*n + k];
      b[r] -= f*b[c];
    }
  }
  for (int r = n - 1; r >= 0; r--) {
    double s = b[r];
    for (int k = r + 1; k < n; k++) s -= A[r*n + k]*x[k];
    x[r] = s/A[r*n + r];
  }
  return 0;
}

/** Inverse of an n x n matrix (n <= NU) by solving against the identity. Returns 0 on success. */
static int mc_inverse (int n, const double * A, double * Ainv)
{
  for (int col = 0; col < n; col++) {
    double M[NU*NU], b[NU], x[NU];
    memcpy (M, A, n*n*sizeof (double));
    for (int r = 0; r < n; r++) b[r] = (r == col);
    if (mc_gauss (n, M, b, x)) return 1;
    for (int r = 0; r < n; r++) Ainv[r*n + col] = x[r];
  }
  return 0;
}

/** Cyclic Jacobi eigen-decomposition of a symmetric matrix: A = Q diag(lam) Q^T. A is destroyed. */
static void mc_jacobi (int n, double * A, double * Q, double * lam)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) Q[i*n + j] = (i == j);
  for (int sweep = 0; sweep < 100; sweep++) {
    double off = 0.;
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++) off += A[i*n + j]*A[i*n + j];
    if (off < 1e-30) break;
    for (int p = 0; p < n; p++)
      for (int q = p + 1; q < n; q++) {
        if (fabs (A[p*n + q]) < 1e-300) continue;
        double th = 0.5*atan2 (2.*A[p*n + q], A[q*n + q] - A[p*n + p]);
        double c = cos (th), s = sin (th);
        for (int k = 0; k < n; k++) {
          double akp = A[k*n + p], akq = A[k*n + q];
          A[k*n + p] = c*akp - s*akq; A[k*n + q] = s*akp + c*akq;
        }
        for (int k = 0; k < n; k++) {
          double apk = A[p*n + k], aqk = A[q*n + k];
          A[p*n + k] = c*apk - s*aqk; A[q*n + k] = s*apk + c*aqk;
        }
        for (int k = 0; k < n; k++) {
          double qkp = Q[k*n + p], qkq = Q[k*n + q];
          Q[k*n + p] = c*qkp - s*qkq; Q[k*n + q] = s*qkp + c*qkq;
        }
      }
  }
  for (int i = 0; i < n; i++) lam[i] = A[i*n + i];
}

/* ------------------------------------------------------------------------------------------------
   Quadratic free energies
   g(c,T) = g0(T) + a(T).(c - c0) + 1/2 (c - c0)^T H (c - c0),
   g0(T) = g0 + g0T (T - Tref),  a(T) = a + aT (T - Tref);  H and c0 are constant.
   ---------------------------------------------------------------------------------------------- */

typedef struct {
  double g0, g0T;
  double a[NC], aT[NC];
  double H[NC][NC], Hinv[NC][NC];
  double c0[NC];
  double Tref;
} QPhase;

/** Must be called after filling H: computes Hinv. Returns 0 on success. */
static int qp_finalize (QPhase * p)
{
  return mc_inverse (NC, &p->H[0][0], &p->Hinv[0][0]);
}

/** True if H is symmetric positive definite (necessary for a stable single-phase free energy). */
static int qp_is_stable (const QPhase * p)
{
  double A[NC*NC], Q[NC*NC], lam[NC];
  for (int i = 0; i < NC; i++)
    for (int j = 0; j < NC; j++) A[i*NC + j] = 0.5*(p->H[i][j] + p->H[j][i]);
  mc_jacobi (NC, A, Q, lam);
  for (int i = 0; i < NC; i++) if (lam[i] <= 0.) return 0;
  return 1;
}

static inline double qp_g0 (const QPhase * p, double T) { return p->g0 + p->g0T*(T - p->Tref); }
static inline double qp_a (const QPhase * p, int i, double T) { return p->a[i] + p->aT[i]*(T - p->Tref); }

/** Diffusion potentials mu_i = dg/dc_i. */
static void qp_mu (const QPhase * p, const double c[NC], double T, double mu[NC])
{
  for (int i = 0; i < NC; i++) {
    mu[i] = qp_a (p, i, T);
    for (int j = 0; j < NC; j++) mu[i] += p->H[i][j]*(c[j] - p->c0[j]);
  }
}

/** Composition for given diffusion potentials: c = c0 + H^{-1} (mu - a). */
static void qp_c_of_mu (const QPhase * p, const double mu[NC], double T, double c[NC])
{
  for (int i = 0; i < NC; i++) {
    c[i] = p->c0[i];
    for (int j = 0; j < NC; j++) c[i] += p->Hinv[i][j]*(mu[j] - qp_a (p, j, T));
  }
}

static double qp_g (const QPhase * p, const double c[NC], double T)
{
  double g = qp_g0 (p, T);
  for (int i = 0; i < NC; i++) {
    double di = c[i] - p->c0[i];
    g += qp_a (p, i, T)*di;
    for (int j = 0; j < NC; j++) g += 0.5*di*p->H[i][j]*(c[j] - p->c0[j]);
  }
  return g;
}

/** Grand potential omega(mu,T) = g - mu.c evaluated at c(mu,T). */
static double qp_omega (const QPhase * p, const double mu[NC], double T)
{
  double c[NC];
  qp_c_of_mu (p, mu, T, c);
  double w = qp_g (p, c, T);
  for (int i = 0; i < NC; i++) w -= mu[i]*c[i];
  return w;
}

/** Local-equilibrium tie line: given the liquid composition cl at T, returns the solid composition in
common-chemical-potential equilibrium and the driving force dOmega = omega_l - omega_s (> 0: the
solid is favoured). */
static void mc_tie (const QPhase * S, const QPhase * L, const double cl[NC], double T,
                    double cs[NC], double * dOmega)
{
  double mu[NC];
  qp_mu (L, cl, T, mu);
  qp_c_of_mu (S, mu, T, cs);
  *dOmega = qp_omega (L, mu, T) - qp_omega (S, mu, T);
}

/* ------------------------------------------------------------------------------------------------
   Surface energy and interface kinetics (2D anisotropy; 3D is isotropic only)
   gamma(theta,T) = gamma0(T) (1 + eps cos(nfold (theta - theta0))),  gamma0(T) = gamma_m + gamma_T (T - T_ref)
   stiffness = gamma + gamma'' = gamma0 (1 + eps (1 - nfold^2) cos(nfold (theta - theta0)))
   The dependence of gamma on position through T adds a term (dgamma/dT) d_n T to the capillary force;
   it is not included (model simplification, same as the Python reference).
   ---------------------------------------------------------------------------------------------- */

typedef struct { double gamma_m, eps, theta0, gamma_T, T_ref; int nfold; } SurfEnergy;
typedef struct { double beta0, eps, theta0; int nfold; } Kinetics;

static inline double se_gamma0 (const SurfEnergy * s, double T) { return s->gamma_m + s->gamma_T*(T - s->T_ref); }

static inline double se_gamma (const SurfEnergy * s, double theta, double T)
{
  return se_gamma0 (s, T)*(1. + s->eps*cos (s->nfold*(theta - s->theta0)));
}

static inline double se_stiffness (const SurfEnergy * s, double theta, double T)
{
  return se_gamma0 (s, T)*(1. + s->eps*(1. - (double) (s->nfold*s->nfold))*cos (s->nfold*(theta - s->theta0)));
}

static inline double kin_beta (const Kinetics * k, double theta)
{
  return k->beta0*(1. + k->eps*cos (k->nfold*(theta - k->theta0)));
}

/* ------------------------------------------------------------------------------------------------
   Decoupling the interdiffusion matrix
   For constant D, c_i diffuses as  dc/dt = div (D grad c).  With D = Vm diag(lam) Vm^{-1} the modal
   fields w = Vm^{-1} c obey independent equations dw_k/dt = lam_k lap w_k.
   ---------------------------------------------------------------------------------------------- */

typedef struct { double V[NC][NC], Vinv[NC][NC], lam[NC]; } Modes;

/** Modes of a Fickian matrix D. Supported: NC = 1, NC = 2 (real eigenvalues), or symmetric D of any size.
Returns 0 on success, 1 on unsupported/complex/defective input. */
static int modes_from_D (const double D[NC][NC], Modes * m)
{
#if NC == 1
  m->V[0][0] = m->Vinv[0][0] = 1.; m->lam[0] = D[0][0];
  return D[0][0] > 0. ? 0 : 1;
#elif NC == 2
  double a = D[0][0], b = D[0][1], c = D[1][0], d = D[1][1];
  double tr2 = 0.5*(a + d), disc = tr2*tr2 - (a*d - b*c);
  if (disc < -1e-14*tr2*tr2) return 1;                 // complex eigenvalues
  double sq = sqrt (disc > 0. ? disc : 0.);
  m->lam[0] = tr2 + sq; m->lam[1] = tr2 - sq;
  for (int k = 0; k < 2; k++) {
    double l = m->lam[k], v0, v1;
    if (fabs (b) > 1e-14*fabs (tr2)) { v0 = b; v1 = l - a; }
    else if (fabs (c) > 1e-14*fabs (tr2)) { v0 = l - d; v1 = c; }
    else { v0 = (fabs (l - a) <= fabs (l - d)); v1 = 1. - v0; }   // diagonal D
    double nrm = sqrt (v0*v0 + v1*v1);
    m->V[0][k] = v0/nrm; m->V[1][k] = v1/nrm;
  }
  if (mc_inverse (2, &m->V[0][0], &m->Vinv[0][0])) return 1;   // repeated eigenvalue of a defective matrix
  return (m->lam[0] > 0. && m->lam[1] > 0.) ? 0 : 1;
#else
  double A[NC*NC], Q[NC*NC];
  for (int i = 0; i < NC; i++)
    for (int j = 0; j < NC; j++) {
      if (fabs (D[i][j] - D[j][i]) > 1e-12*(fabs (D[i][j]) + fabs (D[j][i]) + 1e-300)) return 1;
      A[i*NC + j] = D[i][j];
    }
  mc_jacobi (NC, A, Q, m->lam);
  for (int i = 0; i < NC; i++)
    for (int j = 0; j < NC; j++) { m->V[i][j] = Q[i*NC + j]; m->Vinv[j][i] = Q[i*NC + j]; }
  for (int i = 0; i < NC; i++) if (m->lam[i] <= 0.) return 1;
  return 0;
#endif
}

/** Modes of the free-energy-driven form D = L H (L: Onsager mobility, SPD; H: Hessian of the phase free
energy, SPD).  D is similar to the symmetric matrix R L R^T with H = R^T R (Cholesky), so the
decomposition is valid for any NC.  Also returns D. Returns 0 on success. */
static int modes_from_onsager (const double L[NC][NC], const double H[NC][NC], double D[NC][NC], Modes * m)
{
  double R[NC][NC] = {{0}};
  for (int i = 0; i < NC; i++)
    for (int j = 0; j <= i; j++) {                        // H = Rc Rc^T with lower-triangular Rc
      double s = H[i][j];
      for (int k = 0; k < j; k++) s -= R[i][k]*R[j][k];
      if (i == j) { if (s <= 0.) return 1; R[i][i] = sqrt (s); }
      else R[i][j] = s/R[j][j];
    }
  /* H = Rc Rc^T, so with U = Rc^T: H = U^T U, D = L H ~ U L U^T (symmetric) */
  double S[NC*NC], Q[NC*NC], lam[NC];
  for (int i = 0; i < NC; i++)
    for (int j = 0; j < NC; j++) {
      double s = 0.;
      for (int p = 0; p < NC; p++)
        for (int q = 0; q < NC; q++) s += R[p][i]*L[p][q]*R[q][j];   // (Rc^T L Rc)_ij
      S[i*NC + j] = s;
    }
  /* D = L Rc Rc^T = Rc^{-T} (Rc^T L Rc) Rc^T  =>  D Rc^{-T} Q = Rc^{-T} S Q = Rc^{-T} Q lam */
  mc_jacobi (NC, S, Q, lam);
  double Rinv[NC*NC], Rm[NC*NC];
  for (int i = 0; i < NC; i++)
    for (int j = 0; j < NC; j++) Rm[i*NC + j] = R[j][i];            // Rc^T
  if (mc_inverse (NC, Rm, Rinv)) return 1;
  for (int i = 0; i < NC; i++)
    for (int k = 0; k < NC; k++) {
      double s = 0.;
      for (int p = 0; p < NC; p++) s += Rinv[i*NC + p]*Q[p*NC + k];
      m->V[i][k] = s;
    }
  for (int k = 0; k < NC; k++) m->lam[k] = lam[k];
  double Vf[NC*NC];
  for (int i = 0; i < NC; i++) for (int j = 0; j < NC; j++) Vf[i*NC + j] = m->V[i][j];
  if (mc_inverse (NC, Vf, &m->Vinv[0][0])) return 1;
  for (int i = 0; i < NC; i++)
    for (int j = 0; j < NC; j++) {
      D[i][j] = 0.;
      for (int p = 0; p < NC; p++) D[i][j] += L[i][p]*H[p][j];
    }
  for (int k = 0; k < NC; k++) if (m->lam[k] <= 0.) return 1;
  return 0;
}

/* ------------------------------------------------------------------------------------------------
   Parameters of the whole model
   ---------------------------------------------------------------------------------------------- */

typedef struct {
  QPhase S, L;                 // solid / liquid free energies
  double Ds[NC][NC], Dl[NC][NC];   // interdiffusion matrices (Fickian, constant)
  Modes ms, ml;                // their diffusion modes (filled by mc_setup)
  SurfEnergy gam;
  Kinetics kin;
  int evolveT;                 // 0: prescribed ("frozen") temperature, 1: heat equations + Stefan balance
  double alphaS, alphaL;       // thermal diffusivities  k / (rho c)   (evolving T)
  double kS, kL;               // thermal conductivities               (evolving T)
  double rhoLat;               // volumetric latent heat rho L          (evolving T)
  double Vscale, Escale, Tscale;   // reference speed, energy density, temperature (residual/unknown scaling)
  double newton_tol; int newton_max;
} McParams;

/** Fill derived quantities. Returns 0 on success (prints the reason otherwise). */
static int mc_setup (McParams * P)
{
  if (qp_finalize (&P->S) || qp_finalize (&P->L)) { fprintf (stderr, "mc_setup: singular Hessian\n"); return 1; }
  if (!qp_is_stable (&P->S) || !qp_is_stable (&P->L)) fprintf (stderr, "mc_setup: warning: free energy Hessian is not positive definite\n");
  if (modes_from_D (P->Ds, &P->ms) || modes_from_D (P->Dl, &P->ml)) {
    fprintf (stderr, "mc_setup: cannot diagonalise D (need real positive eigenvalues; for NC >= 3 use modes_from_onsager)\n");
    return 1;
  }
  if (P->newton_tol <= 0.) P->newton_tol = 1e-10;
  if (P->newton_max <= 0) P->newton_max = 40;
  if (P->Vscale <= 0.) P->Vscale = 1.;
  if (P->Escale <= 0.) P->Escale = 1.;
  if (P->Tscale <= 0.) P->Tscale = 1.;
#if dimension == 3
  if (P->gam.eps != 0. || P->kin.eps != 0.) { fprintf (stderr, "mc_setup: anisotropy is implemented in 2D only\n"); return 1; }
#endif
  return 0;
}

/* ------------------------------------------------------------------------------------------------
   Local interface problem at one interface cell
   Unknowns u = (c_l[0..NC-1], V, [T]); equations
     mass (NC):  V (c_l - c_s) - (D_l g_l + D_s g_s) = 0      with c_s from the tie line,
     GT   (1) :  (omega_l - omega_s) - stiffness(theta,T) kappa - beta(theta) V = 0,
     Stefan (1, evolving T): rhoL V - (kS gT_s + kL gT_l) = 0.
   g_l (g_s) are the component gradients along the embedded normal of the liquid (solid), reconstructed
   from the *affine* response of the one-sided embedded gradient to the Dirichlet value of each mode:
     d/dn_e w_k = g0_k + g1_k * w_k^Gamma,   w^Gamma = Vinv c^Gamma.
   ---------------------------------------------------------------------------------------------- */

typedef struct {
  double Nx, Ny, Nz;           // unit normal, solid -> liquid
  double kappa;                // div N (> 0 for convex solid)
  double Tfrozen;              // interface temperature when !evolveT
  double gl0[NF], gl1[NF];     // liquid modes (0..NC-1) and temperature (NC): affine gradient along n_e = -N
  double gs0[NF], gs1[NF];     // solid  modes and temperature: affine gradient along n_e = +N
  double cl_guess[NC], V_guess, T_guess;
} McLocalIn;

typedef struct {
  double cl[NC], cs[NC], V, T, dOmega;
  double res;                  // scaled residual norm at exit
  int iters, ok;
} McLocalOut;

static inline int mc_nunknowns (const McParams * P) { return NC + 1 + (P->evolveT != 0); }

static void mc_residual (const McParams * P, const McLocalIn * in, const double * u, double * R)
{
  const double * cl = u;
  double V = u[NC];
  double T = P->evolveT ? u[NC + 1] : in->Tfrozen;
  double cs[NC], dO;
  mc_tie (&P->S, &P->L, cl, T, cs, &dO);
  /* component gradients along n_e for both phases */
  double wl[NC], ws[NC], gwl[NC], gws[NC], gcl[NC], gcs[NC];
  for (int k = 0; k < NC; k++) {
    wl[k] = ws[k] = 0.;
    for (int j = 0; j < NC; j++) { wl[k] += P->ml.Vinv[k][j]*cl[j]; ws[k] += P->ms.Vinv[k][j]*cs[j]; }
    gwl[k] = in->gl0[k] + in->gl1[k]*wl[k];
    gws[k] = in->gs0[k] + in->gs1[k]*ws[k];
  }
  for (int j = 0; j < NC; j++) {
    gcl[j] = gcs[j] = 0.;
    for (int k = 0; k < NC; k++) { gcl[j] += P->ml.V[j][k]*gwl[k]; gcs[j] += P->ms.V[j][k]*gws[k]; }
  }
  for (int i = 0; i < NC; i++) {
    double flux = 0.;
    for (int j = 0; j < NC; j++) flux += P->Dl[i][j]*gcl[j] + P->Ds[i][j]*gcs[j];
    R[i] = V*(cl[i] - cs[i]) - flux;
  }
  double theta = atan2 (in->Ny, in->Nx);
#if dimension == 3
  double stiff = se_gamma0 (&P->gam, T), beta = P->kin.beta0;
#else
  double stiff = se_stiffness (&P->gam, theta, T), beta = kin_beta (&P->kin, theta);
#endif
  R[NC] = dO - stiff*in->kappa - beta*V;
  if (P->evolveT) {
    double gTl = in->gl0[NC] + in->gl1[NC]*T, gTs = in->gs0[NC] + in->gs1[NC]*T;
    R[NC + 1] = P->rhoLat*V - (P->kS*gTs + P->kL*gTl);
  }
}

/** Scaled infinity norm of the residual. */
static double mc_resnorm (const McParams * P, const double * R)
{
  double m = 0.;
  for (int i = 0; i < NC; i++) m = fmax (m, fabs (R[i])/P->Vscale);
  m = fmax (m, fabs (R[NC])/P->Escale);
  if (P->evolveT) m = fmax (m, fabs (R[NC + 1])/(P->rhoLat*P->Vscale + 1e-300));
  return m;
}

/** Damped Newton with a finite-difference Jacobian (the system has at most NC + 2 unknowns). */
static void mc_local_solve (const McParams * P, const McLocalIn * in, McLocalOut * out)
{
  int n = mc_nunknowns (P);
  double u[NU], R[NU], su[NU];
  for (int i = 0; i < NC; i++) { u[i] = in->cl_guess[i]; su[i] = 1.; }
  u[NC] = in->V_guess; su[NC] = P->Vscale;
  if (P->evolveT) { u[NC + 1] = in->T_guess; su[NC + 1] = P->Tscale; }
  mc_residual (P, in, u, R);
  double nr = mc_resnorm (P, R);
  int it = 0;
  for (; it < P->newton_max && nr > P->newton_tol; it++) {
    double J[NU*NU], b[NU], du[NU];
    for (int j = 0; j < n; j++) {
      double h = 1e-7*(fabs (u[j]) + 1e-3*su[j]), up[NU], Rp[NU];
      memcpy (up, u, n*sizeof (double));
      up[j] += h;
      mc_residual (P, in, up, Rp);
      for (int i = 0; i < n; i++) J[i*n + j] = (Rp[i] - R[i])/h;
    }
    for (int i = 0; i < n; i++) b[i] = -R[i];
    if (mc_gauss (n, J, b, du)) break;                       // singular Jacobian
    double lam = 1., un[NU], Rn[NU], nn = nr;
    for (int ls = 0; ls < 12; ls++, lam *= 0.5) {
      for (int i = 0; i < n; i++) un[i] = u[i] + lam*du[i];
      mc_residual (P, in, un, Rn);
      nn = mc_resnorm (P, Rn);
      if (nn < (1. - 1e-4*lam)*nr || nn < P->newton_tol) break;
    }
    memcpy (u, un, n*sizeof (double)); memcpy (R, Rn, n*sizeof (double));
    if (!(nn < nr) && nn > P->newton_tol) { nr = nn; break; }  // line search stalled
    nr = nn;
  }
  for (int i = 0; i < NC; i++) out->cl[i] = u[i];
  out->V = u[NC];
  out->T = P->evolveT ? u[NC + 1] : in->Tfrozen;
  mc_tie (&P->S, &P->L, out->cl, out->T, out->cs, &out->dOmega);
  out->res = nr; out->iters = it; out->ok = (nr <= P->newton_tol*10.) && isfinite (nr);
}
