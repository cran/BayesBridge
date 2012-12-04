
// This code has been taken from the copula package and slightly modified for
// our use.

#include <cmath>
#include <stdio.h>

#include "RNG.h"

#ifndef M_SQRT_PI
#define M_SQRT_PI sqrt(M_PI)
#endif

double sinc_MM(double x) {
  double ax = fabs(x);
  if(ax < 0.006) {
    if(x == 0.) return 1;
    double x2 = x*x;
    if(ax < 2e-4)
      return 1. - x2/6.;
    else return 1. - x2/6.*(1 - x2/20.);
  }
  /**< else */
  return sin(x)/x;
}

/**
 * Evaluation of Zolotarev's function, see Devroye (2009), to the power 1-alpha.
 * The 3-arg. version allows more precision for  alpha ~=~ 1
 *
 * @param x argument
 * @param alpha parameter in (0,1]
 * @return sin(alpha*x)^alpha * sin((1-alpha)*x)^(1-alpha) / sin(x)
 * @author Martin Maechler (2010-04-28)
 */
#define _A_3(_x, _alpha_, _I_alpha)                             \
  pow(_I_alpha* sinc_MM(_I_alpha*_x), _I_alpha) *               \
  pow(_alpha_ * sinc_MM(_alpha_ *_x), _alpha_ ) / sinc_MM(_x)

double A_(double x, double alpha) {
  double Ialpha = 1.-alpha;
  return _A_3(x, alpha, Ialpha);
}

// /**< to be called from R---see experiments in ../tests/retstable-ex.R */
// SEXP A__c(SEXP x_, SEXP alpha, SEXP I_alpha) {
//   int n = LENGTH(PROTECT(x_ = coerceVector(x_, REALSXP)));
//   double alp = asReal(alpha), I_alp = asReal(I_alpha);
//   if(fabs(alp + I_alp - 1.) > 1e-12)
//     error("'I_alpha' must be == 1 - alpha more accurately");
//   SEXP r_ = allocVector(REALSXP, n); /**< the result */
//   double *x = REAL(x_), *r = REAL(r_);

//   for(int i=0; i < n; i++)
//     r[i] = _A_3(x[i], alp, I_alp);

//   UNPROTECT(1);
//   return r_;
// } */

/**
 * Evaluation of B(x)/B(0), see Devroye (2009).
 *
 * @param x argument
 * @param alpha parameter in (0,1]
 * @return sinc(x) / (sinc(alpha*x)^alpha * sinc((1-alpha)*x)^(1-alpha))
 * @author Martin Maechler (2010-04-28)
 */
double BdB0(double x,double alpha) {
  double Ialpha = 1.-alpha;
  double den = pow(sinc_MM(alpha*x),alpha) * pow(sinc_MM(Ialpha*x),Ialpha);
  return sinc_MM(x) / den;
}

/**
 * Sample St ~ \tilde{S}(alpha, 1, (cos(alpha*pi/2)*V_0)^{1/alpha},
 *                       V_0*I_{alpha = 1}, h*I_{alpha != 1}; 1)
 * with Laplace-Stieltjes transform exp(-V_0((h+t)^alpha-h^alpha)),
 * see Nolan's book for the parametrization, via double rejection,
 * see Devroye (2009).
 *
 * @param St vector of random variates (result)
 * @param V0 vector of random variates V0
 * @param h parameter in [0,infinity)
 * @param alpha parameter in (0,1]
 * @param n length of St
 * @return none
 * @author Marius Hofert, Martin Maechler
 */
double retstable_LD(double h, double alpha, RNG& r)
{
  double V0 = 1.0; // Just set to 1 since that's what we want anyway.
  // h is the tilting parameter
  // alpha is the exponent
  // V0 is theta by Devroye's notation
  /**
   * alpha == 1 => St corresponds to a point mass at V0 with Laplace-Stieltjes
   * transform exp(-V0*t)
   */
  if(alpha == 1.){
    return 1.0;
    // for(int i = 0; i < n; i++) {
    //   St[i] = V0[i];
    // }
    // return;
  }

  if (h < 0 || alpha < 0 || alpha > 1 || V0 < 0) {
    Rprintf("Problem with parameter.\n");
    Rprintf("V0: %g; h: %g; alpha: %g\n", V0, h, alpha);
  }

  // report input
  // Rprintf("V0: %g; h: %g; alpha: %g\n", V0, h, alpha);

  // compute variables not depending on V0
  const double c1 = sqrt(M_PI_2);
  const double c2 = 2.+c1;
  double b = (1.-alpha)/alpha;

  // MYMY
  // GetRNGstate();

  // for(int i = 0; i < n; i++) { /**< for each of the n required variates */

    /**< set lambda for our parameterization */
    double lambda_alpha = pow(h,alpha)*V0; /**< Marius Hofert: work directly with lambda^alpha (numerically more stable for small alpha) */

    /**
     * Apply the algorithm of Devroye (2009) to draw from
     * \tilde{S}(alpha, 1, (cos(alpha*pi/2))^{1/alpha}, I_{alpha = 1},
     *       lambda*I_{alpha != 1};1) with Laplace-Stieltjes transform
     * exp(-((lambda+t)^alpha-lambda^alpha))
     */
    double gamma = lambda_alpha*alpha*(1.-alpha);
    double sgamma = sqrt(gamma);
    double c3 = c2* sgamma;
    double xi = (1. + M_SQRT2 * c3)/M_PI; /**< according to John Lau */
    double psi = c3*exp(-gamma*M_PI*M_PI/8.)/M_SQRT_PI;
    double w1 = c1*xi/sgamma;
    double w2 = 2.*M_SQRT_PI * psi;
    double w3 = xi*M_PI;
    double X, c, E;

    #ifdef USE_R
    int outiter = 0;
    int initer  = 0;
    #endif

    do {
      double U, z, Z;

      #ifdef USE_R
      if (outiter++ % 1000 == 0) R_CheckUserInterrupt();
      #endif

      do {

      #ifdef USE_R
	if (initer++ % 1000 == 0) R_CheckUserInterrupt();
      #endif

	// MYMY
	// Rprintf("lambda^alpha=%g; b=%g; w1=%g; w2=%g, w3=%g\n", lambda_alpha, b, w1, w2, w3);

        // double V = unif_rand();
	double V = r.unif();
        if(gamma >= 1) {
          //if(V < w1/(w1+w2)) U = fabs(norm_rand())/sgamma;
	  if(V < w1/(w1+w2)) U = fabs(r.norm(0.0, 1.0))/sgamma;
          else{
            // double W_ = unif_rand();
	    double W_ = r.unif();
            U = M_PI*(1.-W_*W_);
          }
        }
        else{
          // double W_ = unif_rand();
	  double W_ = r.unif();
          if(V < w3/(w2+w3)) U = M_PI*W_;
          else U = M_PI*(1.-W_*W_);
        }
        // double W = unif_rand();
	double W = r.unif();
        double zeta = sqrt(BdB0(U,alpha));
        z = 1/(1-pow(1+alpha*zeta/sgamma,-1/alpha)); /**< Marius Hofert: numerically more stable for small alpha */
        /**< compute rho */
        double rho = M_PI*exp(-lambda_alpha*(1.-1. \
                                             /(zeta*zeta))) / \
          ((1.+c1)*sgamma/zeta \
           + z);
        double d = 0.;
        if(U >= 0 && gamma >= 1) d += xi*exp(-gamma*U*U/2.);
        if(U > 0 && U < M_PI) d += psi/sqrt(M_PI-U);
        if(U >= 0 && U <= M_PI && gamma < 1) d += xi;
        rho *= d;
        Z = W*rho;

	// MYMY
	// Rprintf("U=%g; Z=%g\n", U, Z);

      } while( !(U < M_PI && Z <= 1.)); /* check rejection condition */

      // MYMY
      // Rprintf("Inner rejection sampling complete.\n");

      double a = pow(A_(U,alpha), 1./(1.-alpha));
      double m = pow(b/a,alpha)*lambda_alpha;
      double delta = sqrt(m*alpha/a);
      double a1 = delta*c1;
      double a2 = delta; // MYMY
      double a3 = z/a;
      double s = a1+a2+a3;

      // MYMY
      // Rprintf("m=%g; delta=%g; a1=%g; a2=%g; a3=%g\n", m, delta, a1, a2, a3);

      // double V_ = unif_rand(), N_ = 0., E_ = 0. /**< -Wall */;
      double V_ = r.unif(), N_ = 0., E_ = 0. /**< -Wall */;
      if(V_ < a1/s) {
        // N_ = norm_rand();
	N_ = r.norm(0.0, 1.0);
        X = m-delta*fabs(N_);
      } else {
        if(V_ < (a1+a2)/s)
          // X = m+delta*unif_rand();
	  X = m+delta*r.unif();
        else {
          // E_ = exp_rand();
	  E_ = r.expon_rate(1.0);
          X = m+delta+E_*a3;
        }
      }
      E = -log(Z);
      /**< check rejection condition */
      // c = a*(X-m)+exp((1/alpha)*log(lambda_alpha)-b*log(m))*(pow(m/X,b)-1); /**< Marius Hofert: numerically more stable for small alpha */

      // MYMY - I want to lambda to be able to be zero.
      // c = a * (X - m) + h * (pow(X, -1.*b) - pow(m, -1.*b));
      // Problem is that 0^-1 is nan to C++ and nan * 0 = 0.
      c = a * (X - m);
      c += (m!=0) ? h * (pow(X, -1.*b) - pow(m, -1.*b)) : 0.0;

      if(X < m) c -= N_*N_/2.;
      else if(X > m+delta) c -= E_;

      // MYMY
      // Rprintf("X=%g; c=%g; E=%g\n", X, c, E);

    } while (!(X >= 0 && c <= E));
    /**
     * Transform variates from the distribution corresponding to the
     * Laplace-Stieltjes transform exp(-((lambda+t)^alpha-lambda^alpha))
     * to those of the distribution corresponding to the Laplace-Stieltjes
     * transform exp(-V_0((h+t)^alpha-h^alpha)).
     */
    // St[i] = exp(1/alpha*log(V0[i])-b*log(X)); /**< Marius Hofert: numerically more stable for small alpha */

    // } /**< end for */

  // MYMY
  // PutRNGstate();

  return exp(1/alpha*log(V0)-b*log(X));
}
