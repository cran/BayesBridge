// -*- mode: c++; -*-

//////////////////////////////////////////////////////////////////////

// Copyright 2012 Nicholas G. Polson, James G. Scott, Jesse Windle
// Contact info: <jwindle@ices.utexas.edu>.

// This file is part of BayesBridge.

// BayesBridge is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
  
// BayesBridge is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
  
// You should have received a copy of the GNU General Public License
// along with BayesBridge.  If not, see <http://www.gnu.org/licenses/>.
			      
//////////////////////////////////////////////////////////////////////
/*
 Herein we provide conditional posterior sampling and expectation
 maximization for Bridge Regression a la Polson, Scott, and Windle.
 For a detailed description of the specifics of the setup and
 algorithm please see their paper The Bayesian Bridge
 (http://arxiv.org/abs/1109.2279).

 One starts with the basic regression:

   y = X beta + ep, ep ~ N(0, sig2 I).

 To regularize the selection of beta one may chose a variety of
 priors.  In the Bridge regression the prior lives within the familiy
 of exponential prior distributions described by

   p(beta | alpha, tau) \propto exp( - | beta_j / tau |^alpha ).

 GIBBS SAMPLING:

 The challenge is to compute the posterior distribution efficiently.
 Polson and Scott use a Normal mixure of Bartlett-Fejer kernels to
 carry out this procedure.

 EXPECTATION MAXIMIZATION:

 They also present an expectation maximization algorithm to calculate
 the posterior mode of beta.

 */
//////////////////////////////////////////////////////////////////////

#ifndef __BRIDGEREGRESSION__
#define __BRIDGEREGRESSION__

#include "retstable.h"
#include <cmath>
#include <ctime>
#include <limits>
// HMC HMC HMC
// #include <Eigen/Core>
// #include <Eigen/SVD>
// #include "HmcSampler.h"

#include "Matrix.h"
#include "RNG.h"

#ifdef USE_R
#include <R_ext/Utils.h>
#endif

#define MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )

//////////////////////////////////////////////////////////////////////
			// CLASS DEFINITION //
//////////////////////////////////////////////////////////////////////

class BridgeRegression
{

 protected:

  // Dimension of beta.
  uint P;

  // Consider two cases: N > P and N < P.

  // Stored values, which are reused.
  Matrix y;
  Matrix X;
  Matrix XX;
  Matrix Xy;
  Matrix XX_sub;
  Matrix bhat;

  Matrix RT;
  Matrix RTInv;

  Matrix tV;
  Matrix a;
  Matrix d;

  // HMC HMC HMC
  // Eigen::MatrixXd F;
  // Eigen::MatrixXd tUy;
  // Eigen::MatrixXd DtV;
  // Eigen::MatrixXd EXX;
  // Eigen::MatrixXd ebhat;
  // Eigen::MatrixXd Fb;

 public:

  // Constructors:
  BridgeRegression();
  BridgeRegression(const MF& X_, const MF& y_);

  // Least squares solution.
  void least_squares(Matrix & ls);

  // For sampling beta
  void rtnorm_gibbs(MF beta, MF bmean, MF Prec, double sig2, MF b, RNG& r);
  void rtnorm_gibbs_wrapper(MF beta, double sig2, MF b, RNG& r);

  static void rtnorm_gibbs(double *betap, 
			   double *ap, double *tVp, double *dp, 
			   double* bp, double* sig2p, 
			   int *Pp, RNG& r);

  void rtnorm_hmc(MF beta, MF beta_prev, double sig2, MF b, int niter=1, int seed=0);

  void sample_beta(MF beta, const MF& beta_prev, 
		   const MF& u, const MF& omega, 
		   double sig2, double tau, double alpha, 
		   RNG& r, int burn=0, bool use_hmc=false);

  void sample_beta_ortho(MF beta, const MF& beta_prev, 
			 const MF& u, const MF& omega, 
			 double sig2, double tau, double alpha, 
			 RNG& r, int burn=0);

  // For sampling everything else.
  void sample_u(MF u, const MF& beta, const MF& omega, double tau, double alpha, RNG& r);
  void sample_omega(MF omega, const MF& beta, const MF& u, double tau, double alpha, RNG& r);
  void sample_omega(MF omega, MF shape, const MF& beta, const MF& u, double tau, double alpha, RNG& r);
  void sample_sig2(MF sig2, const MF& beta, double sig2_shape, double sig2_scale, RNG& r);
  void sample_tau_tri(MF tau, const MF& beta, const MF& u, const MF& w, double alpha, 
		      double tau2_shape, double tau2_scale, RNG& r);

  void sample_tau_marg(MF tau, const MF& beta, double alpha, double nu_shape, double nu_rate, RNG& r);
  double llh_alpha_marg(double alpha, const MF& s, RNG& r);
  void sample_alpha_marg(MF alpha, const MF& alpha_prev, const MF& beta, double tau, 
			 RNG& r, double pr_a, double pr_b, double ep=0.1);

  void sample_lambda(MF lambda, MF beta, double alpha, double tau, RNG& r);
  void sample_beta_stable(MF beta, MF lambda, double alpha, double sig2, double tau, RNG& r);
  void sample_beta_stable_ortho(MF beta, MF lambda, double alpha, double sig2, double tau, RNG& r);
  void sample_tau_stable(MF tau, const MF& beta, const MF& lambda, double tau2_shape, double tau2_scale, RNG& r);

  // Expectation Maximization.
  int EM(Matrix& beta, double sig, double tau, double alpha,
	 double lambda_max, double tol, int max_iter, bool use_cg=false);

};

// BR is BridgeRegression
#ifndef BR
typedef BridgeRegression BR;
#endif

//////////////////////////////////////////////////////////////////////
			  // Constructors //
//////////////////////////////////////////////////////////////////////

BR::BridgeRegression()
{
  Rprintf( "Warning: Default constructor called.");
} // BridgeRegression

//--------------------------------------------------------------------
BR::BridgeRegression(const MF& X_, const MF& y_) 
  : P(X_.cols())
  , y(y_)
  , X(X_)
  , XX(P, P)
  , Xy(P)
{
  // Check conformity.
  if (X.rows()!=y.rows())
    Rprintf( "Error: X and y do not conform.");

  gemm(XX, X, X, 'T', 'N');
  gemm(Xy, X, y, 'T', 'N');

  // Need to deal with P = 1.
  if (P > 1) {
    XX_sub.resize(1, P-1, P);

    Matrix ss("N", P-1);
    for(uint j = 0; j < P; j++){
      XX_sub[j].copy(XX, j, ss);
      if (j < P-1) ss(j) = j;
    }
  }

  symsqrt(RT, XX);
  syminvsqrt(RTInv, XX);

  bhat.resize(P);
  least_squares(bhat);

  int n = X_.rows();
  int p = X_.cols();

  Matrix U;
  if (n > p)
    svd(U, d, tV, X, 'S');
  else
    svd(U, d, tV, X, 'A');

  Matrix A(U); prodonrow(A, d);
  mult(a, A, y, 'T', 'N');

  // HMC HMC HMC
  // For HMC: We need to have a non-singular precision for this to work.
  // F = svd.singularValues().asDiagonal().inverse() * EtV; // D^{-1} V'
  // DtV = svd.singularValues().asDiagonal() * EtV;
  // tUy = svd.matrixU().transpose() * Eigen::Map<Eigen::MatrixXd>(&y(0), n, 1);
  
  // EXX = Eigen::Map<Eigen::MatrixXd>(&XX(0), p, p);
  // ebhat = Eigen::Map<Eigen::MatrixXd>(&bhat(0), p, 1);

  // Fb.resize(2*p, p);
  // Fb.block(0,0,p,p) = Eigen::MatrixXd::Identity(p, p);
  // Fb.block(p,0,p,p) = -1.0 * Eigen::MatrixXd::Identity(p,p);

} // Bridge Regression

//////////////////////////////////////////////////////////////////////
			// Helper Functions //
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
			 // Least Squares //
//////////////////////////////////////////////////////////////////////

void BR::least_squares(Matrix & ls)
{
  try {
    ls.clone(Xy);
    symsolve(XX, ls);
  }
  catch (std::exception& e){
    ls.clone(Matrix(P)); 
    ls.fill(0.0);
    Rprintf("Warning: cannot calculate least squares estimate; X'X is singular.\n");
    Rprintf("Warning: setting least squares estimate to 0.0.\n");
  }
}

//////////////////////////////////////////////////////////////////////
		       // Posterior Samplers //
//////////////////////////////////////////////////////////////////////

void BR::sample_u(MF u, const MF& beta, const MF& omega, double tau, double alpha, RNG& r)
{
  double right;
  for(uint j = 0; j < P; j++){
    right = 1 - fabs(beta(j)) / tau * exp( -1.0 * log(omega(j)) / alpha );
    // u(j) = right > 0 ? r.flat(0, right) : 0;
    u(j) = r.flat(0, right);
    // COMMENT COMMENT
    if (u(j) < 0) {
      Rprintf("Warning: sampled negative value for u.\n");
      Rprintf("%g %g %g %g\n", beta(j), omega(j), tau, right);
    }
  }
} // sample_u

//--------------------------------------------------------------------
void BR::sample_omega(MF omega, const MF& beta, const MF& u, double tau, double alpha, RNG& r)
{
  double prob;
  for(uint j = 0; j < P; j++){
    double a_j = exp( alpha * log( fabs(beta(j)) / ( (1 - u(j)) * tau) ) );
    // prob = ( 1 - alpha * (1 + a(j)) ) / (1 - alpha * a(j));
    prob = alpha / (1 + alpha * a_j);
    if (r.unif() > prob){
      omega(j) = r.gamma_rate(1.0, 1.0);
    }
    else{
      omega(j) = r.gamma_rate(2.0, 1.0);
    }
    omega(j) += a_j;
  }
} // sample_omega

void BR::sample_omega(MF omega, MF shape, const MF& beta, const MF& u, double tau, double alpha, RNG& r)
{
  double prob;
  for(uint j = 0; j < P; j++){
    double a_j = exp( alpha * log( fabs(beta(j)) / ( (1 - u(j)) * tau) ) );
    // prob = ( 1 - alpha * (1 + a(j)) ) / (1 - alpha * a(j));
    prob = alpha / (1 + alpha * a_j);
    if (r.unif() > prob){
      shape(j) = 1.0;
      omega(j) = r.gamma_rate(1.0, 1.0);
    }
    else{
      shape(j) = 2.0;
      omega(j) = r.gamma_rate(2.0, 1.0);
    }
    omega(j) += a_j;
  }
} // sample_omega

//------------------------------------------------------------------------------
void BR::sample_tau_tri(MF tau, const MF& beta, const MF& u, const MF& w, double alpha,
		    double tau2_shape, double tau2_scale, RNG& r)
{
  double m = -1.0;
  for(int j=0; j < (int)P; ++j) {
    double m_j = fabs(beta(j)) / ( (1-u(j)) * exp(log(w(j)) / alpha) );
    m = m < m_j ? m_j : m;
  }
  double ap = tau2_shape + 0.5 * (double)P;
  double bp = tau2_scale;
  double phi = r.rtgamma_rate(ap,bp, 1.0/(m*m));
  tau(0) = sqrt(1.0/phi);
}

//--------------------------------------------------------------------

// Not using SVD.
// CONSIDER TAKING THIS OUT.  THE SVD METHOD SEEMS TO BE THE WAY TO GO.
void BR::rtnorm_gibbs(MF beta, MF bmean, MF Prec, double sig2, MF b, RNG& r)
{
  // Matrix RT, RTInv;
  // symsqrt(RT, Prec);
  // syminvsqrt(RTInv, Prec);

  Matrix RTInvZSub(P, P);
  RTInvZSub.fill(0.0);

  Matrix m; mult(m, RT, bmean);
  Matrix z; mult(z, RT, beta);
  double v = sig2; 

  Matrix ss("N", P-1);

  for (uint i=0; i<P; i++) {

    Matrix left(P); left.fill(0.0);
    Matrix right(P); right.fill(0.0);

    for (uint j=0; j<P; j++) {

      for (uint k=0; k<P-1; k++)
	RTInvZSub(j,i) += RTInv(j,ss(k)) * z(ss(k));

      if (RTInv(j,i) > 0) {
	left(j)  = (-1.0*b(j) - RTInvZSub(j,i)) / fabs(RTInv(j,i));
	right(j) = (     b(j) - RTInvZSub(j,i)) / fabs(RTInv(j,i));
      }
      else {
	left(j)  = -1.0*(b(j) - RTInvZSub(j,i)) / fabs(RTInv(j,i));
	right(j) = (     b(j) + RTInvZSub(j,i)) / fabs(RTInv(j,i));
      }

    }

    double lmax = maxAll(left);
    double rmin = minAll(right);
    
    try {
      z(i) = r.tnorm(lmax, rmin, m(i), sqrt(v));
    }
    catch (std::exception& e) {
      Rprintf("left: %g, right: %g, z[i]: %g\n", lmax, rmin, z(i));
      // cout << "beta: " << beta;
      // cout << "b: " << b;
      throw e;
    }

    if (i < P-1) ss(i) = i;

  }

  gemm(beta, RTInv, z);

}

////////////////////////////////////////////////////////////////////////////////

// Using the SVD.
void BR::rtnorm_gibbs_wrapper(MF beta, double sig2, MF b, RNG& r)
{
  int Pint = P;
  rtnorm_gibbs(&beta(0), &a(0), &tV(0), &d(0), &b(0), &sig2, &Pint, r);
}

void BR::rtnorm_gibbs(double *betap, 
		      double *ap, double *tVp, double *dp, 
		      double* bp, double* sig2p, 
		      int *Pp, RNG& r)
{
  // Anything with a "p" suffix is a pointer.

  int P = *Pp;
  double sig = sqrt(*sig2p);
  MatrixFrame beta(betap, P);
  MatrixFrame tV(tVp, P, P);
  Matrix z; mult(z, tV, beta);
  double *zp = &z(0);
  // Matrix vj(P);

  for (int i=0; i<P; i++) {
    double lmax = -1.*std::numeric_limits<double>::max();
    double rmin =     std::numeric_limits<double>::max();

    for (int j=0; j<P; j++) {
      double vji = tVp[i+j*P];
      MatrixFrame vj(&tVp[j*P], P);
      // double rji = dot(tV.col(j), z) - vji * zp[i];
      double rji = dot(vj, z) - vji * zp[i];
      double dif = bp[j] - rji;
      double sum = bp[j] + rji;
      double left  = (vji > 0 ? -sum : -dif) / fabs(vji);
      double right = (vji > 0 ?  dif :  sum) / fabs(vji);
      lmax = MAX(lmax, left );
      rmin = MIN(rmin, right);
    }

    // double dx = rmin - lmax;
    double mean = ap[i] / (dp[i] * dp[i]);
    double sd   = sig / dp[i];

    // I need to be careful here.  It may be the case that dp is almost zero or negative!
    if (dp[i] > 1e-16){
      zp[i] = r.tnorm(lmax, rmin, mean, sd);
    }
    else {
      // double lw = lmax < rmin ? lmax : rmin;
      // double up = lmax > rmin ? lmax : rmin;
      // if (lw!=lmax) Rprintf("Problem with lmax,rmin: %g, %g \n", lmax, rmin);
      // zp[i] = lw + (up-lw) * r.unif();
      zp[i] = r.flat(lmax, rmin);
    }

  }

  gemm(beta, tV, z, 'T', 'N');
}

////////////////////////////////////////////////////////////////////////////////

// HMC HMC HMC
// // Hamiltonian Mone Carlo
// void BR::rtnorm_hmc(MF beta, MF beta_prev, double sig2, MF b, int burn, int seed)
// {
//   if (seed==0) {
//     // Since using ctime time is in std namespace.
//     seed = std::time(NULL);
//     // Checkout clock_gettime for POSIX systems.
//   }

//   int d = P;
//   // HmcSampler hmc(d, seed);
//   // double sig = sqrt(sig2);

//   // Eigen::MatrixXd tFtUy = F.transpose() * tUy;

//   // // Set initial value.
//   // Eigen::Map<Eigen::VectorXd> ebeta_prev(&beta_prev(0), d);
//   // Eigen::VectorXd z_init = (DtV * ebeta_prev - tUy) / sig;
//   // hmc.setInitialValue(z_init);

//   // // Set constraints.
//   // for (int j=0; j<d; j++) {
//   //   double gj_plus = (b(j) + tFtUy(j)) / sig;
//   //   double gj_mnus = (b(j) - tFtUy(j)) / sig;
//   //   Eigen::VectorXd Fj = F.col(j);
//   //   hmc.addLinearConstraint(Fj, gj_plus);
//   //   hmc.addLinearConstraint(-1.0 * Fj, gj_mnus);
//   //   // double cj = Fj.dot(z_init);
//   //   // Rprintf("cj: %g, gj+: %g, gj-: %g\n", cj, gj_plus, gj_mnus);
//   // }

//   // for (int i=0; i<niter-1; i++) {
//   //   hmc.sampleNext(false);
//   // }  

//   // // Returns a samples in row format.  I had some problems with this.
//   // Eigen::VectorXd draw = hmc.sampleNext(false);
//   // Eigen::MatrixXd newbeta = F.transpose() * (sig * draw + tUy);

//   // New attempt
//   Eigen::MatrixXd prec = EXX / sig2;
//   Eigen::MatrixXd eb   = Eigen::Map<Eigen::MatrixXd>(&b(0), d, 1);
//   Eigen::MatrixXd ebprev = Eigen::Map<Eigen::MatrixXd>(&beta_prev(0), d, 1);
//   Eigen::VectorXd gb(2*d);

//   gb.segment(0,d) = eb;
//   gb.segment(d,d) = eb;
  
//   // std::cerr << "Fb:\n" << Fb << "\n";
//   // std::cerr << "eb:\n" << eb.transpose() << "\n";
//   // std::cerr << "iv:\n" << ebprev.transpose() << "\n";

//   Eigen::MatrixXd newbeta(d, 1);
//   // newbeta = hmc.rtnorm(ebhat, prec, Fb, gb, ebprev, 1, burn, false, seed);
//   Rprintf("You should NOT be using hmc sampling.\n");

//   // std::cout << "newbeta:\n" << newbeta << "\n";

//   for(int i=0; i<d; i++)
//      beta(i) = newbeta(i);

// }

////////////////////////////////////////////////////////////////////////////////

// There are multiple ways one may calculate the conditional distributions of
// beta_j | beta_{-j}.  Initially, I considered the joint distribution and then
// used regression theory to calculate the conditional.  This is a bad idea--you
// could have singular precisions.  It is better to calculate beta_j based upon
// likelihood.

void BR::sample_beta_ortho(MF beta, const MF& beta_prev, const MF& u, const MF& omega, double sig2, double tau, double alpha, RNG& r, int burn)
{

  Matrix beta_sub(P-1);
  Matrix XXbeta(1); 

  beta.copy(beta_prev);

  int niter = burn + 1;

  for(int i=0; i<niter; i++){

    Matrix ss("N", P-1);

    for(uint j = 0; j < P; j++){

      XXbeta(0) = 0.0;
      
      // If P > 1.
      beta_sub.copy(beta, ss, 0);
      gemm(XXbeta, XX_sub[j], beta_sub);
      // Else keep XXbeta(0) = 0.

      // mean and variance
      double m = ( Xy(j) - XXbeta(0) ) / XX(j,j);
      double v = sig2 / XX(j,j);

      // Calculate b(j).
      double b_j = (1.0 - u(j)) * exp( log(omega(j)) / alpha) * tau;

      beta(j) = r.tnorm(-1.0*b_j, b_j, m, sqrt(v));

      // COMMENT COMMENT
      if (fabs(beta(j)) > b_j) {
	Rprintf("b(j) problem: b(j): %g, beta(j): %g\n", b_j, beta(j));
      }
      if (j < P-1) ss(j) = j;
    }

  }

}

void BR::sample_beta(MF beta, const MF& beta_prev, const MF& u, const MF& omega, double sig2, double tau, double alpha, RNG& r, int burn, bool use_hmc)
{
  burn = burn > 0 ? burn : 0;

  Matrix b(P);
  for(uint j=0; j<P; j++)
    b(j) = (1.0 - u(j)) * exp( log(omega(j)) / alpha) * tau;

  beta.copy(beta_prev);

  // Could precompute.
  // Matrix bhat(P);
  // least_squares(bhat);

  use_hmc = false;
  if (!use_hmc) {
    int niter = burn + 1;
    for(int i=0; i<niter; i++) {
      // rtnorm_gibbs(beta, bhat, XX, sig2, b, r);
      rtnorm_gibbs_wrapper(beta, sig2, b, r);
    }
  }
  else {
    // HMC
    // rtnorm_hmc(beta, beta_prev, sig2, b, burn, floor(r.unif() * 10000000));
    // Try setting the seed to 0 for all draws.  Look what happens.  Things are off.
  }

} // sample_beta

//--------------------------------------------------------------------
void BR::sample_sig2(MF sig2, const MF& beta, double sig2_shape, double sig2_scale, RNG& r)
{
  double shape = sig2_shape + 0.5 * y.rows();

  Matrix temp(y);
  gemm(temp, X, beta, 'N', 'N', -1.0, 1.0);

  double rss = dot(temp, temp);

  double scale = sig2_scale + 0.5 * rss;

  // cout << shape << " " << scale << "\n";

  sig2(0) = r.igamma(shape, scale);
} // sample_sig2

//--------------------------------------------------------------------
void BR::sample_tau_marg(MF tau, const MF& beta, double alpha, double nu_shape, double nu_rate, RNG& r)
{
  double shape = nu_shape + ((double)P) / alpha;

  double rate = nu_rate;
  for(uint j = 0; j < P; j++){
    rate += exp( alpha * log(fabs(beta(j))) );
  }

  double nu = r.gamma_rate(shape, rate);

  tau(0) = exp(-1.0 * log(nu) / alpha);
} // sample_tau

//------------------------------------------------------------------------------

double BR::llh_alpha_marg(double alpha, const MF& s, RNG& r)
{
  double p = (double)s.vol();
  double llh = p * log(alpha) - p * r.Gamma(1.0/alpha, true);
  for (int i = 0; i < (int)p; i++)
    llh -= exp(alpha * s(i));
  return llh;
}

void BR::sample_alpha_marg(MF alpha, const MF& alpha_prev, const MF& beta, double tau, 
			   RNG& r, double pr_a, double pr_b, double ep)
{
  Matrix s(P);
  for (uint i = 0; i < P; i++)
    s(i) = log(fabs(beta(i) / tau));

  double a_old = alpha_prev(0);

  double l_new = max(0.0, a_old - ep);
  double r_new = min(1.0, a_old + ep);
  double d_new = r_new - l_new;
  double a_new = r.flat(l_new, r_new);

  double l_old = max(0.0, a_new - ep);
  double r_old = min(1.0, a_new + ep);
  double d_old = r_old - l_old;

  double log_accept = 
    llh_alpha_marg(a_new, s, r) - llh_alpha_marg(a_old, s, r) \
    + log(r.d_beta(a_new, pr_a, pr_b)) - log(r.d_beta(a_old, pr_a, pr_b)) \
    + log(d_old) - log(d_new);
    
  alpha(0) = a_new;
  if (r.unif() > exp(log_accept)) alpha(0) = a_old;
}

//------------------------------------------------------------------------------
void BR::sample_lambda(MF lambda, MF beta, double alpha, double tau, RNG& r)
{
  for (int j=0; j<(int)P; j++)
    lambda(j) = 2 * retstable_LD(beta(j)*beta(j) / (tau*tau), 0.5 * alpha, r);
}

//------------------------------------------------------------------------------

void BR::sample_beta_stable_ortho(MF beta, MF lambda, double alpha, double sig2, double tau, RNG& r){
     for(uint i=0; i<P; i++) {
      double u = XX(i,i) + lambda(i) * sig2 / (tau * tau);
      double s = sqrt(sig2 / u);
      double m = Xy(i) / u;
      beta(i) = r.norm(m, s);
    }
 }

// void BR::sample_beta_stable(MF beta, MF lambda, double alpha, double sig2, double tau, RNG& r)
// {

//   Matrix VInv(XX);
//   for(uint i=0; i<P; i++)
//     VInv(i,i) += lambda(i) * sig2 / (tau * tau);
//   // cout << "VInv:\n" << VInv;

//   Matrix V;
//   syminv(VInv, V);
//   // cout << "V:\n" << V;

//   Matrix L;
//   chol(L, V);
//   hprodeq(L, sqrt(sig2));
//   // cout << "L:\n" << L;

//   // The mean
//   gemm(beta, V, Xy, 'N', 'N');
//   // cout << "Beta:\n" << beta;

//   Matrix ndraw(P);
//   r.norm(ndraw, 0.0, 1.0);
//   // cout << "ndraw:\n" << ndraw;

//   gemm(beta, L, ndraw, 'N', 'N', 1.0, 1.0);

// }

void BR::sample_beta_stable(MF beta, MF lambda, double alpha, double sig2, double tau, RNG& r)
{
  Matrix VInv(XX);
  for(uint i=0; i<P; i++)
    VInv(i,i) += lambda(i) * sig2 / (tau * tau);
  // cout << "VInv:\n" << VInv;

  Matrix U;
  chol(U, VInv, 'U');

  // The mean
  beta.copy(Xy);
  trsm(U, beta, 'U', 'L', 'T');
  trsm(U, beta, 'U', 'L');

  Matrix ndraw(P);
  r.norm(ndraw, 0.0, 1.0);

  trsm(U, ndraw, 'U', 'L');

  double sig = sqrt(sig2);
  for(int i=0; i<(int)P; i++)
    beta(i) += sig * ndraw(i);
}

//------------------------------------------------------------------------------
void BR::sample_tau_stable(MF tau, const MF& beta, const MF& lambda, double tau2_shape, double tau2_scale, RNG& r)
{
  double ap = tau2_shape + 0.5 * (double) P;
  double bp = tau2_scale;
  for (int i = 0; i < (int)P; ++i)
    bp += 0.5 * beta(i)*beta(i)*lambda(i);
  // Don't forget the 0.5
  double phi = r.gamma_rate(ap,bp);
  tau(0) = sqrt(1/phi);
}

//////////////////////////////////////////////////////////////////////
			   // EM DIRECT //
//////////////////////////////////////////////////////////////////////

// Solves system directly or using conjugate gradient method.

// Returns the order of the number of "solves" needed to find a solution.  To
// find x in Ax = b you need p solves where p is the dimension of b.  Using the
// conjugate gradient method you cand find an okay x in fewer than p iterations.
// This algorithm boils down to solving a linear system several times.

int BR::EM(Matrix& beta, double sig, double tau, double alpha,
	   double lambda_max, double tol, int max_iter, bool use_cg)
{

  Matrix lambda(P);
  Matrix ss("W", P);

  // Matrix XX; mult(XX, X, X, 'T', 'N'); // Already exists
  Matrix b;  mult(b, X, y, 'T', 'N');  // Already exists

  double dist = tol + 1.0;
  int    iter = 0;
  int    p    = P;
  int    N    = X.rows();

  int total_iter = p;  // We do a symsolve below.

  // Regarding comment below.  I think James let tau^* = tau/sigma and then
  // dropped the *.  Having both parameters tau and sig2 is redundant since only
  // the ratio tau/sigma matters.  However, it is somewhat confusing to read the
  // paper since they don't say this.
  double c1   = alpha * exp( (2-alpha) * (log(tau) - log(sig)) );
  double c2   = exp( -2 * (log(tau) - log(sig)) );

  Matrix EM_X(X);
  Matrix EM_b(b);
  Matrix old_beta(b);

  // Do one maximization step.
  Matrix A(XX);
  Matrix new_beta(b);
  symsolve(A, new_beta);

  while (dist > tol && iter < max_iter){

    // Expectation Step.
    uint num = 0;
    for(uint j = 0; j < (uint)p; j++){
      lambda(j) = c1 * exp( (alpha - 2) * log( fabs(new_beta(j)) ) );
      if (lambda(j) < lambda_max) {
	  ss(num) = ss(j);
	  lambda(num) = lambda(j);
	  old_beta(num) = new_beta(j);
	  ++num;
      }
    }

    // Delete entries that are too large.
    if (num < (uint)p) {
      if (num == 0){
	beta.clone(Matrix(P));
	return iter;
      }
      p = num;
      ss.resize(p);
      lambda.resize(p);
      old_beta.resize(p);
      EM_X.clone(X, Matrix("W", N), ss);
      XX.resize(p, p);
      gemm(XX, EM_X, EM_X, 'T', 'N');
      EM_b.clone(b, ss, 0);
    }

    // Maximization step.
    A.clone(XX);
    for(uint j = 0; j < (uint)p; j++)
      A(j,j) += c2 * lambda(j);
    if (!use_cg) { // Solve the system using LAPACK - by Cholesky I think.
      new_beta.clone(EM_b);
      symsolve(A, new_beta);
      total_iter = total_iter + p;
    }
    else{ // Solve the system using conjugate gradient method.
      new_beta.clone(old_beta);
      int cg_iter = cg(new_beta, A, EM_b, tol, p);
      total_iter = total_iter + cg_iter;
    }

    // Regarding the conjugate gradient method.  There is the following
    // potential problem: I found that things weren't converging when the
    // tolerance wasn't strict enough, i.e. if the conjugate gradient method
    // stops without getting close to the solution to Ax = b.  Suppose when I
    // don't set the tolerance low enough I calculate x1.  On the expectation
    // step x1 may not move lambda that much.  Then on the next maximization
    // step we are going to be solving essentially the same problem since the
    // operator A won't have changed that much.  But the tolerance will be the
    // same so we will be calculating x2 which is almost the same as x1, both of
    // which are the wrong solution.

    // UPDATE: This may have had something to do with the tolerance in the cg
    // algorithm.  Previously the tolernace wasn't an absolute thing but
    // relative to delta_0.

    // One possible solution.  At the last step, run the conjugate gradient
    // method p steps to get the best possible solution.

    // Calculate distance and increment iter.
    Matrix diff = new_beta - old_beta;
    dist = sqrt( dot(diff, diff) );
    ++iter;
  }

  beta.clone(Matrix(P));
  for(uint j = 0; j < (uint)p; j++){
    beta(ss(j)) = new_beta(j);
  }

  return total_iter;
}

//------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////
			  // END OF FILE //
//////////////////////////////////////////////////////////////////////

#endif

// void BR::preprocess(const MF& var)
// {
//   Matrix idx;

//   Matrix ss("N", P-1);

//   Matrix Sigma_12((uint)1, P-1);
//   Matrix Sigma_22(    P-1, P-1);

//   for(uint j = 0; j < P; j++){

//     Sigma_12.copy(var,  j, ss);
//     Sigma_22.copy(var, ss, ss);

//     // Regression Matrix.
//     Matrix Prec_22("I", P-1);
//     posv(Sigma_22, Prec_22, 'L');
//     gemm(regmat[j], Sigma_12, Prec_22);

//     // Mean Term.
//     Matrix m2(P-1);
//     m2.copy(betahat, ss, 0);
//     mean_term[j](0) = betahat(j);
//     gemm(mean_term[j], regmat[j], m2, 'N', 'N', -1.0, 1.0);

//     // Conditional Variance / SD.
//     Matrix temp(1);
//     gemm(temp, regmat[j], Sigma_12, 'N', 'T');
//     condsd(j) = sqrt(var(j,j) - temp(0));

//     if (j < P-1) ss(j) = j;
//   }

//   // cout << "mean_term:\n" << MatrixFrame(&mean_term(0), P);
//   // cout << "betahat:\n" << betahat;

// }
