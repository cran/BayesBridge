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

#include <time.h>
// #include <math.h>
#include <cmath>
#include <exception>

#include "BridgeWrapper.h"

// Trying to avoid problems with length macro.
#ifdef USE_R
#include "R.h"
#include "Rmath.h"
#include <R_ext/Utils.h>
#include <R_ext/PrtUtil.h>
#endif

// #include <algorithm>

// Test function.
// double add(double *sum, double *a, double *b)
// {
//   *sum = *a + *b;
// }

//////////////////////////////////////////////////////////////////////
		    // EXPECTATION MAXIMIZATION //
//////////////////////////////////////////////////////////////////////

int EM(Matrix & beta, MatrixFrame &y, MatrixFrame &X,
	double ratio, double alpha, double lambda_max,
	double tol, int max_iter, bool use_cg)
{
  BridgeRegression br(X, y);
  int iter = -1;

  try{
    iter =  br.EM(beta, 1.0, ratio, alpha, lambda_max, tol, max_iter, use_cg);
  }
  catch (std::exception& e) {
    Rprintf("Error: %s\n", e.what());
    Rprintf("Aborting EM.\n");
  }

  return iter;
}

//////////////////////////////////////////////////////////////////////
		       // BRIDGE REGRESSION //
//////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------
double bridge_regression(MatrixFrame & beta,
			 MatrixFrame & u,
			 MatrixFrame & omega,
			 MatrixFrame & shape,
			 MatrixFrame & sig2,
			 MatrixFrame & tau,
			 MatrixFrame & alpha, 
			 const MatrixFrame & y,
			 const MatrixFrame & X,
			 double sig2_shape,
			 double sig2_scale,
			 double nu_shape,
			 double nu_rate,
			 double alpha_a,
			 double alpha_b,
			 double true_sig2, // 
			 double true_tau , // Stored in tau  if true.
			 double true_alpha, 
			 uint burn,
			 int betaburn,
			 bool use_hmc)
{
  BridgeRegression br(X, y);

  // uint P = X.cols();
  // uint N = X.rows();
  uint M = beta.mats();
  bool know_sig2 = true_sig2 > 0;
  bool know_tau  = true_tau  > 0;
  bool know_alpha = true_alpha > 0;

  // Details.
  Rprintf("Bridge Regression (mix. of triangles):");
  if (know_alpha) Rprintf(" known alpha=%g", true_alpha);
  if (know_sig2) Rprintf(", sig2=%g", true_sig2);
  if (know_tau ) Rprintf(", tau=%g", true_tau);
  Rprintf("\nBurn-in: %i, Num. Samples: %i\n", burn, M);
  if (use_hmc) Rprintf("Using HMC!\n");

  // Initialize.
  Matrix ls;
  br.least_squares(ls);
  beta[0].copy(ls);

  u[0].fill(0.5);
  alpha[0](0) = 0.5;

  if ( know_sig2 ) sig2.fill(true_sig2);
  if ( know_tau  ) tau.fill(true_tau);
  if ( know_alpha) alpha.fill(true_alpha);

  RNG r;

  // Keep track of time.
  clock_t start, end;
  double total_time;

  start = clock();

  try {

    // if (!know_sig2) br.sample_sig2(sig2[0], beta[0], sig2_shape, sig2_scale, r);  // Sample sig2.
    if (!know_tau ) br.sample_tau_marg(tau[0], beta[0], alpha[0](0), nu_shape, nu_rate, r);  // Sample tau.

    // Burn-In.
    for(uint i = 0; i < burn; i++){
      if (!know_tau) br.sample_tau_marg(tau[0], beta[0], alpha[0](0), nu_shape, nu_rate, r);  // Sample tau.
      if (!know_sig2) br.sample_sig2(sig2[0], beta[0], sig2_shape, sig2_scale, r);  // Sample sig2.
      br.sample_omega(omega[0], shape[0], beta[0], u[0], tau[0](0), alpha[0](0), r);
      // br.sample_omega(omega[0], beta[0], u[0], tau[0](0), alpha[0](0), r);
      br.sample_u(u[0], beta[0], omega[0], tau[0](0), alpha[0](0), r);
      br.sample_beta(beta[0], beta[0], u[0], omega[0], sig2[0](0), tau[0](0), alpha[0](0), r, betaburn, use_hmc);
      if (!know_alpha) br.sample_alpha_marg(alpha[0], alpha[0], beta[0], tau[0](0), r, alpha_a, alpha_b);
      // br.sample_tau_tri(tau[0], beta[0], u[0], omega[0], alpha, 4.0, 4.0, r);
      #ifdef USE_R
      if (i % 10 == 0) R_CheckUserInterrupt();
      #endif
    }

    end = clock();

    total_time = (double)(end - start) / CLOCKS_PER_SEC;
    Rprintf("Burn-in complete: %g sec. for %i iterations.\n", total_time, burn);
    Rprintf("Expect approx. %g sec. for %i samples.\n", total_time * M / burn, M);

    start = clock();

    // MCMC
    for(uint i = 1; i < M; i++){
      if (!know_tau) br.sample_tau_marg(tau[i], beta[i-1], alpha[i-1](0), nu_shape, nu_rate, r);  // Sample tau.
      if (!know_sig2) br.sample_sig2(sig2[i], beta[i-1], sig2_shape, sig2_scale, r);  // Sample sig2.
      br.sample_omega(omega[i], shape[i], beta[i-1], u[i-1], tau[i](0), alpha[i-1](0), r);
      // br.sample_omega(omega[i], beta[i-1], u[i-1], tau[i](0), alpha[i-1](0), r);
      br.sample_u(u[i], beta[i-1], omega[i], tau[i](0), alpha[i-1](0), r);
      // for (uint k=0; k < omegauburn; k++) { // begin omegauburn
      // 	br.sample_omega(omega[i], beta[i-1], u[i], tau[i](0), alpha, r);
      // 	br.sample_u(u[i], beta[i-1], omega[i], tau[i](0), alpha, r);
      // } // end omegauburn
      br.sample_beta(beta[i], beta[i-1], u[i], omega[i], sig2[i](0), tau[i](0), alpha[i-1](0), r, betaburn, use_hmc);
      // for (uint k=0; k < betauburn; k++) { // begin betauburn
      // 	br.sample_u(u[i], beta[i], omega[i], tau[i](0), alpha, r);
      // 	br.sample_beta(beta[i], beta[i], u[i], omega[i], sig2[i](0), tau[i](0), alpha, r, betaburn, use_hmc);
      // } // end betauburn
      // You must change indexing if the following is uncommented.
      // if (!know_tau) br.sample_tau_tri(tau[i], beta[i], u[i], omega[i], alpha, 4.0, 4.0, r);
      if (!know_alpha) br.sample_alpha_marg(alpha[i], alpha[i-1], beta[i], tau[i](0), r, alpha_a, alpha_b);
      #ifdef USE_R
      if (i % 10 == 0) R_CheckUserInterrupt();
      #endif
    }

  }
  catch (std::exception& e) {
    Rprintf("Error: %s\n", e.what());
    Rprintf("Aborting Gibbs sampler.\n");
  }

  end = clock();
  
  total_time = (double)(end - start) / CLOCKS_PER_SEC;
  Rprintf("Sampling complete: %g sec. for %i iterations.\n", total_time, M);

  return (double)(end - start) / CLOCKS_PER_SEC;

} // bridge_regression

//--------------------------------------------------------------------
double bridge_regression_stable(MatrixFrame & beta,
				MatrixFrame & lambda,
				MatrixFrame & sig2,
				MatrixFrame & tau,
				MatrixFrame & alpha,
				const MatrixFrame & y,
				const MatrixFrame & X,
				double sig2_shape,
				double sig2_scale,
				double nu_shape,
				double nu_rate,
				double alpha_a,
				double alpha_b,
				double true_sig2,
				double true_tau,
				double true_alpha,
				uint burn)
{
  BridgeRegression br(X, y);

  // uint P = X.cols();
  // uint N = X.rows();
  uint M = beta.mats();
  bool know_sig2 = true_sig2 > 0;
  bool know_tau  = true_tau  > 0;
  bool know_alpha = true_alpha > 0;

  // Details.
  Rprintf("Bridge Regression (mix. of normals):");
  if (know_alpha) Rprintf(" known alpha=%g", true_alpha);
  if (know_sig2) Rprintf(", sig2=%g", true_sig2);
  if (know_tau ) Rprintf(", tau=%g", true_tau);
  Rprintf("\nBurn-in: %i, Num. Samples: %i\n", burn, M);

  // Initialize.
  Matrix ls;
  br.least_squares(ls);
  beta[0].copy(ls);

  alpha[0](0) = 0.5;

  if ( know_sig2 ) sig2.fill(true_sig2);
  if ( know_tau  ) tau.fill(true_tau);
  if ( know_alpha) alpha.fill(true_alpha);

  RNG r;

  // Keep track of time.
  clock_t start, end;
  double total_time;

  start = clock();

  try {

    if (!know_tau) br.sample_tau_marg(tau[0], beta[0], alpha[0](0), nu_shape, nu_rate, r);  // Marginal
    // if (!know_sig2) br.sample_sig2(sig2[0], beta[0], sig2_shape, sig2_scale, r);  // Sample sig2.

    // Burn-In.
    for(uint i = 0; i < burn+1; i++){
      if (!know_tau) br.sample_tau_marg(tau[0], beta[0], alpha[0](0), nu_shape, nu_rate, r);  // Marginal
      if (!know_sig2) br.sample_sig2(sig2[0], beta[0], sig2_shape, sig2_scale, r);
      br.sample_lambda(lambda[0], beta[0], alpha[0](0), tau[0](0), r);
      br.sample_beta_stable(beta[0], lambda[0], alpha[0](0), sig2[0](0), tau[0](0), r);
      // if (!know_tau) br.sample_tau_stable(tau[0], beta[0], lambda[0], 4.0, 4.0, r);
      if (!know_alpha) br.sample_alpha_marg(alpha[0], alpha[0], beta[0], tau[0](0), r, alpha_a, alpha_b);
      #ifdef USE_R
      if (i % 10 == 0) R_CheckUserInterrupt();
      #endif
    }

    end = clock();
    
    total_time = (double)(end - start) / CLOCKS_PER_SEC;
    Rprintf("Burn-in complete: %g sec. for %i iterations.\n", total_time, burn);
    Rprintf("Expect approx. %g sec. for %i samples.\n", total_time * M / burn, M);
    
    start = clock();
    
    // MCMC
    for(uint i = 1; i < M; i++){
      if (!know_tau) br.sample_tau_marg(tau[i], beta[i-1], alpha[i-1](0), nu_shape, nu_rate, r);  // Marginal.
      if (!know_sig2) br.sample_sig2(sig2[i], beta[i-1], sig2_shape, sig2_scale, r);  // Sample sig2.
      br.sample_lambda(lambda[i], beta[i-1], alpha[i-1](0), tau[i](0), r);
      br.sample_beta_stable(beta[i], lambda[i], alpha[i-1](0), sig2[i](0), tau[i](0), r); 
      // You must change indexing if the following is uncommented.
      // if (!know_tau) br.sample_tau_stable(tau[i], beta[i], lambda[i], 4.0, 4.0, r);
      if (!know_alpha) br.sample_alpha_marg(alpha[i], alpha[i-1], beta[i], tau[i](0), r, alpha_b, alpha_b);
      #ifdef USE_R
      if (i % 10 == 0) R_CheckUserInterrupt();
      #endif
    }
   
  }
  catch (std::exception& e) {
    Rprintf("Error: %s\n", e.what());
    Rprintf("Aborting Gibbs sampler.\n");
  }

  end = clock();
  
  total_time = (double)(end - start) / CLOCKS_PER_SEC;
  Rprintf("Sampling complete: %g sec. for %i iterations.\n", total_time, M);

  return (double)(end - start) / CLOCKS_PER_SEC;

} // bridge_regression

////////////////////////////////////////////////////////////////////////////////
				// ORTHOGONAL //
////////////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------
double bridge_regression_ortho(MatrixFrame & beta,
			       MatrixFrame & u,
			       MatrixFrame & omega,
			       MatrixFrame & shape,
			       MatrixFrame & sig2,
			       MatrixFrame & tau,
			       MatrixFrame & alpha,
			       const MatrixFrame & y,
			       const MatrixFrame & X,
			       double sig2_shape,
			       double sig2_scale,
			       double nu_shape,
			       double nu_rate,
			       double alpha_a,
			       double alpha_b,
			       double true_sig2, // 
			       double true_tau , // Stored in tau  if true.
			       double true_alpha, 
			       uint burn)
{
  BridgeRegression br(X, y);

  // uint P = X.cols();
  // uint N = X.rows();
  uint M = beta.mats();
  bool know_sig2 = true_sig2 > 0;
  bool know_tau  = true_tau  > 0;
  bool know_alpha = true_alpha > 0;

  // Details.
  Rprintf("Bridge Regression (mix. of triangles):");
  if (know_alpha) Rprintf(" known alpha=%g", true_alpha);
  if (know_sig2) Rprintf(", sig2=%g", true_sig2);
  if (know_tau ) Rprintf(", tau=%g", true_tau);
  Rprintf("\nAssuming orthogonal design matrix!");
  Rprintf("\nBurn-in: %i, Num. Samples: %i\n", burn, M);

  // Initialize.
  Matrix ls;
  br.least_squares(ls);
  beta[0].copy(ls);

  u[0].fill(0.5);
  alpha[0](0) = 0.5;

  if ( know_sig2 ) sig2.fill(true_sig2);
  if ( know_tau  ) tau.fill(true_tau);
  if ( know_alpha) alpha.fill(true_alpha);

  RNG r;

  // Keep track of time.
  clock_t start, end;
  double total_time;

  start = clock();

  try {

    if (!know_sig2) br.sample_sig2(sig2[0], beta[0], sig2_shape, sig2_scale, r);  // Sample sig2.
    if (!know_tau ) br.sample_tau_marg(tau[0], beta[0], alpha[0](0), nu_shape, nu_rate, r);  // Sample tau.

    // Burn-In.
    for(uint i = 0; i < burn; i++){
      if (!know_tau) br.sample_tau_marg(tau[0], beta[0], alpha[0](0), nu_shape, nu_rate, r);  // Sample tau.
      br.sample_omega(omega[0], shape[0], beta[0], u[0], tau[0](0), alpha[0](0), r);
      // br.sample_omega(omega[0], beta[0], u[0], tau[0](0), alpha[0](0), r);
      br.sample_u(u[0], beta[0], omega[0], tau[0](0), alpha[0](0), r);
      if (!know_sig2) br.sample_sig2(sig2[0], beta[0], sig2_shape, sig2_scale, r);  // Sample sig2.
      br.sample_beta_ortho(beta[0], beta[0], u[0], omega[0], sig2[0](0), tau[0](0), alpha[0](0), r);
      if (!know_alpha) br.sample_alpha_marg(alpha[0], alpha[0], beta[0], tau[0](0), r, alpha_a, alpha_b);
      #ifdef USE_R
      if (i % 10 == 0) R_CheckUserInterrupt();
      #endif
    }

    end = clock();

    total_time = (double)(end - start) / CLOCKS_PER_SEC;
    Rprintf("Burn-in complete: %g sec. for %i iterations.\n", total_time, burn);
    Rprintf("Expect approx. %g sec. for %i samples.\n", total_time * M / burn, M);

    start = clock();

    // MCMC
    for(uint i = 1; i < M; i++){
      if (!know_tau) br.sample_tau_marg(tau[i], beta[i-1], alpha[i-1](0), nu_shape, nu_rate, r);  // Sample tau.
      br.sample_omega(omega[i], shape[i], beta[i-1], u[i-1], tau[i](0), alpha[i-1](0), r);
      // br.sample_omega(omega[i], beta[i-1], u[i-1], tau[i](0), alpha[i-1](0), r);
      br.sample_u(u[i], beta[i-1], omega[i], tau[i](0), alpha[i-1](0), r);
      if (!know_sig2) br.sample_sig2(sig2[i], beta[i-1], sig2_shape, sig2_scale, r);  // Sample sig2.
      br.sample_beta_ortho(beta[i], beta[i-1], u[i], omega[i], sig2[i](0), tau[i](0), alpha[i-1](0), r);
      if (!know_alpha) br.sample_alpha_marg(alpha[i], alpha[i-1], beta[i], tau[i](0), r, alpha_a, alpha_b);
      #ifdef USE_R
      if (i % 10 == 0) R_CheckUserInterrupt();
      #endif
    }

  }
  catch (std::exception& e) {
    Rprintf("Error: %s\n", e.what());
    Rprintf("Aborting Gibbs sampler.\n");
  }

  end = clock();

  total_time = (double)(end - start) / CLOCKS_PER_SEC;
  Rprintf("Sampling complete: %g sec. for %i iterations.\n", total_time, M);

  return (double)(end - start) / CLOCKS_PER_SEC;

} // bridge_regression

//--------------------------------------------------------------------
double bridge_regression_stable_ortho(MatrixFrame & beta,
				      MatrixFrame & lambda,
				      MatrixFrame & sig2,
				      MatrixFrame & tau,
				      MatrixFrame & alpha,
				      const MatrixFrame & y,
				      const MatrixFrame & X,
				      double sig2_shape,
				      double sig2_scale,
				      double nu_shape,
				      double nu_rate,
				      double alpha_a,
				      double alpha_b,
				      double true_sig2,
				      double true_tau,
				      double true_alpha,
				      uint burn)
{
  BridgeRegression br(X, y);

  // uint P = X.cols();
  // uint N = X.rows();
  uint M = beta.mats();
  bool know_sig2 = true_sig2 > 0;
  bool know_tau  = true_tau  > 0;
  bool know_alpha = true_alpha > 0;

  // Details.
  Rprintf("Bridge Regression (mix. of normals):");
  if (know_alpha) Rprintf(" known alpha=%g", true_alpha);
  if (know_sig2) Rprintf(", sig2=%g", true_sig2);
  if (know_tau ) Rprintf(", tau=%g", true_tau);
  Rprintf("\nAssuming orthogonal design matrix!");
  Rprintf("\nBurn-in: %i, Num. Samples: %i\n", burn, M);

  // Initialize.
  Matrix ls;
  br.least_squares(ls);
  beta[0].copy(ls);

  alpha[0](0) = 0.5;

  if ( know_sig2 ) sig2.fill(true_sig2);
  if ( know_tau  ) tau.fill(true_tau);
  if ( know_alpha) alpha.fill(true_alpha);

  RNG r;

  // Keep track of time.
  clock_t start, end;
  double total_time;

  start = clock();

  try{

    // br.sample_sig2(sig2[0], beta[0], sig2_shape, sig2_scale, r);  // Sample sig2.

    // Burn-In.
    for(uint i = 0; i < burn+1; i++){
      if (!know_tau) br.sample_tau_marg(tau[0], beta[0], alpha[0](0), nu_shape, nu_rate, r);  // Sample tau.
      br.sample_lambda(lambda[0], beta[0], alpha[0](0), tau[0](0), r);
      if (!know_sig2) br.sample_sig2(sig2[0], beta[0], sig2_shape, sig2_scale, r);  // Sample sig2.
      br.sample_beta_stable_ortho(beta[0], lambda[0], alpha[0](0), sig2[0](0), tau[0](0), r);
      if (!know_alpha) br.sample_alpha_marg(alpha[0], alpha[0], beta[0], tau[0](0), r, alpha_a, alpha_b);
      #ifdef USE_R
      if (i % 10 == 0) R_CheckUserInterrupt();
      #endif
    }

    end = clock();

    total_time = (double)(end - start) / CLOCKS_PER_SEC;
    Rprintf("Burn-in complete: %g sec. for %i iterations.\n", total_time, burn);
    Rprintf("Expect approx. %g sec. for %i samples.\n", total_time * M / burn, M);

    start = clock();

    // MCMC
    for(uint i = 1; i < M; i++){
      if (!know_tau) br.sample_tau_marg(tau[i], beta[i-1], alpha[i-1](0), nu_shape, nu_rate, r);  // Sample tau.
      br.sample_lambda(lambda[i], beta[i-1], alpha[i-1](0), tau[i](0), r);
      if (!know_sig2) br.sample_sig2(sig2[i], beta[i-1], sig2_shape, sig2_scale, r);  // Sample sig2.
      br.sample_beta_stable_ortho(beta[i], lambda[i], alpha[i-1](0), sig2[i](0), tau[i](0), r);
      if (!know_alpha) br.sample_alpha_marg(alpha[i], alpha[i-1], beta[i], tau[i](0), r, alpha_a, alpha_b);
      #ifdef USE_R
      if (i % 10 == 0) R_CheckUserInterrupt();
      #endif
    }

  }
  catch (std::exception& e) {
    Rprintf("Error: %s\n", e.what());
    Rprintf("Aborting Gibbs sampler.\n");
  }

  end = clock();

  total_time = (double)(end - start) / CLOCKS_PER_SEC;
  Rprintf("Sampling complete: %g sec. for %i iterations.\n", total_time, M);

  return (double)(end - start) / CLOCKS_PER_SEC;

} // bridge_regression


//////////////////////////////////////////////////////////////////////
				 // WRAPPERS //
//////////////////////////////////////////////////////////////////////

void bridge_EM(double *betap,
	       const double *yp,
	       const double *Xp,
	       const double *ratio,
	       const double *alpha,
	       const int *P,
	       const int *N,
	       const double *lambda_max,
	       const double *tol,
	             int *max_iter,
	       const bool *use_cg)
{
  Matrix beta(*P, 1);

  Matrix y(yp, *N, 1);
  Matrix X(Xp, *N, *P);

  if (*use_cg) Rprintf("Using conjugate gradient method.\n");

  int total_iter = EM(beta, y, X, *ratio, *alpha, *lambda_max, *tol, *max_iter, *use_cg);
  *max_iter = total_iter;

  MatrixFrame beta_mf(betap, *P);
  beta_mf.copy(beta);
}

//------------------------------------------------------------------------------

void bridge_regression(double *betap,
		       double *up,
		       double *omegap,
		       double *shapep,
		       double *sig2p,
		       double *taup,
		       double *alphap,
		       const double *yp,
		       const double *Xp,
		       const double *sig2_shape,
		       const double *sig2_scale,
		       const double *nu_shape,
		       const double *nu_rate,
		       const double *alpha_a,
		       const double *alpha_b,
		       const double *true_sig2,
		       const double *true_tau,
		       const double *true_alpha,
		       const int *P,
		       const int *N,
		       const int *M,
		       const int *burn,
		       double *runtime,		 
		       const bool *ortho,
		       const int *betaburn,
		       const bool *use_hmc)
{
  Matrix y(yp, *N,  1);
  Matrix X(Xp, *N, *P);

  Matrix beta(*P, 1, *M);
  Matrix u    (*P, 1, *M, 0.0);
  Matrix omega(*P, 1, *M, 1.0);
  Matrix shape(*P, 1, *M, 0.0);
  Matrix sig2( 1, 1, *M);
  Matrix tau ( 1, 1, *M);
  Matrix alpha(1, 1, *M);

  #ifdef USE_R
  GetRNGstate();
  #endif

  if (!*ortho) {


    *runtime = bridge_regression(beta, u, omega, shape, sig2, tau, alpha, y, X,
				 *sig2_shape, *sig2_scale,
				 *nu_shape, *nu_rate,
				 *alpha_a, *alpha_b,
				 *true_sig2, *true_tau, *true_alpha,
				 *burn, *betaburn, *use_hmc);
  }
  else {
    
    *runtime = bridge_regression_ortho(beta, u, omega, shape, sig2, tau, alpha,
				       y, X,
				       *sig2_shape, *sig2_scale,
				       *nu_shape, *nu_rate,
				       *alpha_a, *alpha_b,
				       *true_sig2, *true_tau, *true_alpha,
				       *burn);

  }

  #ifdef USE_R
  PutRNGstate();
  #endif

  MatrixFrame beta_mf (betap, *P, 1 , *M);
  MatrixFrame u_mf    (up    , *P, 1, *M);
  MatrixFrame omega_mf(omegap, *P, 1, *M);
  MatrixFrame shape_mf(shapep, *P, 1, *M);
  MatrixFrame sig2_mf (sig2p, 1 , 1 , *M);
  MatrixFrame tau_mf  (taup , 1 , 1 , *M);
  MatrixFrame alpha_mf(alphap, 1, 1,  *M);

  beta_mf.copy(beta);
  u_mf.copy(u);
  omega_mf.copy(omega);
  shape_mf.copy(shape);
  sig2_mf.copy(sig2);
  tau_mf.copy (tau );
  alpha_mf.copy(alpha);

} // bridge_regression

//--------------------------------------------------------------------
void bridge_reg_stable(double *betap,
		       double *lambdap,
		       double *sig2p,
		       double *taup,
		       double *alphap,
		       const double *yp,
		       const double *Xp,
		       const double *sig2_shape,
		       const double *sig2_scale,
		       const double *nu_shape,
		       const double *nu_rate,
		       const double *alpha_a,
		       const double *alpha_b,
		       const double *true_sig2,
		       const double *true_tau,
		       const double *true_alpha,
		       const int *P,
		       const int *N,
		       const int *M,
		       const int *burn,
		       double *runtime,
		       const bool *ortho)
{
  Matrix y(yp, *N,  1);
  Matrix X(Xp, *N, *P);

  Matrix beta  (*P, 1, *M);
  Matrix lambda(*P, 1, *M, 1.0);
  Matrix sig2  ( 1, 1, *M);
  Matrix tau   ( 1, 1, *M);
  Matrix alpha ( 1, 1, *M);

  #ifdef USE_R
  GetRNGstate();
  #endif

  if (!*ortho) {
    *runtime = bridge_regression_stable(beta, lambda, sig2, tau, alpha,
					y, X,
					*sig2_shape, *sig2_scale,
					*nu_shape, *nu_rate,
					*alpha_a, *alpha_b,
					*true_sig2, *true_tau, *true_alpha,
					(uint)*burn);
  }
  else {
    *runtime = bridge_regression_stable_ortho(beta, lambda, sig2, tau, alpha,
					      y, X,
					      *sig2_shape, *sig2_scale,
					      *nu_shape, *nu_rate,
					      *alpha_a, *alpha_b,
					      *true_sig2, *true_tau, *true_alpha,
					      (uint)*burn);
  }

  #ifdef USE_R
  PutRNGstate();
  #endif

  // std::copy(&beta(0), &beta(0) + beta.vol(), betap);

  MatrixFrame beta_mf (betap, *P, 1, *M);
  MatrixFrame lambda_mf(lambdap, *P, 1, *M);
  MatrixFrame sig2_mf (sig2p,  1, 1, *M);
  MatrixFrame tau_mf  (taup,   1, 1, *M);
  MatrixFrame alpha_mf(alphap, 1, 1, *M);

  beta_mf.copy(beta);
  lambda_mf.copy(lambda);
  sig2_mf.copy(sig2);
  tau_mf.copy(tau);
  alpha_mf.copy(alpha);

} // bridge_regression

////////////////////////////////////////////////////////////////////////////////
			     // TRUNCATED NORMAL //
////////////////////////////////////////////////////////////////////////////////

void rtnorm_left(double *x, double *left, double *mu, double *sig, int *num)
{
  RNG r;

  #ifdef USE_R
  GetRNGstate();
  #endif

  for(int i=0; i < *num; ++i){
    #ifdef USE_R
    if (i%1==0) R_CheckUserInterrupt();
    #endif

    x[i] = r.tnorm(left[i], mu[i], sig[i]);
  }

  #ifdef USE_R
  PutRNGstate();
  #endif
}

void rtnorm_both(double *x, double *left, double* right, double *mu, double *sig, int *num)
{
  RNG r;

  #ifdef USE_R
  GetRNGstate();
  #endif

  for(int i=0; i < *num; ++i){
    #ifdef USE_R
    if (i%1==0) R_CheckUserInterrupt();
    #endif

    x[i] = r.tnorm(left[i], right[i], mu[i], sig[i]);
  }

  #ifdef USE_R
  PutRNGstate();
  #endif
}

void rrtgamma_rate(double *x, double *scale, double *rate, double *right_t, int *num)
{
  RNG r;

  #ifdef USE_R
  GetRNGstate();
  #endif

  for(int i=0; i < *num; ++i){
    #ifdef USE_R
    if (i%1==0) R_CheckUserInterrupt();
    #endif

    x[i] = r.rtgamma_rate(scale[i], rate[i], right_t[i]);
  }

  #ifdef USE_R
  PutRNGstate();
  #endif
}

void retstable_LD(double* x, double* alpha, double* V0, double* h, int* num)
{
  RNG r;

  #ifdef USE_R
  GetRNGstate();
  #endif

  for(int i=0; i < *num; ++i){
    #ifdef USE_R
    if (i%10000==0) R_CheckUserInterrupt();
    #endif

    x[i] = retstable_LD(h[i], alpha[i], r, V0[i]);
  }

  #ifdef USE_R
  PutRNGstate();
  #endif
}

//////////////////////////////////////////////////////////////////////
// END OF CODE //
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
			    // APPENDIX //
//////////////////////////////////////////////////////////////////////

/*
  Fall 2011

  It appears that the order in which I draw u, beta, omega, and tau
  makes a difference.  I need to figure out why this is.

  When I drew thing by omega, beta, u, tau I got a different answer
  than when I drew things by u, omega, tau, beta.  James uses the
  latter.

  sig2 is independent of u and omega given beta.  So its order doesn't
  matter.

  Order clearly matters because we want to sample u immediately after
  we have sampled beta.  Otherwise we might end up with
  1-tau|beta_j|omega_j^{-1/alpha} being negative.

  I think this has something to do with how we are thinking about
  likelihood.  The posterior for tau is calculated as if we have
  integrated out the auxillery variables u and omega.  But I haven't
  sampled omega and u "jointly" if I sample tau in between.

  I think we need to sample beta, u, omega, tau.  We want u to follow
  beta and, apparently, we need u and omega to be sampled using the
  same tau.  I suppose this makes sense.  Or the same beta and tau for
  that matter.  We can't switch beta and tau here because we want to
  sample u immediately after sampling beta.  u and omega need to be
  sampled using the same tau because they were constructed using the
  same tau.

  We didn't calculate the posterior distribution of tau using (beta,
  omega, u) joint likelihood.  Presumably we could do this.  I'd need
  to write routine for some sort of truncated distribution.  I think
  it would just be \1{|beta_j| / ((1-u_j) * omega_j^{1/alpha})<=tau}
  p(tau).
 */

