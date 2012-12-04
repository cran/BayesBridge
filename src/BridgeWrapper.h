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
 Herein we implement Gibbs sampling for Bridge Regression a la Polson,
 Scott and Windle.  For a detailed description of the specifics of the
 setup and algorithm please see their paper The Bayesian Bridge
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

 See BridgeRegression.h for the conditional posteriors used when Gibbs
 Sampling.

 R WRAPPER:

 We also provide functions that may be called from R so that one can
 speed up their Gibbs sampling or prevent copying large matrices when
 doing an expectation maximization.

 In what follows:

   - y is N x 1.
   - X is N x P.
   - beta is P x 1.

 */
//////////////////////////////////////////////////////////////////////

#ifndef __BRIDGE__
#define __BRIDGE__

#include "Matrix.h"
#include "RNG.h"
#include "BridgeRegression.h"

//////////////////////////////////////////////////////////////////////
		    // EXPECTATION MAXIMIZATION //
//////////////////////////////////////////////////////////////////////

int EM(Matrix & beta, MatrixFrame &y, MatrixFrame &X,
	double ratio, double alpha, double lambda_max,
	double tol, int max_iter, bool use_cg=false);

//////////////////////////////////////////////////////////////////////
		       // BRIDGE REGRESSION //
//////////////////////////////////////////////////////////////////////

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
			 double true_sig2,  
			 double true_tau , 
			 double true_alpha,
			 uint burn,
			 int betaburn=0,
			 bool use_hmc=false);

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
				uint burn);

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
			       double true_sig2,  
			       double true_tau , 
			       double true_alpha,
			       uint burn);

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
				      uint burn);

//////////////////////////////////////////////////////////////////////
			    // WRAPPERS //
//////////////////////////////////////////////////////////////////////


extern "C"
{
  void bridge_EM(double *beta,
		 const double *y,
		 const double *X,
		 const double *ratio,
		 const double *alpha,
		 const int *P,
		 const int *N,
		 const double *lambda_max,
		 const double *tol,
		       int *max_iter,
		 const bool *use_cg);

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
			 const bool *use_hmc);

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
			 const bool *ortho);

  void rtnorm_left(double *x, double *left, double *mu, double *sig, int *num);

  void rtnorm_both(double *x, double *left, double* right, double *mu, double *sig, int *num);

  void rrtgamma_rate(double *x, double *scale, double *rate, double *right_t, int *num);

}

#endif

//////////////////////////////////////////////////////////////////////
			  // END OF CODE //
//////////////////////////////////////////////////////////////////////


