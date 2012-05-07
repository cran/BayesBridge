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

//------------------------------------------------------------------------------

#endif
