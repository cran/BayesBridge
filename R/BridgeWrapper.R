######################################################################

## Copyright 2012 Nicholas G. Polson, James G. Scott, Jesse Windle
## Contact info: <jwindle@ices.utexas.edu>.

## This file is part of BayesBridge.

## BayesBridge is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
  
## BayesBridge is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
  
## You should have received a copy of the GNU General Public License
## along with BayesBridge.  If not, see <http:##www.gnu.org/licenses/>.
			      
######################################################################

## Load C++ code.
## dyn.load("libbridge.so");
## dyn.load("Bridge.so");
## dyn.load("Bridge.so", PACKAGE=BayesBridge);
  
################################################################################
                             ## HELPER FUNCTIONS ##
################################################################################

# Check if param >= val.  "name" is the name of the param.
is.above <- function(param, val, name){
    above = TRUE;
    if (param < val){
        alert = paste("Error: ", name, "<", val, sep="");
        print(alert);
        above = FALSE;
    }
    # While we are at it, check that we are working with a number.
    if (!is.numeric(param)) {
        alert = paste("Error:", name, "is not numeric.");
        print(alert);
        above = FALSE;
    }
    above;
}

# Check that the parameters are valid.
check.parameters <- function(N, R, M, sig2.shape, sig2.scale, nu.shape, nu.rate, alpha.a, alpha.b){
    ok = TRUE;
    if (N!=R)    { print("Error: y and X do not conform."); ok=FALSE; }
    ok = ok *
         is.above(M         , 1, "niter") *
         is.above(sig2.shape, 0, "sig2.shape") *
         is.above(sig2.scale, 0, "sig2.scale") *
         is.above(nu.shape  , 0, "nu.shape") *
         is.above(nu.rate   , 0, "nu.rate") *
         is.above(alpha.a   , -1, "alpha.a") *
         is.above(alpha.b   , -1, "alpha.b");
    

    ## if (M > 1000){
    ##     ans = readline(paste("niter =", M, "> 1000.  Do you really want to proceed? [n] "));
    ##     ans = substr(ans, 1, 1);
    ##     ok = ok * (ans=="y" || ans=="Y");
    ## }

    ok
}

# Check the extra parameters in expectation maximization.
check.EM <- function(lambda.max, tol, max.iter)
{
    ok = TRUE;
    ok = ok *
        is.above(lambda.max, 0.0, "lambda.max") *
        is.above(tol       , 0.0, "tolerance") *
        is.above(max.iter  , 1.0, "max.iter");

    ok
}

################################################################################
                         ## EXPECTATION MAXIMIZATION ##
################################################################################

bridge.EM <- function(y, X,
                      alpha=0.5,
                      ratio=1.0,
                      lambda.max=1e9*ratio, tol=1e-9, max.iter=30,
                      use.cg=FALSE, ret.solves=FALSE)
{
    N = length(y);
    R = dim(X)[1];
    P = dim(X)[2];

    sig2 = 1.0;
    tau = ratio;

    if (ratio < 0) {
      print("bridge.EM: ratio < 0")
      return (0)
    }
    if (alpha < 0) {
      print("bridge.EM: alpha < 0")
      return(0)
    }

    ok = check.parameters(N, R, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0) *
         check.EM(lambda.max, tol, max.iter);

    if (!ok) { break; }

    beta = array(0, dim=c(P));

    OUT = .C("bridge_EM",
             beta,
             as.double(y), as.double(X),
             ratio, alpha,
             as.integer(P), as.integer(N),
             lambda.max, tol, as.integer(max.iter),
             as.integer(use.cg),
             PACKAGE="BayesBridge");

    output = OUT[[1]]; # beta
    rownames(output) = colnames(X);

    if(ret.solves) {
        output = list("beta"=output, "num.solves"=OUT[[10]]);
    }

    output
}

################################################################################
                  ## BRIDGE REGRESSION MIXTURE OF TRIANGLES ##
################################################################################

bridge.reg.tri <- function(y, X,
                           nsamp,
                           alpha=0.5,
                           sig2.shape=0.0, sig2.scale=0.0,
                           nu.shape=2.0, nu.rate=2.0,
                           alpha.a=1.0, alpha.b=1.0,
                           sig2.true=0.0, tau.true=0.0,
                           burn=500, ortho=FALSE, betaburn=0)
{
    N = length(y);
    R = dim(X)[1];
    P = dim(X)[2];
    M = nsamp;
    rtime = 0;

    ok = check.parameters(N, R, M, sig2.shape, sig2.scale, nu.shape, nu.rate, alpha.a, alpha.b);
    if (!ok) { break; }

    alpha.true = alpha;
    
    beta  = array(0, dim=c(P, M));
    u     = array(0, dim=c(P, M));
    omega = array(0, dim=c(P, M));
    shape = array(0, dim=c(P, M));
    sig2  = array(0, dim=c(M));
    tau   = array(0, dim=c(M));
    alpha = array(0, dim=c(M));

    use.hmc = FALSE;
    
    OUT <- .C("bridge_regression",
              beta, u, omega, shape, sig2, tau, alpha,
              as.double(y), as.double(X),
              sig2.shape, sig2.scale,
              nu.shape, nu.rate,
              alpha.a, alpha.b,
              sig2.true, tau.true, alpha.true,
              as.integer(P), as.integer(N), as.integer(M),
              as.integer(burn), rtime, as.integer(ortho), as.integer(betaburn), as.integer(use.hmc),
              PACKAGE="BayesBridge");

    output <- list("beta"=t(OUT[[1]]), "u"=t(OUT[[2]]), "w"=t(OUT[[3]]), "shape"=t(OUT[[4]]),
                   "sig2"=OUT[[5]], "tau"=OUT[[6]], "alpha"=OUT[[7]],
                   "runtime"=OUT[[23]]);

    colnames(output$beta) = colnames(X);

    output
}

################################################################################
               ## BRIDGE REGRESSION MIXTURE OF NORMAL KERNELS ##
################################################################################

bridge.reg.stb <- function(y, X,
                           nsamp,
                           alpha=0.5,
                           sig2.shape=0.0, sig2.scale=0.0,
                           nu.shape=2.0, nu.rate=2.0,
                           alpha.a=1.0, alpha.b=1.0,
                           sig2.true=0.0, tau.true=0.0,
                           burn=500, ortho=FALSE)
{
    N = length(y);
    R = dim(X)[1];
    P = dim(X)[2];
    M = nsamp;
    rt = 0;

    alpha.true = alpha;
    
    ok = check.parameters(N, R, M, sig2.shape, sig2.scale, nu.shape, nu.rate, alpha.a, alpha.b);
    if (!ok) { break; }

    beta   = array(0, dim=c(P, M));
    lambda = array(0, dim=c(P, M));
    sig2  = array(0, dim=c(M));
    tau   = array(0, dim=c(M));
    alpha = array(0, dim=c(M));

    OUT <- .C("bridge_reg_stable",
              beta, lambda, sig2, tau, alpha,
              as.double(y), as.double(X),
              sig2.shape, sig2.scale,
              nu.shape, nu.rate,
              alpha.a, alpha.b,
              sig2.true, tau.true, alpha.true,
              as.integer(P), as.integer(N), as.integer(M), as.integer(burn), rt, as.integer(ortho),
              PACKAGE="BayesBridge");

    output = list("beta"=t(OUT[[1]]), "lambda"=t(OUT[[2]]), "sig2"=OUT[[3]], "tau"=OUT[[4]], "alpha"=OUT[[5]], "runtime"=OUT[[21]])
    colnames(output$beta) = colnames(X);

    output
}

################################################################################
                               ## WRAP TO BOTH ##
################################################################################

bridge.reg <- function(y, X,
                       nsamp,
                       alpha=0.5,
                       sig2.shape=0.0, sig2.scale=0.0,
                       nu.shape=2.0, nu.rate=2.0,
                       alpha.a=1.0, alpha.b=1.0,
                       sig2.true=0.0, tau.true=0.0,
                       burn=500, method="triangle", ortho=FALSE)
{
  out = NULL
  
  if (method=="triangle") {
    out = bridge.reg.tri(y, X,
      nsamp,
      alpha=0.5,
      sig2.shape=0.0, sig2.scale=0.0,
      nu.shape=0.5, nu.rate=0.5,
      alpha.a=1.0, alpha.b=1.0,
      sig2.true=0.0, tau.true=0.0,
      burn=500, ortho=ortho)
  }
  else if (method=="stable") {
    out = bridge.reg.stb(y, X,
      nsamp,
      alpha=0.5,
      sig2.shape=0.0, sig2.scale=0.0,
      nu.shape=0.5, nu.rate=0.5,
      alpha.a=1.0, alpha.b=1.0,
      sig2.true=0.0, tau.true=0.0,
      burn=500, ortho=ortho)
  }
  else {
    print("Unrecognized method.  Use \"triangles\" or \"stable\".");
  }

  out
}

################################################################################
                             ## TRUNCATED NORMAL ##
################################################################################

## Draw truncated normal
##------------------------------------------------------------------------------
rtnorm.left <- function(num=1, left=0.0, mu=0.0, sig=1.0)
{
    ## Check Parameters.
    if (sum(sig<=0)!=0) {
        print("sig must be greater than zero.");
        return(NA);
    }
    if (! (num>0) ) {
      print("num must be greater than zero.");
      return(NA);
    }

    x = rep(0, num);
    
    if (length(mu)  != num) { mu  = array(mu,  num); }
    if (length(sig) != num) { sig = array(sig, num); }

    if (length(left)  != num) { left  = array(left,  num); }
    
    OUT = .C("rtnorm_left", x, left, mu, sig, as.integer(num), PACKAGE="BayesBridge");

    OUT[[1]]
}

rtnorm.both <- function(num=1, left=-1.0, right=1.0, mu=0.0, sig=1.0)
{
  LGER = left>=right;
    ## Check Parameters.
    if (sum(sig<=0)!=0) {
        print("sig must be greater than zero.");
        return(NA);
    }
    if (sum(LGER)!=0) {
      print("rtnorm: left must be less than right.");
      return(NA);
    }
    if (! (num>0) ) {
      print("rtnorm: num must be greater than zero.");
      return(NA);
    }

    x = rep(0, num);

    if (length(mu)  != num) { mu  = array(mu,  num); }
    if (length(sig) != num) { sig = array(sig, num); }

    if (length(left)  != num) { left  = array(left,  num); }
    if (length(right) != num) { right = array(right, num); }

    OUT = .C("rtnorm_both", x, left, right, mu, sig, as.integer(num), PACKAGE="BayesBridge");

    OUT[[1]]
}

rtnorm <- function(num=1, mu=0.0, sig=1.0, left=-Inf, right=Inf)
{
  LGER = left>=right;
  ## Check Parameters.
  if (sum(sig<=0)!=0) {
    print("sig must be greater than zero.");
    return(NA);
  }
  if (sum(LGER)!=0) {
    print("left must be less than right.");
    cat("left:", left[LGER], "\n");
    cat("rght:", right[LGER], "\n");
    return(NA);
  }

  if (length(mu)  != num) { mu  = array(mu,  num); }
  if (length(sig) != num) { sig = array(sig, num); }
  
  if (length(left)  != num) { left  = array(left,  num); }
  if (length(right) != num) { right = array(right, num); }
  
  x = rep(0, num);
  
  u  = (left ==-Inf) & (right == Inf);
  l  = (left !=-Inf) & (right == Inf);
  r  = (left ==-Inf) & (right != Inf);
  b  = (left !=-Inf) & (right != Inf);

  n.u = sum(u);
  n.l = sum(l);
  n.r = sum(r);
  n.b = sum(b);

  if (n.b > 0) {
    x[b] = .C("rtnorm_both", x[b], left[b], right[b], mu[b], sig[b], as.integer(n.b), PACKAGE="BayesBridge")[[1]];
  }
  if (n.r > 0) {
    x[r] = -1*.C("rtnorm_left", x[r], -1*right[r], -1*mu[r], sig[r], as.integer(n.r), PACKAGE="BayesBridge")[[1]];
  }
  if (n.l > 0) {
    x[l] = .C("rtnorm_left", x[l], left[l], mu[l], sig[l], as.integer(n.l), PACKAGE="BayesBridge")[[1]]
  }
  if (n.u > 0) {
    x[u] = rnorm(n.u, mu[u], sig[u]);
  }

  x
}

rrtgamma <- function(num=1, shape=1.0, rate=1.0, rtrunc=1.0, scale=1.0/rate)
{
  rate = 1.0 / scale;
  
  ## Check Parameters.
  if (!all(shape>0)) {
    print("shape must be greater than zero.");
    return(NA);
  }
  if (!all(rate>0)) {
    print("scale/rate must be greater than zero.");
    return(NA);
  }
  if (!all(rtrunc>0)) {
    print("rtrunc must be greater than zero.");
    return(NA);
  }

  shape = array(shape, num);
  rate  = array(rate , num);
  rtrunc = array(rtrunc, num);
  
  x = rep(0, num);

  out = .C("rrtgamma_rate", x, shape, rate, rtrunc, as.integer(num), PACKAGE="BayesBridge");

  out[[1]]
}

retstable.ld <- function(num=1, alpha=1, V0=1, h=1)
{
  if (!all(V0>0)) {
    print("V0 must be > 0.");
    return(NA);
  }

  if (!all(h>=0)) {
    print("h must be >= 0");
    return(NA);
  }

  if (!all(alpha>0) || !all(alpha<=1)) {
    print("alpha must be in (0,1].");
    return(NA);
  }

  alpha = array(alpha, num);
  h     = array(h    , num);
  V0    = array(V0   , num);

  x = rep(0, num)

  out = .C("retstable_LD", x, alpha, V0, h, as.integer(num), PACKAGE="BayesBridge");

  out[[1]]
}
