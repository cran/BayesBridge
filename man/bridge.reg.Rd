% package: BayesBridge
\name{bridge.reg}

\alias{bridge.reg}
\alias{bridge.reg.tri}
\alias{bridge.reg.stb}

\title{Bayesian Bridge Regression}

\description{
  Bayesian Bridge Regression via Gibbs sampling.
}

\details{

  Bridge regression is a regularized regression in which the
  regression coefficient's prior is an exponential power distribution.
  Specifically, inference on the regression coefficient beta is made
  using the model

  \deqn{y = X \beta + \epsilon, \epsilon \sim N(0, \sigma^2 \; I),}
  \deqn{p(\beta) \propto \exp(\sum_j -(|\beta_j|/\tau)^{\alpha}).}

  The variance sig2, the scale parameter tau, and alpha may be
  estimated.  This routine uses an inverse-gamma prior for sig2, a
  gamma prior for \eqn{\nu=\tau^{-\alpha}}, and a beta prior for
  alpha, when estimating those parameters.  That is

  \deqn{\sigma^2 \sim IG(\textmd{sig2.shape}, \textmd{sig2.scale}),}
  \deqn{\nu \sim  Ga(\textmd{nu.shape}, \textmd{rate}=\textmd{nu.rate}),}
  \deqn{\alpha \sim Be(\textmd{alpha.a}, \textmd{alpha.b}).}

  The parameters sig2, tau, or alpha are taken to be known when
  \code{sig2.true} > 0, \code{tau.true} > 0, or \code{alpha} > 0
  respectively.

  When the design matrix is orthogonal set \code{ortho=TRUE}.

}

\usage{
bridge.reg(y, X, nsamp, alpha=0.5,
           sig2.shape=0.0, sig2.scale=0.0, nu.shape=2.0, nu.rate=2.0,
           alpha.a=1.0, alpha.b=1.0,
           sig2.true=0.0, tau.true=0.0,
           burn=500, method="triangle", ortho=FALSE)

bridge.reg.tri(y, X, nsamp, alpha=0.5,
           sig2.shape=0.0, sig2.scale=0.0, nu.shape=2.0, nu.rate=2.0,
           alpha.a=1.0, alpha.b=1.0,	   
           sig2.true=0.0, tau.true=0.0,
           burn=500, ortho=FALSE, betaburn=0)

bridge.reg.stb(y, X, nsamp, alpha=0.5,
           sig2.shape=0.0, sig2.scale=0.0, nu.shape=2.0, nu.rate=2.0,
           alpha.a=1.0, alpha.b=1.0, 
           sig2.true=0.0, tau.true=0.0,
           burn=500, ortho=FALSE)
}

\arguments{
  \item{y}{An N dimensional vector of data.}
  \item{X}{An N x P dimensional design matrix.}
  \item{nsamp}{The number of MCMC samples saved.}
  \item{alpha}{The exponential power parameter; set <= 0 to estimate.}
  \item{sig2.shape}{Shape parameter for sig2's prior.}
  \item{sig2.scale}{Scale parameter for sig2's prior.}
  \item{nu.shape}{Shape parameter for nu's prior.}
  \item{nu.rate}{Rate parameter for nu's prior.}
  \item{alpha.a}{First shape parameter for alpha's prior.}
  \item{alpha.b}{Second shape parameter for alpha's prior.}
  \item{sig2.true}{The variance when it is known.}
  \item{tau.true}{The scale parameter tau when it is known.}
  \item{burn}{The number of MCMC samples discarded.}
  \item{method}{The method to use.  Either "triangle" or "stable".}
  \item{ortho}{When the design matrix is orthogonal set to TRUE.}
  \item{betaburn}{Number of burn iterations when sampling beta.}
}

\examples{
# Load the diabetes data...
data(diabetes, package="BayesBridge");
cov.name = colnames(diabetes$x);
y = diabetes$y;
X = diabetes$x;

# Center the data.
y = y - mean(y);
mX = colMeans(X);
for(i in 1:442){
    X[i,] = X[i,] - mX;
}

# Run the bridge regression when sig2 and tau are unknown.
gb = bridge.reg(y, X, nsamp=10000, alpha=0.5, 
     sig2.shape=0.0, sig2.scale=0.0, nu.shape=2.0, nu.rate=2.0);
}

\seealso{
\code{\link{bridge.EM}}.
}

\references{
  Nicholas G. Poslon, James G. Scott, and Jesse Windle.  The Bayesian Bridge.
  \url{http://arxiv.org/abs/1109.2279}.
}

\keyword{bridge}
\keyword{regression}

% Citation:
% Adapted from <http://cran.r-project.org/doc/manuals/R-exts.html>.

% Notes:
% To check this documentation use R CMD Rdconv -t txt <filename>.
