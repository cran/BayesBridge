% package: BayesBridge
\name{bridge.EM}
\alias{bridge.EM}
\alias{bridge.EM.R}

\title{Bridge Regression - Expectation Maximization}

\description{
  Expectation Maximization for bridge regression.
}

\details{

  Bridge regression is a regularized regression in which the
  regression coefficient's prior is an exponential power distribution.
  Specifically, inference on the regression coefficient beta is made
  using the model

  \deqn{y = X \beta + \epsilon, \epsilon \sim N(0, \sigma^2 \; I),}
  \deqn{p(\beta) \propto \exp(\sum_j -(|\beta_j|/\tau)^{\alpha}).}

  This procedure calculates the posterior mode of beta, given
  ratio=\eqn{\tau/\sigma} and alpha using the expectation maximization
  algorithm.
  
}

\usage{

bridge.EM  (y, X, alpha=0.5, ratio=1.0, lambda.max=1e9*ratio, 
  tol=1e-9, max.iter=30, use.cg=FALSE, ret.solves=FALSE)

bridge.EM.R(y, X, alpha, ratio=1.0, 
  lambda.max=1e9*ratio, tol=1e-9, max.iter=30, init=NULL)

}

\arguments{
  \item{y}{An N dimensional vector of data.}
  \item{X}{An N x P dimensional design matrix.}
  \item{ratio}{The ratio tau/sigma.}
  \item{alpha}{A parameter.}
  \item{lambda.max}{A cut-off used to determine when a variable
    vanishes.}
  \item{tol}{The threshold at which the algorithm terminates.}
  \item{max.iter}{The maximum number of iterations to use.}
  \item{use.cg}{Use the conjugate gradient method.}
  \item{ret.solves}{Return the number of times a linear system is solved.}
  \item{init}{Initial value.}
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

# Expectation maximization.
bridge.EM(y, X, 0.5, 1.0, 1e8, 1e-9, 30, use.cg=TRUE);
}

\seealso{
\code{\link{bridge.reg}}.
}

\references{
  Nicolas G. Poslon, James G. Scott, and Jesse Windle.  The Bayesian Bridge.
  \url{http://arxiv.org/abs/1109.2279}.
  }
\keyword{bridge}
\keyword{EM}
\keyword{expectation maximization}