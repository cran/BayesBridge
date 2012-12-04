% package: BayesBridge
\name{trace.beta}
\alias{trace.beta}

\title{Trace Plot}

\description{
  Trace plot using expectation maximization.
}

\usage{

trace.beta(y, X, alpha=0.5, ratio.grid=exp(seq(-20,20,0.1)),
           tol=1e-9, max.iter=30, use.cg=FALSE, plot.it=FALSE)
}

\arguments{
  \item{y}{An N dimensional vector of data.}
  \item{X}{An N x P dimensional design matrix.}
  \item{alpha}{A parameter.}
  \item{ratio.grid}{A grid of ratio=tau/sigma.}
  \item{tol}{The threshold at which the algorithm terminates.}
  \item{max.iter}{The maximum number of iterations to use.}
  \item{use.cg}{Use the conjugate gradient method.}
  \item{plot.it}{Plot it.}
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
out = trace.beta(y, X);
}

\seealso{
\code{\link{bridge.reg}, \link{bridge.EM}}.
}

\references{
  Nicolas G. Poslon, James G. Scott, and Jesse Windle.  The Bayesian Bridge.
  \url{http://arxiv.org/abs/1109.2279}.
  }
\keyword{bridge}
\keyword{EM}
\keyword{expectation maximization}