% package: BayesBridge
\name{rtnorm}
\alias{rtnorm}

\title{Truncated Normal Random Variates}

\description{
  Sample from a truncated normal distribution.
}

\details{

  This routine follows Robert (1995) to sample a truncated normal.  

  For mu, sig, left, and right you may send sequences of parameters in addition
  to scalars.
 
}

\usage{

rtnorm(num=1, mu=0.0, sig=1.0, left=-Inf, right=Inf)

}

\arguments{
  \item{num}{The number of random variates to generate.}
  \item{mu}{The mean.}
  \item{sig}{The standard deviation.}
  \item{left}{The left truncation point.}
  \item{right}{The right truncation point.}
}

\examples{

  samp = rtnorm(100, c(-1.0, 1.0), c(0.5, 0.2), -4, 4);

}

% \seealso{
% \code{\link{bridge.reg}}.
% }

\references{

  Christian Robert.  Simulation of truncated normal random variables.
  Statistics and Computing.  1995.  Vol 5, p. 121-125.

  }
\keyword{truncated}
\keyword{normal}
\keyword{gaussian}
