% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SmCCNet-source.R
\name{getRobustWeightsSingle}
\alias{getRobustWeightsSingle}
\title{Single-omics SmCCA with Quantitative Phenotype}
\usage{
getRobustWeightsSingle(
  X1,
  Trait,
  Lambda1,
  s1 = 0.7,
  SubsamplingNum = 1000,
  trace = FALSE
)
}
\arguments{
\item{X1}{An \eqn{n\times p_1} data matrix (e.g. mRNA) with \eqn{p_1}
features and \eqn{n} subjects.}

\item{Trait}{An \eqn{n\times 1} trait (phenotype) data matrix for the same \eqn{n} subjects.}

\item{Lambda1}{LASSO penalty parameter for \code{X1}. \code{Lambda1} needs
to be between 0 and 1.}

\item{s1}{Proportion of features in \code{X1} to be included, default at \code{s1 =
0.7}. \code{s1} needs to be between 0 and 1, default is set to 0.7.}

\item{SubsamplingNum}{Number of feature subsamples. Default is 1000. Larger
number leads to more accurate results, but at a higher computational cost.}

\item{trace}{Whether to display the CCA algorithm trace, default is set to FALSE.}
}
\value{
A canonical correlation weight matrix with \eqn{p_1} rows. Each
column is the canonical correlation weights based on subsampled \code{X1}
features. The number of columns is \code{SubsamplingNum}.
}
\description{
Compute aggregated (SmCCA) canonical weights for single omics data with quantitative phenotype (subampling enabled).
}
\examples{


## For illustration, we only subsample 5 times.
set.seed(123)

# Single Omics SmCCA
W1 <- getRobustWeightsSingle(X1, Trait = Y, Lambda1 = 0.05,
  s1 = 0.7, 
  SubsamplingNum = 5, trace = FALSE)
  
  
}
