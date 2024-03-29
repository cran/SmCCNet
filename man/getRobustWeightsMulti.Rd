% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/MultiOmicsSmCCA.R
\name{getRobustWeightsMulti}
\alias{getRobustWeightsMulti}
\title{Run Sparse multiple Canonical Correlation Analysis and Obtain Canonical Weights (with Subsampling)}
\usage{
getRobustWeightsMulti(
  X,
  Trait,
  Lambda,
  s = NULL,
  NoTrait = FALSE,
  SubsamplingNum = 1000,
  CCcoef = NULL,
  trace = FALSE,
  TraitWeight = FALSE
)
}
\arguments{
\item{X}{A list of omics data each with n subjects.}

\item{Trait}{An \eqn{n\times 1} trait (phenotype) data matrix for the same n subjects.}

\item{Lambda}{Lasso penalty vector with length equals to the number of omics data (\eqn{X}). \code{Lambda} needs
to be between 0 and 1.}

\item{s}{A vector with length equals to the number of omics data (\eqn{X}), specifying the 
percentage of omics feature being subsampled at each subsampling iteration.}

\item{NoTrait}{Logical, default is \code{FALSE}. Whether trait information
is provided.}

\item{SubsamplingNum}{Number of feature subsamples. Default is 1000. Larger
number leads to more accurate results, but at a higher computational cost.}

\item{CCcoef}{Optional scaling factors for the SmCCA pairwise canonical
correlations. If \code{CCcoef = NULL} (default), then the objective function
is the total sum of all pairwise canonical correlations. This 
coefficient vector follows the column order of \code{combn(T+1, 2)} assuming there are T omics data and a phenotype data.}

\item{trace}{Whether to display the CCA algorithm trace, default is set to \code{FALSE}.}

\item{TraitWeight}{Whether to return canonical weight for trait (phenotype), default is set to \code{FALSE}.}
}
\value{
A canonical correlation weight matrix with \eqn{p = \sum_{i} p_i} rows, where \eqn{p_i} is the number of features for the \eqn{i}th omics. Each
column is the canonical correlation weights based on subsampled features. The number of columns is \code{SubsamplingNum}.
}
\description{
SmCCNet algorithm with multi-omics data and quantitative phenotype. Calculate the canonical weights for SmCCA.
}
\examples{


## For illustration, we only subsample 5 times.
set.seed(123)
X1 <- matrix(rnorm(600,0,1), nrow = 60)
X2 <- matrix(rnorm(600,0,1), nrow = 60)
Y <- matrix(rnorm(60,0,1), nrow = 60)
# Unweighted SmCCA
result <- getRobustWeightsMulti(X = list(X1, X2), Trait = Y, NoTrait = FALSE,
Lambda = c(0.5, 0.5),s = c(0.7, 0.7), SubsamplingNum = 20)
  
}
