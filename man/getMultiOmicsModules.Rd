% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SmCCNetSource.R
\name{getMultiOmicsModules}
\alias{getMultiOmicsModules}
\title{Extract multi-omics modules based on the similarity matrix.}
\usage{
getMultiOmicsModules(Abar, P1, CutHeight = 1 - 0.1^10, PlotTree = TRUE)
}
\arguments{
\item{Abar}{A similary matrix for all features (both omics data types).}

\item{P1}{Total number of features for the first omics data type.}

\item{CutHeight}{Height threshold for the hierarchical tree cutting. Default 
is \eqn{1-0.1^{10}}.}

\item{PlotTree}{Logical. Whether to create a hierarchical tree plot.}
}
\value{
A list of multi-omics modules.
}
\description{
Apply hierarchical tree cutting to the similarity matrix and extract
modules that contain both omics data types.
}
\examples{
set.seed(123)
w <- rnorm(5)
w <- w/sqrt(sum(w^2))
abar <- getAbar(w, P1 = 2, FeatureLabel = NULL)
modules <- getMultiOmicsModules(abar, P1 = 2, CutHeight = 0.5)

}
