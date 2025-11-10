#'@title Functional Multi-Source Geographically Weighted Regression
#'@description fdagwr fits Functional Weighted Regression models, providing the corresponding predictor. 
#' Covariates can be stationary and, if varying over geographical space, non-stationary. At most two different sources
#' of non stationarity (where an event happens and where it is measured) can be handled.
#' Available models: Functional Weighted Regression (FWR), Functional Geographically Weighted Regression (FGWR),
#' Functional Mixed Geographically Weighted Regression (FGWR), Functional Multi-Source Geographically Weighted Regression (FMSGWR), both ESC and SEC versions.
#' Functional data, over a 1D domain, rather than response reconstruction weights, have to be represented as basis expansion in advance.
#'
#'@references
#'\itemize{
#'\item M. Fervari, A. Menafoglio and T. Bortolotti, \emph{"A Functional Multi-Source Geographically Weighted Regression for Ground Motion Modelling in Italy"}.2025.
#'\item T. Bortolotti, R. Peli, G. Lanzano, S. Sgobba and A. Menafoglio, \emph{"Weighted functional data analysis for the calibration of a ground motion model in italy"}. Journal of the American Statistical Association, 119(547):1697–1708, 2024.
#'\item L. Caramenti, A. Menafoglio, S. Sgobba and G. Lanzano, \emph{"Multi-source geographically weighted regression for regionalized ground-motion models"}. Spatial Statistics, 47:100610, 2022.
#'\item Source code: \href{https://github.com/AndreaEnricoFranzoni/FunctionalMultiSourceGeographicallyWeightedRegression}{fdagwr implementation}
#'\item To cite the package, please use: \preformatted{citation("fdagwr")}
#'}
#'@seealso
#'\itemize{
#'\item Functional Multi-Source Geographically Weighted Regression ESC:
#'\itemize{
#'\item model fitting: \code{\link{FMSGWR_ESC}}
#'\item predictor: \code{\link{predict_FMSGWR_ESC}}
#'\item beta tuner: \code{\link{beta_new_FMSGWR_ESC}}
#'\item prediction: \code{\link{y_new_FMSGWR_ESC}}}
#'\item Functional Multi-Source Geographically Weighted Regression SEC:
#'\itemize{
#'\item model fitting: \code{\link{FMSGWR_SEC}}
#'\item predictor: \code{\link{predict_FMSGWR_SEC}}
#'\item beta tuner: \code{\link{beta_new_FMSGWR_SEC}}
#'\item prediction: \code{\link{y_new_FMSGWR_SEC}}}
#'\item Functional Mixed Geographically Weighted Regression:
#'\itemize{
#'\item model fitting: \code{\link{FMGWR}}
#'\item predictor: \code{\link{predict_FMGWR}}
#'\item beta tuner: \code{\link{beta_new_FMGWR}}
#'\item prediction: \code{\link{y_new_FMGWR}}}
#'\item Functional Geographically Weighted Regression:
#'\itemize{
#'\item model fitting: \code{\link{FGWR}}
#'\item predictor: \code{\link{predict_FGWR}}
#'\item beta tuner: \code{\link{beta_new_FGWR}}
#'\item prediction: \code{\link{y_new_FGWR}}}
#'\item Functional Weighted Regression:
#'\itemize{
#'\item model fitting: \code{\link{FWR}}
#'\item predictor: \code{\link{predict_FWR}}}
#'\item Package installation check: \code{\link{installation_fdagwr}}}
#'
#'@author Andrea Enrico Franzoni, Alessandra Menafoglio
#'
#'@docType package
#'@name fdagwr
NULL