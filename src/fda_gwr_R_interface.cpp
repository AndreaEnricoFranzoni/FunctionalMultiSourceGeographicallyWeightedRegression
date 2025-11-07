// Copyright (c) 2025 Andrea Enrico Franzoni (andreaenrico.franzoni@gmail.com)
//
// This file is part of fdagwr
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of fdagwr and associated documentation files (the fdagwr software), to deal
// fdagwr without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of fdagwr, and to permit persons to whom fdagwr is
// furnished to do so, subject to the following conditions:
//
// fdagwr IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH fdagwr OR THE USE OR OTHER DEALINGS IN
// fdagwr.


#include <RcppEigen.h>


#include "utility/include_fdagwr.hpp"
#include "utility/traits_fdagwr.hpp"
#include "utility/concepts_fdagwr.hpp"
#include "utility/utility_fdagwr.hpp"
#include "utility/data_reader.hpp"
#include "utility/parameters_wrapper_fdagwr.hpp"

#include "basis/basis_include.hpp"
#include "basis/basis_bspline_systems.hpp"
#include "basis/basis_factory_proxy.hpp"

#include "functional_data/functional_data.hpp"
#include "functional_data/functional_data_covariates.hpp"

#include "weight_matrix/functional_weight_matrix_stat.hpp"
#include "weight_matrix/functional_weight_matrix_no_stat.hpp"
#include "weight_matrix/distance_matrix.hpp"
#include "weight_matrix/distance_matrix_pred.hpp"

#include "penalization_matrix/penalization_matrix.hpp"

#include "functional_matrix/functional_matrix.hpp"
#include "functional_matrix/functional_matrix_sparse.hpp"
#include "functional_matrix/functional_matrix_diagonal.hpp"
#include "functional_matrix/functional_matrix_operators.hpp"
#include "functional_matrix/functional_matrix_product.hpp"
#include "functional_matrix/functional_matrix_into_wrapper.hpp"


#include "fwr/fwr_factory.hpp"
#include "fwr_predictor/fwr_predictor_factory.hpp"


/*!
* @file fda_gwr_R_interface.cpp
* @brief Contains the R-interfaced main functions of the package 'fdagwr', which implement Functional Geographical Weighted Regression
*        coefficients estimation, for different (multi-source (FMSGWR), mixed (FMGWR), geographically weighted (FGWR) and 
*        weighted (FWR)) functional regression models.
* @author Andrea Enrico Franzoni
*/




using namespace Rcpp;

//
// [[Rcpp::depends(RcppEigen)]]


/*!
* @brief Check fdagwr package installation
*/
//
// [[Rcpp::export]]
void installation_fdagwr(){   Rcout << "fdagwr9 installation successful"<< std::endl;}





/*!
* @brief Fitting a Functional Multi-Source Geographically Weighted Regression ESC model. The covariates are functional objects, divided into
*        three categories: stationary covariates (C), constant over geographical space, event-dependent covariates (E), that vary depending on the spatial coordinates of the event, 
*        station-dependent covariates (S), that vary depending on the spatial coordinates of the stations that measure the event. Regression coefficients are estimated 
*        in the following order: C, S, E. The functional response is already reconstructed according to the method proposed by Bortolotti et Al. (2024) (link below)
* @param y_points matrix of double containing the raw response: each row represents a specific abscissa for which the response evaluation is available, each column a statistical unit. Response is a already reconstructed.
* @param t_points vector of double with the abscissa points with respect of the raw evaluations of y_points are available (length of t_points is equal to the number of rows of y_points).
* @param left_extreme_domain double indicating the left extreme of the functional data domain (not necessarily the smaller element in t_points).
* @param right_extreme_domain double indicating the right extreme of the functional data domain (not necessarily the biggest element in t_points).
* @param coeff_y_points matrix of double containing the coefficient of response's basis expansion: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param knots_y_points vector of double with the abscissa points with respect which the basis expansions of the response and response reconstruction weights are performed (all elements contained in [a,b]). 
* @param degree_basis_y_points non-negative integer: the degree of the basis used for the basis expansion of the (functional) response. Default explained below (can be NULL).
* @param n_basis_y_points positive integer: number of basis for the basis expansion of the (functional) response. It must match number of rows of coeff_y_points. Default explained below (can be NULL).
* @param coeff_rec_weights_y_points matrix of double containing the coefficients of the basis expansion of the weights to reconstruct the (functional) response: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param degree_basis_rec_weights_y_points non-negative integer: the degree of the basis used for response reconstruction weights. Default explained below (can be NULL).
* @param n_basis_rec_weights_y_points positive integer: number of basis for the basis expansion of response reconstruction weights. It must match number of rows of coeff_rec_weights_y_points. Default explained below (can be NULL).
* @param coeff_stationary_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                             The name of the i-th element is the name of the i-th stationary covariate (default: "reg.Ci" if no name present).
* @param basis_types_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariate basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate. Default explained below (can be NULL).
* @param n_basis_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate. It must match number of rows of the i-th element of coeff_stationary_cov. Default explained below (can be NULL).
* @param penalization_stationary_cov vector of non-negative double: element i-th is the penalization used for the i-th stationary covariate.
* @param knots_beta_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates functional regression coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate functional regression coefficients. Default explained below (can be NULL).
* @param n_basis_beta_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate functional regression coefficients. Default explained below (can be NULL).
* @param coeff_events_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th events-dependent covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                         The name of the i-th element is the name of the i-th events-dependent covariate (default: "reg.Ei" if no name present).
* @param basis_types_events_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariate basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_events_cov vector of double with the abscissa points with respect which the basis expansions of the events-dependent covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_events_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th events-dependent covariate. Default explained below (can be NULL).
* @param n_basis_events_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th events-dependent covariate. It must match number of rows of the i-th element of coeff_events_cov. Default explained below (can be NULL).
* @param penalization_events_cov vector of non-negative double: element i-th is the penalization used for the i-th events-dependent covariate.
* @param coordinates_events matrix of double containing the UTM coordinates of the event of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
* @param kernel_bandwith_events positive double indicating the bandwith of the gaussian kernel used to smooth the distances within events.
* @param knots_beta_events_cov vector of double with the abscissa points with respect which the basis expansions of the events-dependent covariates functional regression coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_events_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th events-dependent covariate functional regression coefficient. Default explained below (can be NULL).
* @param n_basis_beta_events_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th events-dependent covariate functional regression coefficient. Default explained below (can be NULL).
* @param coeff_stations_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stations-dependent covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                           The name of the i-th element is the name of the i-th stations-dependent covariate (default: "reg.Si").
* @param basis_types_stations_cov vector of strings, element i-th containing the type of basis used for the i-th stations-dependent covariates basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_stations_cov vector of double with the abscissa points with respect which the basis expansions of the stations-dependent covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_stations_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stations-dependent covariate. Default explained below (can be NULL).
* @param n_basis_stations_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stations-dependent covariate. It must match number of rows of the i-th element of coeff_stations_cov. Default explained below (can be NULL).
* @param penalization_stations_cov vector of non-negative double: element i-th is the penalization used for the i-th stations-dependent covariate.
* @param coordinates_stations matrix of double containing the UTM coordinates of the station of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
* @param kernel_bandwith_stations positive double indicating the bandwith of the gaussian kernel used to smooth the distances within stations.
* @param knots_beta_stations_cov vector of double with the abscissa points with respect which the basis expansions of the stations-dependent covariates functional regression coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_stations_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stations-dependent covariate functional regression coefficient. Default explained below (can be NULL).
* @param n_basis_beta_stations_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stations-dependent covariate functional regression coefficient. Default explained below (can be NULL).
* @param in_cascade_estimation bool: if false, an exact algorithm taking account for the interaction within non-stationary covariates is used to fit the model. Otherwise, the model is fitted in cascade. The first option is more precise, but way more computationally intensive.
* @param n_knots_smoothing number of knots used to perform the smoothing on the response obtained leaving out all the non-stationary components (default: 100).
* @param n_intervals_quadrature number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: 100).
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine.
* @param basis_type_y_points string containing the type of basis used for the functional response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_type_rec_weights_y_points string containing the type of basis used for the weights to reconstruct the functional response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariate functional regression coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_events_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariate functional regression coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stations_cov vector of strings, element i-th containing the type of basis used for the i-th stations-dependent covariate functional regression coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @return an R list containing:
*         - "FGWR": string containing the type of fwr used ("FMSGWR_ESC")
*         - "EstimationTechnique": "Exact" if in_cascade_estimation false, "Cascade" if in_cascade_estimation true 
*         - "Bc": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "basis_coeff": a Lc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective element of basis_types_beta_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stationary_cov)
*         - "Beta_c": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Beta_eval": a vector of double containing the discrete evaluations of the stationary beta
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "Be": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Le_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_events_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_events_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_events_cov)
*         - "Beta_e": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)  
*         - "Bs": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Ls_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_stations_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stations_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stations_cov)
*         - "Beta_s": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "predictor_info": a list containing partial residuals and information of the fitted model to perform predictions for new statistical units:
*                             - "partial_res": a list containing information to compute the partial residuals:
*                                              - "c_tilde_hat": vector of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon (if in_cascade_estimation is true, contains only 0s).
*                                              - "A__": vector of matrices with the operator A_e for each statistical unit (if in_cascade_estimation is true, each matrix contains only 0s).
*                                              - "B__for_K": vector of matrices with the operator B_e used for the K_e_s(t) computation, for each statistical unit (if in_cascade_estimation is true, each matrix contains only 0s).
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response (element n_basis_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response. Possible values: "bsplines", "constant". (element basis_type_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response (element degree_basis_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response (element coeff_y_points).
*                                              - "ResponseReconstructionWeights": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response reconstruction weights (element n_basis_rec_weights_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response reconstruction weights. Possible values: "bsplines", "constant". (element basis_type_rec_weights_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response reconstruction weights (element degree_basis_rec_weights_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response reconstruction weights (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response reconstruction weights (element coeff_rec_weights_y_points).
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates": number of stationary covariates (length of coeff_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional stationary covariates (respective elements of n_basis_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional stationary covariates (respective elements of degrees_basis_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional stationary covariates (element knots_stationary_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional stationary covariates (respective elements of coeff_stationary_cov).
*                                              - "beta_Stationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element n_basis_beta_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element degrees_basis_beta_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the stationary covariates (element knots_beta_stationary_cov).                                                            
*                                              - "cov_Event": list:
*                                                            - "number_covariates": number of event-dependent covariates (length of coeff_events_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional event-dependent covariates (respective elements of n_basis_events_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional event-dependent covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_events_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional event-dependent covariates (respective elements of degrees_basis_events_cov).
*                                                            - "knots": knots used to make the basis expansion of functional event-dependent covariates (element knots_events_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional event-dependent covariates (respective elements of coeff_events_cov).
*                                                            - "penalizations": vector containing the penalizations of the event-dependent covariates (respective elements of penalization_events_cov)
*                                                            - "coordinates": UTM coordinates of the events of the training data (element coordinates_events).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the events (element kernel_bandwith_events).
*                                              - "beta_Event": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element n_basis_beta_events_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates. Possible values: "bsplines", "constant". (element basis_types_beta_events_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element degrees_basis_beta_events_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element knots_beta_events_cov).
*                                              - "cov_Station": list:
*                                                            - "number_covariates": number of station-dependent covariates (length of coeff_stations_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional station-dependent covariates (respective elements of n_basis_stations_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional station-dependent covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stations_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional station-dependent covariates (respective elements of degrees_basis_stations_cov).
*                                                            - "knots": knots used to make the basis expansion of functional station-dependent covariates (element knots_stations_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional station-dependent covariates (respective elements of coeff_stations_cov).
*                                                            - "penalizations": vector containing the penalizations of the station-dependent covariates (respective elements of penalization_stations_cov)
*                                                            - "coordinates": UTM coordinates of the stations of the training data (element coordinates_stations).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the stations (element kernel_bandwith_stations).
*                                              - "beta_Station": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element n_basis_beta_stations_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stations_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element degrees_basis_beta_stations_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element knots_beta_stations_cov).
*                                              - "a": domain left extreme  (element left_extreme_domain).
*                                              - "b": domain right extreme (element right_extreme_domain).
*                                              - "abscissa": abscissa for which the evaluations of the functional data are available (element t_points).
*                                              - "InCascadeEstimation": element in_cascade_estimation.
* @details constant basis are used, for a covariate, if it resembles a scalar shape. It consists of a straight line with y-value equal to 1 all over the data domain.
*          Can be seen as a B-spline basis with degree 0, number of basis 1, using one knot, consequently having only one coefficient for the only basis for each statistical unit.
*          fdagwr sets all the feats accordingly if reads constant basis.
*          However, recall that the response is a functional datum, as the regressors coefficients. Since the package's basis variety could be hopefully enlarged in the future 
*          (for example, introducing Fourier basis for handling data that present periodical behaviors), the input parameters regarding basis types for response, response reconstruction
*          weights and regressors coefficients are left at the end of the input list, and defaulted as NULL. Consequently they will use a B-spline basis system, and should NOT use a constant basis,
*          Recall to perform externally the basis expansion before using the package, and afterwards passing basis types, degree and number and basis expansion coefficients and knots coherently
* @note a little excursion about degree and number of basis passed as input. For each specific covariate, or the response, if using B-spline basis, remember that number of knots = number of basis - degree + 1. 
*       By default, if passing NULL, fdagwr uses a cubic B-spline system of basis, the number of basis is computed coherently from the number of knots (that is the only mandatory input parameter).
*       Passing only the degree of the bsplines, the number of basis used will be set accordingly, and viceversa if passing only the number of basis. 
*       But, take care that the number of basis used has to match the number of rows of coefficients matrix (for EACH type of basis). If not, an exception is thrown. No problems arise if letting fdagwr defaulting the number of basis.
*       For response and response reconstruction weights, degree and number of basis consist of integer, and can be NULL. For all the regressors, and their coefficients, the inputs consist of vector of integers: 
*       if willing to pass a default parameter, all the vector has to be defaulted (if passing NULL, a vector with all 3 for the degrees is passed, for example)
* @link https://www.researchgate.net/publication/377251714_Weighted_Functional_Data_Analysis_for_the_Calibration_of_a_Ground_Motion_Model_in_Italy @endlink
*/
//
// [[Rcpp::export]]
Rcpp::List FMSGWR_ESC(Rcpp::NumericMatrix y_points,
                      Rcpp::NumericVector t_points,
                      double left_extreme_domain,
                      double right_extreme_domain,
                      Rcpp::NumericMatrix coeff_y_points,
                      Rcpp::NumericVector knots_y_points,
                      Rcpp::Nullable<int> degree_basis_y_points,
                      Rcpp::Nullable<int> n_basis_y_points,
                      Rcpp::NumericMatrix coeff_rec_weights_y_points,
                      Rcpp::Nullable<int> degree_basis_rec_weights_y_points,
                      Rcpp::Nullable<int> n_basis_rec_weights_y_points,
                      Rcpp::List coeff_stationary_cov,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_stationary_cov,
                      Rcpp::NumericVector knots_stationary_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_stationary_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_stationary_cov,
                      Rcpp::NumericVector penalization_stationary_cov,
                      Rcpp::NumericVector knots_beta_stationary_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_beta_stationary_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_stationary_cov,
                      Rcpp::List coeff_events_cov,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_events_cov,
                      Rcpp::NumericVector knots_events_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_events_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_events_cov,
                      Rcpp::NumericVector penalization_events_cov,
                      Rcpp::NumericMatrix coordinates_events,
                      double kernel_bandwith_events,
                      Rcpp::NumericVector knots_beta_events_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_beta_events_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_events_cov,
                      Rcpp::List coeff_stations_cov,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_stations_cov,
                      Rcpp::NumericVector knots_stations_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_stations_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_stations_cov,
                      Rcpp::NumericVector penalization_stations_cov,
                      Rcpp::NumericMatrix coordinates_stations,
                      double kernel_bandwith_stations,
                      Rcpp::NumericVector knots_beta_stations_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_beta_stations_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_stations_cov,
                      bool in_cascade_estimation = false,
                      int n_knots_smoothing = 100,
                      int n_intervals_quadrature = 100,
                      Rcpp::Nullable<int> num_threads = R_NilValue,
                      std::string basis_type_y_points = "bsplines",
                      std::string basis_type_rec_weights_y_points = "bsplines",
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_events_cov = R_NilValue,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stations_cov = R_NilValue)
{
    Rcout << "Functional Multi-Source Geographically Weighted Regression ESC" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT

    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FMSGWR_ESC_;                         //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _EVENT_ = FDAGWR_COVARIATES_TYPES::EVENT;                        //enum for event covariates
    constexpr auto _STATION_ = FDAGWR_COVARIATES_TYPES::STATION;                    //enum for station covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF KNOTS TO PERFORM SMOOTHING ON THE RESPONSE WITHOUT THE NON-STATIONARY COMPONENTS
    int n_knots_smoothing_y_new = wrap_and_check_n_knots_smoothing(n_knots_smoothing);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA MIDPOINT QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);


    //  RESPONSE
    //raw data
    auto response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(y_points);       //Eigen dense matrix type
    //number of statistical units
    std::size_t number_of_statistical_units_ = response_.cols();
    //coefficients matrix
    auto coefficients_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_y_points);
    auto coefficients_response_out_ = coefficients_response_;
    //c: a dense matrix of double (n*Ly) x 1 containing, one column below the other, the y basis expansion coefficients
    auto c = columnize_coeff_resp(coefficients_response_);
    //reconstruction weights coefficients matrix
    auto coefficients_rec_weights_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_rec_weights_y_points);
    auto coefficients_rec_weights_response_out_ = coefficients_rec_weights_response_;

    //  ABSCISSA POINTS of response
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = wrap_abscissas(t_points,left_extreme_domain,right_extreme_domain);
    // wrapper into eigen
    check_dim_input<_RESPONSE_>(response_.rows(), abscissa_points_.size(), "points for evaluation of raw data vector");   //check that size of abscissa points and number of evaluations of fd raw data coincide
    FDAGWR_TRAITS::Dense_Matrix abscissa_points_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(abscissa_points_.data(),abscissa_points_.size(),1);
    _FD_INPUT_TYPE_ a = left_extreme_domain;
    _FD_INPUT_TYPE_ b = right_extreme_domain;


    //  KNOTS (for basis expansion and for smoothing)
    //response
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = wrap_abscissas(knots_y_points,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    //stationary cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_ = wrap_abscissas(knots_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    //beta stationary cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = wrap_abscissas(knots_beta_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //events cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_events_cov_ = wrap_abscissas(knots_events_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_events_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    //beta events cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_events_cov_ = wrap_abscissas(knots_beta_events_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_events_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size());
    //stations cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stations_cov_ = wrap_abscissas(knots_stations_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_stations_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    //stations beta cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stations_cov_ = wrap_abscissas(knots_beta_stations_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_stations_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());
    //knots for performing smoothing (n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);


    //  COVARIATES names, coefficients and how many (q_), for every type
    //stationary 
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov);
    std::size_t q_C = names_stationary_cov_.size();    //number of stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov);    
    //events
    std::vector<std::string> names_events_cov_ = wrap_covariates_names<_EVENT_>(coeff_events_cov);
    std::size_t q_E = names_events_cov_.size();        //number of events related covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_ = wrap_covariates_coefficients<_EVENT_>(coeff_events_cov);
    //stations
    std::vector<std::string> names_stations_cov_ = wrap_covariates_names<_STATION_>(coeff_stations_cov);
    std::size_t q_S = names_stations_cov_.size();      //number of stations related covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_ = wrap_covariates_coefficients<_STATION_>(coeff_stations_cov);


    //  BASIS TYPES
    //response
    std::string basis_type_response_ = wrap_and_check_basis_type<_RESPONSE_>(basis_type_y_points);
    //response reconstruction weights
    std::string basis_type_rec_weights_response_ = wrap_and_check_basis_type<_REC_WEIGHTS_>(basis_type_rec_weights_y_points);
    //stationary
    std::vector<std::string> basis_types_stationary_cov_ = wrap_and_check_basis_type<_STATIONARY_>(basis_types_stationary_cov,q_C);
    //beta stationary cov 
    std::vector<std::string> basis_types_beta_stationary_cov_ = wrap_and_check_basis_type<_STATIONARY_>(basis_types_beta_stationary_cov,q_C);
    //events
    std::vector<std::string> basis_types_events_cov_ = wrap_and_check_basis_type<_EVENT_>(basis_types_events_cov,q_E);
    //beta events cov 
    std::vector<std::string> basis_types_beta_events_cov_ = wrap_and_check_basis_type<_EVENT_>(basis_types_beta_events_cov,q_E);
    //stations
    std::vector<std::string> basis_types_stations_cov_ = wrap_and_check_basis_type<_STATION_>(basis_types_stations_cov,q_S);
    //beta stations cov 
    std::vector<std::string> basis_types_beta_stations_cov_ = wrap_and_check_basis_type<_STATION_>(basis_types_beta_stations_cov,q_S);


    //  BASIS NUMBER AND DEGREE: checking matrix coefficients dimensions: rows: number of basis; cols: number of statistical units
    //response
    auto number_and_degree_basis_response_ = wrap_and_check_basis_number_and_degree<_RESPONSE_>(n_basis_y_points,degree_basis_y_points,knots_response_.size(),basis_type_response_);
    std::size_t number_basis_response_ = number_and_degree_basis_response_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::size_t degree_basis_response_ = number_and_degree_basis_response_[std::string{FDAGWR_FEATS::degree_basis_string}];
    check_dim_input<_RESPONSE_>(number_basis_response_,coefficients_response_.rows(),"response coefficients matrix rows");
    check_dim_input<_RESPONSE_>(number_of_statistical_units_,coefficients_response_.cols(),"response coefficients matrix columns");     
    //response reconstruction weights
    auto number_and_degree_basis_rec_weights_response_ = wrap_and_check_basis_number_and_degree<_REC_WEIGHTS_>(n_basis_rec_weights_y_points,degree_basis_rec_weights_y_points,knots_response_.size(),basis_type_rec_weights_response_);
    std::size_t number_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::size_t degree_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[std::string{FDAGWR_FEATS::degree_basis_string}];
    check_dim_input<_REC_WEIGHTS_>(number_basis_rec_weights_response_,coefficients_rec_weights_response_.rows(),"response reconstruction weights coefficients matrix rows");
    check_dim_input<_REC_WEIGHTS_>(number_of_statistical_units_,coefficients_rec_weights_response_.cols(),"response reconstruction weights coefficients matrix columns");     
    //stationary cov
    auto number_and_degree_basis_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_stationary_cov,degrees_basis_stationary_cov,knots_stationary_cov_.size(),q_C,basis_types_stationary_cov_);
    std::vector<std::size_t> number_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stationary cov
    auto number_and_degree_basis_beta_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_beta_stationary_cov,degrees_basis_beta_stationary_cov,knots_beta_stationary_cov_.size(),q_C,basis_types_beta_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    //events cov    
    auto number_and_degree_basis_events_cov_ = wrap_and_check_basis_number_and_degree<_EVENT_>(n_basis_events_cov,degrees_basis_events_cov,knots_events_cov_.size(),q_E,basis_types_events_cov_);
    std::vector<std::size_t> number_basis_events_cov_ = number_and_degree_basis_events_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_events_cov_ = number_and_degree_basis_events_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    for(std::size_t i = 0; i < q_E; ++i){   
        check_dim_input<_EVENT_>(number_basis_events_cov_[i],coefficients_events_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_EVENT_>(number_of_statistical_units_,coefficients_events_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta events cov
    auto number_and_degree_basis_beta_events_cov_ = wrap_and_check_basis_number_and_degree<_EVENT_>(n_basis_beta_events_cov,degrees_basis_beta_events_cov,knots_beta_events_cov_.size(),q_E,basis_types_beta_events_cov_);
    std::vector<std::size_t> number_basis_beta_events_cov_ = number_and_degree_basis_beta_events_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = number_and_degree_basis_beta_events_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    //stations cov
    auto number_and_degree_basis_stations_cov_ = wrap_and_check_basis_number_and_degree<_STATION_>(n_basis_stations_cov,degrees_basis_stations_cov,knots_stations_cov_.size(),q_S,basis_types_stations_cov_);
    std::vector<std::size_t> number_basis_stations_cov_ = number_and_degree_basis_stations_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_stations_cov_ = number_and_degree_basis_stations_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    for(std::size_t i = 0; i < q_S; ++i){   
        check_dim_input<_STATION_>(number_basis_stations_cov_[i],coefficients_stations_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATION_>(number_of_statistical_units_,coefficients_stations_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stations cov 
    auto number_and_degree_basis_beta_stations_cov_ = wrap_and_check_basis_number_and_degree<_STATION_>(n_basis_beta_stations_cov,degrees_basis_beta_stations_cov,knots_beta_stations_cov_.size(),q_S,basis_types_beta_stations_cov_);
    std::vector<std::size_t> number_basis_beta_stations_cov_ = number_and_degree_basis_beta_stations_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = number_and_degree_basis_beta_stations_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];


    //  DISTANCES
    //events    DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_events_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_events);
    auto coordinates_events_out_ = coordinates_events_;
    check_dim_input<_EVENT_>(number_of_statistical_units_,coordinates_events_.rows(),"coordinates matrix rows");
    check_dim_input<_EVENT_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_events_.cols(),"coordinates matrix columns");
    distance_matrix<_DISTANCE_> distances_events_cov_(std::move(coordinates_events_),number_threads);
    //stations  DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_stations_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_stations);
    auto coordinates_stations_out_ = coordinates_stations_;
    check_dim_input<_STATION_>(number_of_statistical_units_,coordinates_stations_.rows(),"coordinates matrix rows");
    check_dim_input<_STATION_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_stations_.cols(),"coordinates matrix columns");
    distance_matrix<_DISTANCE_> distances_stations_cov_(std::move(coordinates_stations_),number_threads);


    //  PENALIZATION TERMS: checking their consistency
    //stationary
    std::vector<double> lambda_stationary_cov_ = wrap_and_check_penalizations<_STATIONARY_>(penalization_stationary_cov,q_C);
    //events
    std::vector<double> lambda_events_cov_ = wrap_and_check_penalizations<_EVENT_>(penalization_events_cov,q_E);
    //stations
    std::vector<double> lambda_stations_cov_ = wrap_and_check_penalizations<_STATION_>(penalization_stations_cov,q_S);


    //  KERNEL BANDWITH
    //events
    double kernel_bandwith_events_cov_ = wrap_and_check_kernel_bandwith<_EVENT_>(kernel_bandwith_events);
    //stations
    double kernel_bandwith_stations_cov_ = wrap_and_check_kernel_bandwith<_STATION_>(kernel_bandwith_stations);

    ////////////////////////////////////////
    /////    END PARAMETERS WRAPPING   /////
    ////////////////////////////////////////



    ////////////////////////////////
    /////    OBJECT CREATION   /////
    ////////////////////////////////


    //DISTANCES
    //events
    distances_events_cov_.compute_distances();
    //stations
    distances_stations_cov_.compute_distances();


    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    //events (Theta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_E(knots_beta_events_cov_eigen_w_, 
                                                   degree_basis_beta_events_cov_, 
                                                   number_basis_beta_events_cov_, 
                                                   q_E);
    //stations (Psi)
    basis_systems< _DOMAIN_, bsplines_basis > bs_S(knots_beta_stations_cov_eigen_w_,  
                                                   degree_basis_beta_stations_cov_, 
                                                   number_basis_beta_stations_cov_, 
                                                   q_S);
    
    
    //PENALIZATION MATRICES
    //stationary
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_C(std::move(bs_C),lambda_stationary_cov_);
    std::size_t Lc = R_C.L();
    std::vector<std::size_t> Lc_j = R_C.Lj();
    //events
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_E(std::move(bs_E),lambda_events_cov_);
    std::size_t Le = R_E.L();
    std::vector<std::size_t> Le_j = R_E.Lj();
    //stations
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_S(std::move(bs_S),lambda_stations_cov_);
    std::size_t Ls = R_S.L();
    std::vector<std::size_t> Ls_j = R_S.Lj();


    //FD OBJECTS: RESPONSE and COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_(std::move(coefficients_response_),std::move(basis_response_));
    
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    
    //stationary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_(coefficients_stationary_cov_,
                                                              q_C,
                                                              basis_types_stationary_cov_,
                                                              degree_basis_stationary_cov_,
                                                              number_basis_stationary_cov_,
                                                              knots_stationary_cov_eigen_w_,
                                                              basis_fac);
    
    //events covariates
    functional_data_covariates<_DOMAIN_,_EVENT_> x_E_fd_(coefficients_events_cov_,
                                                         q_E,
                                                         basis_types_events_cov_,
                                                         degree_basis_events_cov_,
                                                         number_basis_events_cov_,
                                                         knots_events_cov_eigen_w_,
                                                         basis_fac);
    
    //stations covariates
    functional_data_covariates<_DOMAIN_,_STATION_> x_S_fd_(coefficients_stations_cov_,
                                                           q_S,
                                                           basis_types_stations_cov_,
                                                           degree_basis_stations_cov_,
                                                           number_basis_stations_cov_,
                                                           knots_stations_cov_eigen_w_,
                                                           basis_fac);


    //FUNCTIONAL WEIGHT MATRIX
    //stationary
    functional_weight_matrix_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATIONARY_> W_C(rec_weights_y_fd_,
                                                                                                                                                    number_threads);
    W_C.compute_weights();                                                      
    //events
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_,_KERNEL_,_DISTANCE_> W_E(rec_weights_y_fd_,
                                                                                                                                                                       std::move(distances_events_cov_),
                                                                                                                                                                       kernel_bandwith_events_cov_,
                                                                                                                                                                       number_threads);
    W_E.compute_weights();                                                                         
    //stations
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_,_KERNEL_,_DISTANCE_> W_S(rec_weights_y_fd_,
                                                                                                                                                                         std::move(distances_stations_cov_),
                                                                                                                                                                         kernel_bandwith_stations_cov_,
                                                                                                                                                                         number_threads);
    W_S.compute_weights();


    ///////////////////////////////
    /////    FGWR ALGORITHM   /////
    ///////////////////////////////
    //wrapping all the functional elements in a functional_matrix

    //y: a column vector of dimension nx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_,number_threads);
    //phi: a sparse functional matrix nx(n*L), where L is the number of basis for the response
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> phi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_.fdata_basis(),number_of_statistical_units_,number_basis_response_);
    //c: a dense matrix of double (n*Ly) x 1 containing, one column below the other, the y basis expansion coefficients
    //already done at the beginning
    //basis used for doing response basis expansion
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //Xc: a functional matrix of dimension nxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_,number_threads);
    //Wc: a diagonal functional matrix of dimension nxn
    functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Wc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATIONARY_>(W_C,number_threads);
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);
    //Xe: a functional matrix of dimension nxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xe = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_EVENT_>(x_E_fd_,number_threads);
    //We: n diagonal functional matrices of dimension nxn
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > We = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_>(W_E,number_threads);
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> theta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_E);
    //Xs: a functional matrix of dimension nxqs
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xs = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATION_>(x_S_fd_,number_threads);
    //Ws: n diagonal functional matrices of dimension nxn
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Ws = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_>(W_S,number_threads);
    //psi: a sparse functional matrix of dimension qsxLs
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> psi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_S);


    //fwr model 
    auto fgwr_algo = fwr_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(y),
                                                                                   std::move(phi),
                                                                                   std::move(c),
                                                                                   number_basis_response_,
                                                                                   std::move(basis_y),
                                                                                   std::move(knots_smoothing),
                                                                                   std::move(Xc),
                                                                                   std::move(Wc),
                                                                                   std::move(R_C.PenalizationMatrix()),
                                                                                   std::move(omega),
                                                                                   q_C,
                                                                                   Lc,
                                                                                   Lc_j,
                                                                                   std::move(Xe),
                                                                                   std::move(We),
                                                                                   std::move(R_E.PenalizationMatrix()),
                                                                                   std::move(theta),
                                                                                   q_E,
                                                                                   Le,
                                                                                   Le_j,
                                                                                   std::move(Xs),
                                                                                   std::move(Ws),
                                                                                   std::move(R_S.PenalizationMatrix()),
                                                                                   std::move(psi),
                                                                                   q_S,
                                                                                   Ls,
                                                                                   Ls_j,
                                                                                   a,
                                                                                   b,
                                                                                   n_intervals,
                                                                                   abscissa_points_,
                                                                                   number_of_statistical_units_,
                                                                                   number_threads,
                                                                                   in_cascade_estimation);

    Rcout << "Model fitting" << std::endl;    
    
    //computing the b
    fgwr_algo->compute();
    //evaluating the betas   
    fgwr_algo->evalBetas();

    Rcout << "Model fitted" << std::endl; 

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fgwr_algo->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 names_events_cov_,
                                                 basis_types_beta_events_cov_,
                                                 number_basis_beta_events_cov_,
                                                 knots_beta_events_cov_,
                                                 names_stations_cov_,
                                                 basis_types_beta_stations_cov_,
                                                 number_basis_beta_stations_cov_,
                                                 knots_beta_stations_cov_);
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fgwr_algo->betas(),
                                           abscissa_points_,
                                           names_stationary_cov_,
                                           {},
                                           names_events_cov_,
                                           names_stations_cov_);
    //elements for partial residuals
    Rcpp::List p_res = wrap_PRes_to_R_list(fgwr_algo->PRes());

    
    //returning element
    Rcpp::List l;
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _estimation_iter_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter};     //Exact or Cascade estimation
    std::string _bc_                = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_            = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _be_                = std::string{FDAGWR_B_NAMES::be};                                 //be
    std::string _beta_e_            = std::string{FDAGWR_BETAS_NAMES::beta_e};                         //beta_e
    std::string _bs_                = std::string{FDAGWR_B_NAMES::bs};                                 //bs
    std::string _beta_s_            = std::string{FDAGWR_BETAS_NAMES::beta_s};                         //beta_s
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _partial_residuals_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res};               //partial residuals 
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                        //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                     //response reconstruction weights
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};   //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};   //stationary betas
    std::string _cov_event_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_EVENT_>()};        //event-dependent training covariates
    std::string _beta_event_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_EVENT_>()};        //event-dependent betas
    std::string _cov_station_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates
    std::string _beta_station_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates    
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations
    std::string _cascade_estimate_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cascade_estimate};         //if using in cascade-estimation

    //regression model used 
    l[_model_name_] = std::string{algo_type<_FGWR_ALGO_>()};
    //estimation technique
    l[_estimation_iter_] = estimation_iter(in_cascade_estimation);
    //stationary covariate basis expansion coefficients for beta_c
    l[_bc_]  = b_coefficients[_bc_];
    //beta_c
    l[_beta_c_] = betas[_beta_c_];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[_be_]  = b_coefficients[_be_];
    //beta_e
    l[_beta_e_] = betas[_beta_e_];
    //station-dependent covariate basis expansion coefficients for beta_s
    l[_bs_]  = b_coefficients[_bs_];
    //beta_s
    l[_beta_s_] = betas[_beta_s_];
    //elements needed to perform prediction
    Rcpp::List elem_for_pred;       
    elem_for_pred[_partial_residuals_] = p_res;     //partial residuals, elements to reconstruct them
    Rcpp::List inputs_info;                         //containing training data information needed for prediction purposes

    //adding all the elements of the training set
    //input of y
    Rcpp::List response_input;
    response_input[_n_basis_]     = number_basis_response_;
    response_input[_t_basis_]     = basis_type_response_;
    response_input[_deg_basis_]   = degree_basis_response_;
    response_input[_knots_basis_] = knots_response_;
    response_input[_coeff_basis_] = Rcpp::wrap(coefficients_response_out_);
    inputs_info[_response_]       = response_input;
    //input of w for y  
    Rcpp::List response_rec_w_input;
    response_rec_w_input[_n_basis_]     = number_basis_rec_weights_response_;
    response_rec_w_input[_t_basis_]     = basis_type_rec_weights_response_;
    response_rec_w_input[_deg_basis_]   = degree_basis_rec_weights_response_;
    response_rec_w_input[_knots_basis_] = knots_response_;
    response_rec_w_input[_coeff_basis_] = Rcpp::wrap(coefficients_rec_weights_response_out_);
    inputs_info[_response_rec_w_]       = response_rec_w_input;
    //input of C
    Rcpp::List C_input;
    C_input[_q_]            = q_C;
    C_input[_n_basis_]      = number_basis_stationary_cov_;
    C_input[_t_basis_]      = basis_types_stationary_cov_;
    C_input[_deg_basis_]    = degree_basis_stationary_cov_;
    C_input[_knots_basis_]  = knots_stationary_cov_;
    C_input[_coeff_basis_]  = toRList(coefficients_stationary_cov_,false);
    inputs_info[_cov_stat_] = C_input;
    //input of Beta C   
    Rcpp::List beta_C_input;
    beta_C_input[_n_basis_]     = number_basis_beta_stationary_cov_;
    beta_C_input[_t_basis_]     = basis_types_beta_stationary_cov_;
    beta_C_input[_deg_basis_]   = degree_basis_beta_stationary_cov_;
    beta_C_input[_knots_basis_] = knots_beta_stationary_cov_;
    inputs_info[_beta_stat_]    = beta_C_input;
    //input of E
    Rcpp::List E_input;
    E_input[_q_]             = q_E;
    E_input[_n_basis_]       = number_basis_events_cov_;
    E_input[_t_basis_]       = basis_types_events_cov_;
    E_input[_deg_basis_]     = degree_basis_events_cov_;
    E_input[_knots_basis_]   = knots_events_cov_;
    E_input[_coeff_basis_]   = toRList(coefficients_events_cov_,false);
    E_input[_penalties_]     = lambda_events_cov_;
    E_input[_coords_]        = Rcpp::wrap(coordinates_events_out_);
    E_input[_bdw_ker_]       = kernel_bandwith_events;
    inputs_info[_cov_event_] = E_input;
    //input of Beta E   
    Rcpp::List beta_E_input;
    beta_E_input[_n_basis_]     = number_basis_beta_events_cov_;
    beta_E_input[_t_basis_]     = basis_types_beta_events_cov_;
    beta_E_input[_deg_basis_]   = degree_basis_beta_events_cov_;
    beta_E_input[_knots_basis_] = knots_beta_events_cov_;
    inputs_info[_beta_event_]   = beta_E_input;
    //input of S    
    Rcpp::List S_input;
    S_input[_q_]               = q_S;
    S_input[_n_basis_]         = number_basis_stations_cov_;
    S_input[_t_basis_]         = basis_types_stations_cov_;
    S_input[_deg_basis_]       = degree_basis_stations_cov_;
    S_input[_knots_basis_]     = knots_stations_cov_;
    S_input[_coeff_basis_]     = toRList(coefficients_stations_cov_,false);
    S_input[_penalties_]       = lambda_stations_cov_;
    S_input[_coords_]          = Rcpp::wrap(coordinates_stations_out_);
    S_input[_bdw_ker_]         = kernel_bandwith_stations;
    inputs_info[_cov_station_] = S_input;
    //input of Beta S
    Rcpp::List beta_S_input;
    beta_S_input[_n_basis_]     = number_basis_beta_stations_cov_;
    beta_S_input[_t_basis_]     = basis_types_beta_stations_cov_;
    beta_S_input[_deg_basis_]   = degree_basis_beta_stations_cov_;
    beta_S_input[_knots_basis_] = knots_beta_stations_cov_;
    inputs_info[_beta_station_] = beta_S_input;
    //domain
    inputs_info[_n_]           = number_of_statistical_units_;
    inputs_info[_a_]           = a;
    inputs_info[_b_]           = b;
    inputs_info[_abscissa_]    = abscissa_points_;
    inputs_info[_cascade_estimate_] = in_cascade_estimation;
    //adding all the elements of the training set to perform prediction
    elem_for_pred[_input_info_] = inputs_info;
    l[_elem_for_pred_]          = elem_for_pred;

    return l;
}


/*!
* @brief Function to perform predictions on new statistical units using a fitted Functional Multi-Source Geographically Weighted Regression ESC model. Non-stationary betas have to be recomputed in the new locations.
* @param coeff_stationary_cov_to_pred list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit to be predicted.
* @param coeff_events_cov_to_pred list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th event-dependent covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit to be predicted.
* @param coordinates_events_to_pred matrix of double containing the UTM coordinates of the event of new statistical units: each row represents a statistical unit to be predicted, each column a coordinate (2 columns).
* @param coeff_stations_cov_to_pred list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th station-dependent covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a new statistical unit.
* @param coordinates_stations_to_pred matrix of double containing the UTM coordinates of the station of new statistical units: each row represents a statistical unit to be predicted, each column a coordinate (2 columns).
* @param units_to_be_predicted number of units to be predicted
* @param abscissa_ev abscissa for which then evaluating the predicted reponse and betas, stationary and non-stationary, which have to be recomputed
* @param model_fitted: output of FMSGWR_ESC: an R list containing:
*         - "FGWR": string containing the type of fwr used ("FMSGWR_ESC")
*         - "EstimationTechnique": "Exact" if in_cascade_estimation false, "Cascade" if in_cascade_estimation true 
*         - "Bc": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "basis_coeff": a Lc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective element of basis_types_beta_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stationary_cov)
*         - "Beta_c": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Beta_eval": a vector of double containing the discrete evaluations of the stationary beta
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "Be": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Le_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_events_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_events_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_events_cov)
*         - "Beta_e": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)  
*         - "Bs": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Ls_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_stations_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stations_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stations_cov)
*         - "Beta_s": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "predictor_info": a list containing partial residuals and information of the fitted model to perform predictions for new statistical units:
*                             - "partial_res": a list containing information to compute the partial residuals:
*                                              - "c_tilde_hat": vector of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon (if in_cascade_estimation is true, contains only 0s).
*                                              - "A__": vector of matrices with the operator A_e for each statistical unit (if in_cascade_estimation is true, each matrix contains only 0s).
*                                              - "B__for_K": vector of matrices with the operator B_e used for the K_e_s(t) computation, for each statistical unit (if in_cascade_estimation is true, each matrix contains only 0s).
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response (element n_basis_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response. Possible values: "bsplines", "constant". (element basis_type_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response (element degree_basis_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response (element coeff_y_points).
*                                              - "ResponseReconstructionWeights": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response reconstruction weights (element n_basis_rec_weights_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response reconstruction weights. Possible values: "bsplines", "constant". (element basis_type_rec_weights_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response reconstruction weights (element degree_basis_rec_weights_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response reconstruction weights (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response reconstruction weights (element coeff_rec_weights_y_points).
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates": number of stationary covariates (length of coeff_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional stationary covariates (respective elements of n_basis_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional stationary covariates (respective elements of degrees_basis_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional stationary covariates (element knots_stationary_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional stationary covariates (respective elements of coeff_stationary_cov).
*                                              - "beta_Stationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element n_basis_beta_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element degrees_basis_beta_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the stationary covariates (element knots_beta_stationary_cov).                                                            
*                                              - "cov_Event": list:
*                                                            - "number_covariates": number of event-dependent covariates (length of coeff_events_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional event-dependent covariates (respective elements of n_basis_events_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional event-dependent covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_events_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional event-dependent covariates (respective elements of degrees_basis_events_cov).
*                                                            - "knots": knots used to make the basis expansion of functional event-dependent covariates (element knots_events_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional event-dependent covariates (respective elements of coeff_events_cov).
*                                                            - "penalizations": vector containing the penalizations of the event-dependent covariates (respective elements of penalization_events_cov)
*                                                            - "coordinates": UTM coordinates of the events of the training data (element coordinates_events).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the events (element kernel_bandwith_events).
*                                              - "beta_Event": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element n_basis_beta_events_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates. Possible values: "bsplines", "constant". (element basis_types_beta_events_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element degrees_basis_beta_events_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element knots_beta_events_cov).
*                                              - "cov_Station": list:
*                                                            - "number_covariates": number of station-dependent covariates (length of coeff_stations_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional station-dependent covariates (respective elements of n_basis_stations_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional station-dependent covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stations_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional station-dependent covariates (respective elements of degrees_basis_stations_cov).
*                                                            - "knots": knots used to make the basis expansion of functional station-dependent covariates (element knots_stations_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional station-dependent covariates (respective elements of coeff_stations_cov).
*                                                            - "penalizations": vector containing the penalizations of the station-dependent covariates (respective elements of penalization_stations_cov)
*                                                            - "coordinates": UTM coordinates of the stations of the training data (element coordinates_stations).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the stations (element kernel_bandwith_stations).
*                                              - "beta_Station": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element n_basis_beta_stations_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stations_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element degrees_basis_beta_stations_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element knots_beta_stations_cov).
*                                              - "a": domain left extreme  (element left_extreme_domain).
*                                              - "b": domain right extreme (element right_extreme_domain).
*                                              - "abscissa": abscissa for which the evaluations of the functional data are available (element t_points).
*                                              - "InCascadeEstimation": element in_cascade_estimation.
* @param n_knots_smoothing_pred number of knots used to smooth predicted response and non-stationary, obtaining basis expansion coefficients with respect to the training basis (default: 100).
* @param n_intervals_quadrature number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: 100).
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine.
* @return an R list containing:
*         - "FGWR_predictor": string containing the model used to predict ("predictor_FMSGWR_ESC")
*         - "EstimationTechnique": "Exact" if in_cascade_estimation in the fitted model false, "Cascade" if in_cascade_estimation in the fitted model true 
*         - "prediction": list containing:
*                         - "evaluation": list containing the evaluation of the prediction:
*                                          - "prediction_ev": list containing, for each unit to be predicted, the raw evaluations of the predicted response.
*                                          - "abscissa_ev": the abscissa points for which the prediction evaluation is available (element abscissa_ev).
*                         - "fd": list containing the prediction functional description:
*                                          - "prediction_basis_coeff": matrix containing the prediction basis expansion coefficients (each row a basis, each column a new statistical unit)
*                                          - "prediction_basis_type": basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_basis_num": number of basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_basis_deg": degree of basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_knots": knots used for the predicted response smoothing (n_knots_smoothing_pred equally spaced knots in the functional datum domain)
*         - "Bc_pred": list containing, for each stationary covariate:
*                      - "basis_coeff": coefficients of the basis expansion of the beta (from model_fitted).
*                      - "basis_num": number of basis used for the beta basis epxnasion (from model_fitted).
*                      - "basis_type": type of basis used for the beta basis expansion (from model_fitted).
*                      - "knots": knots used for the beta basis expansion (from model_fitted).
*         - "Beta_c_pred": list containing, for each stationary covariate:
*                           - "Beta_eval": evaluation of the beta along a grid.
*                           - "Abscissa": grid (element abscissa_ev).
*         - "Be_pred": list containing, for each event-dependent covariate:
*                      - "basis_coeff": list, one element for each unit to be predicted, with the recomputed coefficients of the basis expansion of the beta.
*                      - "basis_num": number of basis used for the beta basis expansion (from model_fitted).
*                      - "basis_type": type of basis used for the beta basis expansion (from model_fitted).
*                      - "knots": knots used for the beta basis expansion (from model_fitted).
*         - "Beta_e_pred": list containing, for each event-dependent covariate:
*                           - "Beta_eval": list containing, for each unit to be predicted, the evaluation of the beta along a grid.
*                           - "Abscissa": grid (element abscissa_ev).
*         - "Bs_pred": list containing, for each station-dependent covariate:
*                      - "basis_coeff": list, one element for each unit to be predicted, with the recomputed coefficients of the basis expansion of the beta.
*                      - "basis_num": number of basis used for the beta basis expansion (from model_fitted).
*                      - "basis_type": type of basis used for the beta basis expansion (from model_fitted).
*                      - "knots": knots used for the beta basis expansion (from model_fitted).
*         - "Beta_s_pred": list containing, for each station-dependent covariate:
*                           - "Beta_eval": list containing, for each unit to be predicted, the evaluation of the beta along a grid.
*                           - "Abscissa": grid (element abscissa_ev).
* @details NB: Covariates of units to be predicted have to be sampled in the same sample points for which the training data have been (t_points).
*              Covariates basis expansion for the units to be predicted has to be done with respect to the basis used for the covariates in the training set
*/
//
// [[Rcpp::export]]
Rcpp::List predict_FMSGWR_ESC(Rcpp::List coeff_stationary_cov_to_pred,
                              Rcpp::List coeff_events_cov_to_pred,
                              Rcpp::NumericMatrix coordinates_events_to_pred,   
                              Rcpp::List coeff_stations_cov_to_pred,
                              Rcpp::NumericMatrix coordinates_stations_to_pred,
                              int units_to_be_predicted,
                              Rcpp::NumericVector abscissa_ev,
                              Rcpp::List model_fitted,
                              int n_knots_smoothing_pred = 100,
                              int n_intervals_quadrature = 100,
                              Rcpp::Nullable<int> num_threads = R_NilValue)
{
    Rcout << "Functional Multi-Source Geographically Weighted Regression ESC predictor" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT


    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FMSGWR_ESC_;                         //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _EVENT_ = FDAGWR_COVARIATES_TYPES::EVENT;                        //enum for event covariates
    constexpr auto _STATION_ = FDAGWR_COVARIATES_TYPES::STATION;                    //enum for station covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    if(units_to_be_predicted <= 0){ Rcout << "Number of unit to be predicted has to be a positive number" << std::endl;}
    //checking that the model_fitted contains a fit from FMSGWR_ESC
    wrap_predict_input<_FGWR_ALGO_>(model_fitted);
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF KNOTS TO PERFORM SMOOTHING ON THE RESPONSE WITHOUT THE NON-STATIONARY COMPONENTS
    int n_knots_smoothing_y_new = wrap_and_check_n_knots_smoothing(n_knots_smoothing_pred);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA MIDPOINT QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);



    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    // NAME OF THE LIST ELEMENT COMING FROM THE FITTING MODEL FUNCTION
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _estimation_iter_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter};     //Exact or Cascade estimation
    std::string _bc_                = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_            = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _be_                = std::string{FDAGWR_B_NAMES::be};                                 //be
    std::string _beta_e_            = std::string{FDAGWR_BETAS_NAMES::beta_e};                         //beta_e
    std::string _bs_                = std::string{FDAGWR_B_NAMES::bs};                                 //bs
    std::string _beta_s_            = std::string{FDAGWR_BETAS_NAMES::beta_s};                         //beta_s
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _partial_residuals_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res};               //partial residuals 
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                        //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                     //response reconstruction weights
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};   //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};   //stationary betas
    std::string _cov_event_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_EVENT_>()};        //event-dependent training covariates
    std::string _beta_event_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_EVENT_>()};        //event-dependent betas
    std::string _cov_station_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates
    std::string _beta_station_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates    
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations
    std::string _cascade_estimate_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cascade_estimate};         //if using in cascade-estimation


    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[_elem_for_pred_];
    //list with partial residuals
    Rcpp::List partial_residuals = fitted_model[_partial_residuals_];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[_input_info_];
    //list with elements of the response
    Rcpp::List response_input            = training_input[_response_];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input      = training_input[_response_rec_w_];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[_cov_stat_];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[_beta_stat_];
    //list with elements of events-dependent covariates
    Rcpp::List events_cov_input          = training_input[_cov_event_];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_events_cov_input     = training_input[_beta_event_];
    //list with elements of stations-dependent covariates
    Rcpp::List stations_cov_input        = training_input[_cov_station_];
    //list with elements of the beta of stations-dependent covariates
    Rcpp::List beta_stations_cov_input   = training_input[_beta_station_];

    //ESTIMATION TECHNIQUE
    bool in_cascade_estimation = training_input[_cascade_estimate_];
    //DOMAIN INFORMATION
    std::size_t n_train = training_input[_n_];
    _FD_INPUT_TYPE_ a   = training_input[_a_];
    _FD_INPUT_TYPE_ b   = training_input[_b_];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);     //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_    = training_input[_abscissa_];             //abscissa point for which the training data are discretized
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[_n_basis_];
    std::string basis_type_response_   = response_input[_t_basis_];
    std::size_t degree_basis_response_ = response_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[_coeff_basis_]); 
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[_n_basis_];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[_t_basis_];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[_coeff_basis_]);  
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_       = stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(stationary_cov_input[_coeff_basis_]);
    //EVENTS COV    
    std::size_t q_E                                   = events_cov_input[_q_];
    std::vector<std::size_t> number_basis_events_cov_ = events_cov_input[_n_basis_];
    std::vector<std::string> basis_types_events_cov_  = events_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_events_cov_ = events_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_events_cov_       = events_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_events_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_ = wrap_covariates_coefficients<_EVENT_>(events_cov_input[_coeff_basis_]);
    std::vector<double> lambda_events_cov_ = events_cov_input[_penalties_];
    auto coordinates_events_               = reader_data<_DATA_TYPE_,_NAN_REM_>(events_cov_input[_coords_]);     
    double kernel_bandwith_events_cov_     = events_cov_input[_bdw_ker_];
    //STATIONS COV  
    std::size_t q_S                                     = stations_cov_input[_q_];
    std::vector<std::size_t> number_basis_stations_cov_ = stations_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stations_cov_  = stations_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stations_cov_ = stations_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stations_cov_       = stations_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stations_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_ = wrap_covariates_coefficients<_STATION_>(stations_cov_input[_coeff_basis_]);
    std::vector<double> lambda_stations_cov_ = stations_cov_input[_penalties_];
    auto coordinates_stations_               = reader_data<_DATA_TYPE_,_NAN_REM_>(stations_cov_input[_coords_]);
    double kernel_bandwith_stations_cov_     = stations_cov_input[_bdw_ker_];    
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[_bc_];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[_coeff_basis_]);  //Lc_j x 1
        Bc.push_back(Bc_i);}
    //EVENTS BETAS  
    std::vector<std::size_t> number_basis_beta_events_cov_ = beta_events_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_events_cov_  = beta_events_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = beta_events_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_events_cov_ = beta_events_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_events_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size()); 
    //STATIONS BETAS    
    std::vector<std::size_t> number_basis_beta_stations_cov_ = beta_stations_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stations_cov_  = beta_stations_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = beta_stations_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stations_cov_ = beta_stations_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stations_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());
    //saving the betas basis expansion coefficients for station-dependent covariates
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix>> Bs; //vettore esterno: per ogni covariata S. Interno: per ogni unità di training
    Bs.reserve(q_S);
    Rcpp::List Bs_list = model_fitted[_bs_];
    for(std::size_t i = 0; i < q_S; ++i){
        Rcpp::List Bs_i_list = Bs_list[i];
        auto Bs_i = wrap_covariates_coefficients<_STATION_>(Bs_i_list[_coeff_basis_]);  //Ls_j x 1
        Bs.push_back(Bs_i);}
    //PARTIAL RESIDUALS
    auto c_tilde_hat = reader_data<_DATA_TYPE_,_NAN_REM_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat}]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> A_E_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_A__}]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> B_E_for_K_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_B__for_K}]);


    ////////////////////////////////////////
    /////   TRAINING OBJECT CREATION   /////
    ////////////////////////////////////////
    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    //events (Theta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_E(knots_beta_events_cov_eigen_w_, 
                                                   degree_basis_beta_events_cov_, 
                                                   number_basis_beta_events_cov_, 
                                                   q_E);
    //stations (Psi)
    basis_systems< _DOMAIN_, bsplines_basis > bs_S(knots_beta_stations_cov_eigen_w_,  
                                                   degree_basis_beta_stations_cov_, 
                                                   number_basis_beta_stations_cov_, 
                                                   q_S);


    //PENALIZATION MATRICES                                               
    //events
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_E(std::move(bs_E),lambda_events_cov_);
    std::size_t Le = R_E.L();
    std::vector<std::size_t> Le_j = R_E.Lj();
    //stations
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_S(std::move(bs_S),lambda_stations_cov_);
    std::size_t Ls = R_S.L();
    std::vector<std::size_t> Ls_j = R_S.Lj();
    
    //additional info stationary
    std::size_t Lc = std::reduce(number_basis_beta_stationary_cov_.cbegin(),number_basis_beta_stationary_cov_.cend(),static_cast<std::size_t>(0));
    std::vector<std::size_t> Lc_j = number_basis_beta_stationary_cov_;


    //MODEL FITTED COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y_train_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_y_train_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_train_(std::move(coefficients_response_),std::move(basis_y_train_));
    //sttaionary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_train_(coefficients_stationary_cov_,
                                                                    q_C,
                                                                    basis_types_stationary_cov_,
                                                                    degree_basis_stationary_cov_,
                                                                    number_basis_stationary_cov_,
                                                                    knots_stationary_cov_eigen_w_,
                                                                    basis_fac);
    //events covariates
    functional_data_covariates<_DOMAIN_,_EVENT_> x_E_fd_train_(coefficients_events_cov_,
                                                               q_E,
                                                               basis_types_events_cov_,
                                                               degree_basis_events_cov_,
                                                               number_basis_events_cov_,
                                                               knots_events_cov_eigen_w_,
                                                               basis_fac);
    
    //stations covariates
    functional_data_covariates<_DOMAIN_,_STATION_> x_S_fd_train_(coefficients_stations_cov_,
                                                                 q_S,
                                                                 basis_types_stations_cov_,
                                                                 degree_basis_stations_cov_,
                                                                 number_basis_stations_cov_,
                                                                 knots_stations_cov_eigen_w_,
                                                                 basis_fac);


    //wrapping all the functional elements in a functional_matrix
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> theta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_E);
    //psi: a sparse functional matrix of dimension qsxLs
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> psi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_S);
    //phi: a sparse functional matrix n_trainx(n_train*Ly), where L is the number of basis for the response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >; 
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> phi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(*basis_response_,n_train,number_basis_response_);
    //y_train: a column vector of dimension n_trainx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_train_,number_threads);
    //Xc_train: a functional matrix of dimension n_trainxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_train_,number_threads);
    //Xe_train: a functional matrix of dimension n_trainxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xe_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_EVENT_>(x_E_fd_train_,number_threads);
    //Xs_train: a functional matrix of dimension n_trainxqs
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xs_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATION_>(x_S_fd_train_,number_threads);


    //////////////////////////////////////////////
    ///// WRAPPING COVARIATES TO BE PREDICTED ////
    //////////////////////////////////////////////
    // stationary
    //covariates names
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_to_be_pred_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov_to_pred); 
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(units_to_be_predicted,coefficients_stationary_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //events
    //covariates names
    std::vector<std::string> names_events_cov_ = wrap_covariates_names<_EVENT_>(coeff_events_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_to_be_pred_ = wrap_covariates_coefficients<_EVENT_>(coeff_events_cov_to_pred); 
    for(std::size_t i = 0; i < q_E; ++i){   
        check_dim_input<_EVENT_>(number_basis_events_cov_[i],coefficients_events_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_EVENT_>(units_to_be_predicted,coefficients_events_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //stations
    //covariates names
    std::vector<std::string> names_stations_cov_ = wrap_covariates_names<_STATION_>(coeff_stations_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_to_be_pred_ = wrap_covariates_coefficients<_STATION_>(coeff_stations_cov_to_pred);
    for(std::size_t i = 0; i < q_S; ++i){   
        check_dim_input<_STATION_>(number_basis_stations_cov_[i],coefficients_stations_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATION_>(units_to_be_predicted,coefficients_stations_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    
    //TO BE PREDICTED COVARIATES  
    //stationary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_to_be_pred_(coefficients_stationary_cov_to_be_pred_,
                                                                         q_C,
                                                                         basis_types_stationary_cov_,
                                                                         degree_basis_stationary_cov_,
                                                                         number_basis_stationary_cov_,
                                                                         knots_stationary_cov_eigen_w_,
                                                                         basis_fac);
    //events covariates
    functional_data_covariates<_DOMAIN_,_EVENT_>   x_E_fd_to_be_pred_(coefficients_events_cov_to_be_pred_,
                                                                      q_E,
                                                                      basis_types_events_cov_,
                                                                      degree_basis_events_cov_,
                                                                      number_basis_events_cov_,
                                                                      knots_events_cov_eigen_w_,
                                                                      basis_fac);
    //stations covariates
    functional_data_covariates<_DOMAIN_,_STATION_> x_S_fd_to_be_pred_(coefficients_stations_cov_to_be_pred_,
                                                                      q_S,
                                                                      basis_types_stations_cov_,
                                                                      degree_basis_stations_cov_,
                                                                      number_basis_stations_cov_,
                                                                      knots_stations_cov_eigen_w_,
                                                                      basis_fac);
    //Xc_new: a functional matrix of dimension n_newxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_to_be_pred_,number_threads);                                                               
    //Xe_new: a functional matrix of dimension n_newxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xe_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_EVENT_>(x_E_fd_to_be_pred_,number_threads);
    //Xs_new: a functional matrix of dimension n_newxqs
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xs_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATION_>(x_S_fd_to_be_pred_,number_threads);
    //map containing the X
    std::map<std::string,functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>> X_new = {
        {std::string{covariate_type<_STATIONARY_>()},Xc_new},
        {std::string{covariate_type<_EVENT_>()},     Xe_new},
        {std::string{covariate_type<_STATION_>()},   Xs_new}};

    ////////////////////////////////////////
    /////////        CONSTRUCTING W   //////
    ////////////////////////////////////////
    //distances
    auto coordinates_events_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_events_to_pred);
    check_dim_input<_EVENT_>(units_to_be_predicted,coordinates_events_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_EVENT_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_events_to_pred_.cols(),"coordinates matrix columns");
    auto coordinates_stations_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_stations_to_pred);
    check_dim_input<_STATION_>(units_to_be_predicted,coordinates_stations_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_STATION_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_stations_to_pred_.cols(),"coordinates matrix columns");
    distance_matrix_pred<_DISTANCE_> distances_events_to_pred_(std::move(coordinates_events_),std::move(coordinates_events_to_pred_));
    distance_matrix_pred<_DISTANCE_> distances_stations_to_pred_(std::move(coordinates_stations_),std::move(coordinates_stations_to_pred_));
    distances_events_to_pred_.compute_distances();
    distances_stations_to_pred_.compute_distances();
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    //functional weight matrix
    //events
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_,_KERNEL_,_DISTANCE_> W_E_pred(rec_weights_y_fd_,
                                                                                                                                                                            std::move(distances_events_to_pred_),
                                                                                                                                                                            kernel_bandwith_events_cov_,
                                                                                                                                                                            number_threads,
                                                                                                                                                                            true);
    W_E_pred.compute_weights_pred();                                                                         
    //stations
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_,_KERNEL_,_DISTANCE_> W_S_pred(rec_weights_y_fd_,
                                                                                                                                                                              std::move(distances_stations_to_pred_),
                                                                                                                                                                              kernel_bandwith_stations_cov_,
                                                                                                                                                                              number_threads,
                                                                                                                                                                              true);
    W_S_pred.compute_weights_pred();
    //We_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > We_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_>(W_E_pred,number_threads);
    //Ws_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Ws_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_>(W_S_pred,number_threads);
    //map containing the W
    std::map<std::string,std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>>> W_new = {
        {std::string{covariate_type<_EVENT_>()},  We_pred},
        {std::string{covariate_type<_STATION_>()},Ws_pred}};


    //fwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(Bc),
                                                                                                 std::move(Bs),
                                                                                                 std::move(omega),
                                                                                                 q_C,
                                                                                                 Lc,
                                                                                                 Lc_j,
                                                                                                 std::move(theta),
                                                                                                 q_E,
                                                                                                 Le,
                                                                                                 Le_j,
                                                                                                 std::move(psi),
                                                                                                 q_S,
                                                                                                 Ls,
                                                                                                 Ls_j,
                                                                                                 std::move(phi),
                                                                                                 number_basis_response_,
                                                                                                 std::move(c_tilde_hat),
                                                                                                 std::move(A_E_i),
                                                                                                 std::move(B_E_for_K_i),
                                                                                                 std::move(y_train),
                                                                                                 std::move(Xc_train),
                                                                                                 std::move(Xe_train),
                                                                                                 std::move(R_E.PenalizationMatrix()),
                                                                                                 std::move(Xs_train),
                                                                                                 std::move(R_S.PenalizationMatrix()),
                                                                                                 a,
                                                                                                 b,
                                                                                                 n_intervals,
                                                                                                 n_train,
                                                                                                 number_threads,
                                                                                                 in_cascade_estimation);

    Rcout << "Prediction" << std::endl;

    //retrieve partial residuals
    fwr_predictor->computePartialResiduals();
    //compute the new b for the non-stationary covariates
    fwr_predictor->computeBNew(W_new);
    //compute the beta for stationary covariates
    fwr_predictor->computeStationaryBetas();            
    //compute the beta for non-stationary covariates
    fwr_predictor->computeNonStationaryBetas();   
    //perform prediction
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_pred = fwr_predictor->predict(X_new);
    //evaluating the betas   
    fwr_predictor->evalBetas(abscissa_points_ev_);
    //evaluating the prediction
    std::vector< std::vector< _FD_OUTPUT_TYPE_>> y_pred_ev = fwr_predictor->evalPred(y_pred,abscissa_points_ev_);
    //smoothing of the prediction
    auto y_pred_smooth_coeff = fwr_predictor->smoothPred<_DOMAIN_>(y_pred,*basis_pred,knots_smoothing_pred);

    Rcout << "Prediction done" << std::endl;


    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fwr_predictor->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 names_events_cov_,
                                                 basis_types_beta_events_cov_,
                                                 number_basis_beta_events_cov_,
                                                 knots_beta_events_cov_,
                                                 names_stations_cov_,
                                                 basis_types_beta_stations_cov_,
                                                 number_basis_beta_stations_cov_,
                                                 knots_beta_stations_cov_);
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fwr_predictor->betas(),
                                           abscissa_points_ev_,
                                           names_stationary_cov_,
                                           {},
                                           names_events_cov_,
                                           names_stations_cov_);
    //predictions evaluations
    Rcpp::List y_pred_ev_R = wrap_prediction_to_R_list<_FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_>(y_pred_ev,
                                                                                          abscissa_points_ev_,
                                                                                          y_pred_smooth_coeff,
                                                                                          basis_type_response_,
                                                                                          number_basis_response_,
                                                                                          degree_basis_response_,
                                                                                          knots_smoothing_pred);

    //returning element                                       
    Rcpp::List l;
    //predictor
    l[_model_name_ + "_predictor"] = "predictor_" + std::string{algo_type<_FGWR_ALGO_>()};
    l[_estimation_iter_]           = estimation_iter(in_cascade_estimation);
    //predictions
    l[std::string{FDAGWR_HELPERS_for_PRED_NAMES::pred}] = y_pred_ev_R;
    //stationary covariate basis expansion coefficients for beta_c
    l[_bc_ + "_pred"]     = b_coefficients[_bc_];
    //beta_c
    l[_beta_c_ + "_pred"] = betas[_beta_c_];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[_be_ + "_pred"]     = b_coefficients[_be_];
    //beta_e
    l[_beta_e_ + "_pred"] = betas[_beta_e_];
    //station-dependent covariate basis expansion coefficients for beta_s
    l[_bs_ + "_pred"]     = b_coefficients[_bs_];
    //beta_s
    l[_beta_s_ + "_pred"] = betas[_beta_s_];

    return l;
}





/*!
* @brief Fitting a Functional Multi-Source Geographically Weighted Regression SEC model. The covariates are functional objects, divided into
*        three categories: stationary covariates (C), constant over geographical space, event-dependent covariates (E), that vary depending on the spatial coordinates of the event, 
*        station-dependent covariates (S), that vary depending on the spatial coordinates of the stations that measure the event. Regression coefficients are estimated 
*        in the following order: C, E, S. The functional response is already reconstructed according to the method proposed by Bortolotti et Al. (2024) (link below)
* @param y_points matrix of double containing the raw response: each row represents a specific abscissa for which the response evaluation is available, each column a statistical unit. Response is a already reconstructed.
* @param t_points vector of double with the abscissa points with respect of the raw evaluations of y_points are available (length of t_points is equal to the number of rows of y_points).
* @param left_extreme_domain double indicating the left extreme of the functional data domain (not necessarily the smaller element in t_points).
* @param right_extreme_domain double indicating the right extreme of the functional data domain (not necessarily the biggest element in t_points).
* @param coeff_y_points matrix of double containing the coefficient of response's basis expansion: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param knots_y_points vector of double with the abscissa points with respect which the basis expansions of the response and response reconstruction weights are performed (all elements contained in [a,b]). 
* @param degree_basis_y_points non-negative integer: the degree of the basis used for the basis expansion of the (functional) response. Default explained below (can be NULL).
* @param n_basis_y_points positive integer: number of basis for the basis expansion of the (functional) response. It must match number of rows of coeff_y_points. Default explained below (can be NULL).
* @param coeff_rec_weights_y_points matrix of double containing the coefficients of the basis expansion of the weights to reconstruct the (functional) response: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param degree_basis_rec_weights_y_points non-negative integer: the degree of the basis used for response reconstruction weights. Default explained below (can be NULL).
* @param n_basis_rec_weights_y_points positive integer: number of basis for the basis expansion of response reconstruction weights. It must match number of rows of coeff_rec_weights_y_points. Default explained below (can be NULL).
* @param coeff_stationary_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                             The name of the i-th element is the name of the i-th stationary covariate (default: "reg.Ci" if no name present).
* @param basis_types_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariate basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate. Default explained below (can be NULL).
* @param n_basis_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate. It must match number of rows of the i-th element of coeff_stationary_cov. Default explained below (can be NULL).
* @param penalization_stationary_cov vector of non-negative double: element i-th is the penalization used for the i-th stationary covariate.
* @param knots_beta_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates functional regression coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate functional regression coefficients. Default explained below (can be NULL).
* @param n_basis_beta_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate functional regression coefficients. Default explained below (can be NULL).
* @param coeff_events_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th events-dependent covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                         The name of the i-th element is the name of the i-th events-dependent covariate (default: "reg.Ei" if no name present).
* @param basis_types_events_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariate basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_events_cov vector of double with the abscissa points with respect which the basis expansions of the events-dependent covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_events_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th events-dependent covariate. Default explained below (can be NULL).
* @param n_basis_events_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th events-dependent covariate. It must match number of rows of the i-th element of coeff_events_cov. Default explained below (can be NULL).
* @param penalization_events_cov vector of non-negative double: element i-th is the penalization used for the i-th events-dependent covariate.
* @param coordinates_events matrix of double containing the UTM coordinates of the event of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
* @param kernel_bandwith_events positive double indicating the bandwith of the gaussian kernel used to smooth the distances within events.
* @param knots_beta_events_cov vector of double with the abscissa points with respect which the basis expansions of the events-dependent covariates functional regression coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_events_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th events-dependent covariate functional regression coefficient. Default explained below (can be NULL).
* @param n_basis_beta_events_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th events-dependent covariate functional regression coefficient. Default explained below (can be NULL).
* @param coeff_stations_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stations-dependent covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                           The name of the i-th element is the name of the i-th stations-dependent covariate (default: "reg.Si").
* @param basis_types_stations_cov vector of strings, element i-th containing the type of basis used for the i-th stations-dependent covariates basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_stations_cov vector of double with the abscissa points with respect which the basis expansions of the stations-dependent covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_stations_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stations-dependent covariate. Default explained below (can be NULL).
* @param n_basis_stations_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stations-dependent covariate. It must match number of rows of the i-th element of coeff_stations_cov. Default explained below (can be NULL).
* @param penalization_stations_cov vector of non-negative double: element i-th is the penalization used for the i-th stations-dependent covariate.
* @param coordinates_stations matrix of double containing the UTM coordinates of the station of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
* @param kernel_bandwith_stations positive double indicating the bandwith of the gaussian kernel used to smooth the distances within stations.
* @param knots_beta_stations_cov vector of double with the abscissa points with respect which the basis expansions of the stations-dependent covariates functional regression coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_stations_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stations-dependent covariate functional regression coefficient. Default explained below (can be NULL).
* @param n_basis_beta_stations_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stations-dependent covariate functional regression coefficient. Default explained below (can be NULL).
* @param in_cascade_estimation bool: if false, an exact algorithm taking account for the interaction within non-stationary covariates is used to fit the model. Otherwise, the model is fitted in cascade. The first option is more precise, but way more computationally intensive.
* @param n_knots_smoothing number of knots used to perform the smoothing on the response obtained leaving out all the non-stationary components (default: 100).
* @param n_intervals_quadrature number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: 100).
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine.
* @param basis_type_y_points string containing the type of basis used for the functional response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_type_rec_weights_y_points string containing the type of basis used for the weights to reconstruct the functional response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariate functional regression coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_events_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariate functional regression coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stations_cov vector of strings, element i-th containing the type of basis used for the i-th stations-dependent covariate functional regression coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @return an R list containing:
*         - "FGWR": string containing the type of fwr used ("FMSGWR_SEC")
*         - "EstimationTechnique": "Exact" if in_cascade_estimation false, "Cascade" if in_cascade_estimation true 
*         - "Bc": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "basis_coeff": a Lc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective element of basis_types_beta_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stationary_cov)
*         - "Beta_c": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Beta_eval": a vector of double containing the discrete evaluations of the stationary beta
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "Be": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Le_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_events_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_events_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_events_cov)
*         - "Beta_e": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)  
*         - "Bs": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Ls_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_stations_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stations_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stations_cov)
*         - "Beta_s": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "predictor_info": a list containing partial residuals and information of the fitted model to perform predictions for new statistical units:
*                             - "partial_res": a list containing information to compute the partial residuals:
*                                              - "c_tilde_hat": vector of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon (if in_cascade_estimation is true, contains only 0s).
*                                              - "A__": vector of matrices with the operator A_s for each statistical unit (if in_cascade_estimation is true, each matrix contains only 0s).
*                                              - "B__for_K": vector of matrices with the operator B_s used for the K_s_e(t) computation, for each statistical unit (if in_cascade_estimation is true, each matrix contains only 0s).
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response (element n_basis_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response. Possible values: "bsplines", "constant". (element basis_type_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response (element degree_basis_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response (element coeff_y_points).
*                                              - "ResponseReconstructionWeights": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response reconstruction weights (element n_basis_rec_weights_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response reconstruction weights. Possible values: "bsplines", "constant". (element basis_type_rec_weights_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response reconstruction weights (element degree_basis_rec_weights_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response reconstruction weights (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response reconstruction weights (element coeff_rec_weights_y_points).
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates": number of stationary covariates (length of coeff_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional stationary covariates (respective elements of n_basis_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional stationary covariates (respective elements of degrees_basis_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional stationary covariates (element knots_stationary_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional stationary covariates (respective elements of coeff_stationary_cov).
*                                              - "beta_Stationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element n_basis_beta_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element degrees_basis_beta_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the stationary covariates (element knots_beta_stationary_cov).                                                            
*                                              - "cov_Event": list:
*                                                            - "number_covariates": number of event-dependent covariates (length of coeff_events_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional event-dependent covariates (respective elements of n_basis_events_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional event-dependent covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_events_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional event-dependent covariates (respective elements of degrees_basis_events_cov).
*                                                            - "knots": knots used to make the basis expansion of functional event-dependent covariates (element knots_events_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional event-dependent covariates (respective elements of coeff_events_cov).
*                                                            - "penalizations": vector containing the penalizations of the event-dependent covariates (respective elements of penalization_events_cov)
*                                                            - "coordinates": UTM coordinates of the events of the training data (element coordinates_events).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the events (element kernel_bandwith_events).
*                                              - "beta_Event": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element n_basis_beta_events_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates. Possible values: "bsplines", "constant". (element basis_types_beta_events_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element degrees_basis_beta_events_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element knots_beta_events_cov).
*                                              - "cov_Station": list:
*                                                            - "number_covariates": number of station-dependent covariates (length of coeff_stations_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional station-dependent covariates (respective elements of n_basis_stations_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional station-dependent covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stations_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional station-dependent covariates (respective elements of degrees_basis_stations_cov).
*                                                            - "knots": knots used to make the basis expansion of functional station-dependent covariates (element knots_stations_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional station-dependent covariates (respective elements of coeff_stations_cov).
*                                                            - "penalizations": vector containing the penalizations of the station-dependent covariates (respective elements of penalization_stations_cov)
*                                                            - "coordinates": UTM coordinates of the stations of the training data (element coordinates_stations).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the stations (element kernel_bandwith_stations).
*                                              - "beta_Station": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element n_basis_beta_stations_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stations_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element degrees_basis_beta_stations_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element knots_beta_stations_cov).
*                                              - "a": domain left extreme  (element left_extreme_domain).
*                                              - "b": domain right extreme (element right_extreme_domain).
*                                              - "abscissa": abscissa for which the evaluations of the functional data are available (element t_points).
*                                              - "InCascadeEstimation": element in_cascade_estimation.
* @details constant basis are used, for a covariate, if it resembles a scalar shape. It consists of a straight line with y-value equal to 1 all over the data domain.
*          Can be seen as a B-spline basis with degree 0, number of basis 1, using one knot, consequently having only one coefficient for the only basis for each statistical unit.
*          fdagwr sets all the feats accordingly if reads constant basis.
*          However, recall that the response is a functional datum, as the regressors coefficients. Since the package's basis variety could be hopefully enlarged in the future 
*          (for example, introducing Fourier basis for handling data that present periodical behaviors), the input parameters regarding basis types for response, response reconstruction
*          weights and regressors coefficients are left at the end of the input list, and defaulted as NULL. Consequently they will use a B-spline basis system, and should NOT use a constant basis,
*          Recall to perform externally the basis expansion before using the package, and afterwards passing basis types, degree and number and basis expansion coefficients and knots coherently
* @note a little excursion about degree and number of basis passed as input. For each specific covariate, or the response, if using B-spline basis, remember that number of knots = number of basis - degree + 1. 
*       By default, if passing NULL, fdagwr uses a cubic B-spline system of basis, the number of basis is computed coherently from the number of knots (that is the only mandatory input parameter).
*       Passing only the degree of the bsplines, the number of basis used will be set accordingly, and viceversa if passing only the number of basis. 
*       But, take care that the number of basis used has to match the number of rows of coefficients matrix (for EACH type of basis). If not, an exception is thrown. No problems arise if letting fdagwr defaulting the number of basis.
*       For response and response reconstruction weights, degree and number of basis consist of integer, and can be NULL. For all the regressors, and their coefficients, the inputs consist of vector of integers: 
*       if willing to pass a default parameter, all the vector has to be defaulted (if passing NULL, a vector with all 3 for the degrees is passed, for example)
* @link https://www.researchgate.net/publication/377251714_Weighted_Functional_Data_Analysis_for_the_Calibration_of_a_Ground_Motion_Model_in_Italy @endlink
*/
//
// [[Rcpp::export]]
Rcpp::List FMSGWR_SEC(Rcpp::NumericMatrix y_points,
                      Rcpp::NumericVector t_points,
                      double left_extreme_domain,
                      double right_extreme_domain,
                      Rcpp::NumericMatrix coeff_y_points,
                      Rcpp::NumericVector knots_y_points,
                      Rcpp::Nullable<int> degree_basis_y_points,
                      Rcpp::Nullable<int> n_basis_y_points,
                      Rcpp::NumericMatrix coeff_rec_weights_y_points,
                      Rcpp::Nullable<int> degree_basis_rec_weights_y_points,
                      Rcpp::Nullable<int> n_basis_rec_weights_y_points,
                      Rcpp::List coeff_stationary_cov,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_stationary_cov,
                      Rcpp::NumericVector knots_stationary_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_stationary_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_stationary_cov,
                      Rcpp::NumericVector penalization_stationary_cov,
                      Rcpp::NumericVector knots_beta_stationary_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_beta_stationary_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_stationary_cov,
                      Rcpp::List coeff_events_cov,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_events_cov,
                      Rcpp::NumericVector knots_events_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_events_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_events_cov,
                      Rcpp::NumericVector penalization_events_cov,
                      Rcpp::NumericMatrix coordinates_events,
                      double kernel_bandwith_events,
                      Rcpp::NumericVector knots_beta_events_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_beta_events_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_events_cov,
                      Rcpp::List coeff_stations_cov,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_stations_cov,
                      Rcpp::NumericVector knots_stations_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_stations_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_stations_cov,
                      Rcpp::NumericVector penalization_stations_cov,
                      Rcpp::NumericMatrix coordinates_stations,
                      double kernel_bandwith_stations,
                      Rcpp::NumericVector knots_beta_stations_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_beta_stations_cov,
                      Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_stations_cov,
                      bool in_cascade_estimation = false,                      
                      int n_knots_smoothing = 100,
                      int n_intervals_quadrature = 100,
                      Rcpp::Nullable<int> num_threads = R_NilValue,
                      std::string basis_type_y_points = "bsplines",
                      std::string basis_type_rec_weights_y_points = "bsplines",
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_events_cov = R_NilValue,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stations_cov = R_NilValue)
{
    Rcout << "Functional Multi-Source Geographically Weighted Regression SEC" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT

    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FMSGWR_SEC_;                         //fgwr type (estimating stationary -> event-dependent -> station-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _EVENT_ = FDAGWR_COVARIATES_TYPES::EVENT;                        //enum for event covariates
    constexpr auto _STATION_ = FDAGWR_COVARIATES_TYPES::STATION;                    //enum for station covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF KNOTS TO PERFORM SMOOTHING ON THE RESPONSE WITHOUT THE NON-STATIONARY COMPONENTS
    int n_knots_smoothing_y_new = wrap_and_check_n_knots_smoothing(n_knots_smoothing);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA MIDPOINT QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);


    //  RESPONSE
    //raw data
    auto response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(y_points);       //Eigen dense matrix type (auto is necessary )
    //number of statistical units
    std::size_t number_of_statistical_units_ = response_.cols();
    //coefficients matrix
    auto coefficients_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_y_points);
    auto coefficients_response_out_ = coefficients_response_;
    //c: a dense matrix of double (n*Ly) x 1 containing, one column below the other, the y basis expansion coefficients
    auto c = columnize_coeff_resp(coefficients_response_);
    //reconstruction weights coefficients matrix
    auto coefficients_rec_weights_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_rec_weights_y_points);
    auto coefficients_rec_weights_response_out_ = coefficients_rec_weights_response_;

    //  ABSCISSA POINTS of response
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = wrap_abscissas(t_points,left_extreme_domain,right_extreme_domain);
    // wrapper into eigen
    check_dim_input<_RESPONSE_>(response_.rows(), abscissa_points_.size(), "points for evaluation of raw data vector");   //check that size of abscissa points and number of evaluations of fd raw data coincide
    FDAGWR_TRAITS::Dense_Matrix abscissa_points_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(abscissa_points_.data(),abscissa_points_.size(),1);
    _FD_INPUT_TYPE_ a = left_extreme_domain;
    _FD_INPUT_TYPE_ b = right_extreme_domain;


    //  KNOTS (for basis expansion and for smoothing)
    //response
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = wrap_abscissas(knots_y_points,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    //stationary cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_ = wrap_abscissas(knots_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    //beta stationary cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = wrap_abscissas(knots_beta_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //events cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_events_cov_ = wrap_abscissas(knots_events_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_events_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    //beta events cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_events_cov_ = wrap_abscissas(knots_beta_events_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_events_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size());
    //stations cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stations_cov_ = wrap_abscissas(knots_stations_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_stations_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    //stations beta cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stations_cov_ = wrap_abscissas(knots_beta_stations_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_stations_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());
    //knots for performing smoothing (n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);


    //  COVARIATES names, coefficients and how many (q_), for every type
    //stationary 
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov);
    std::size_t q_C = names_stationary_cov_.size();    //number of stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov);    
    //events
    std::vector<std::string> names_events_cov_ = wrap_covariates_names<_EVENT_>(coeff_events_cov);
    std::size_t q_E = names_events_cov_.size();        //number of events related covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_ = wrap_covariates_coefficients<_EVENT_>(coeff_events_cov);
    //stations
    std::vector<std::string> names_stations_cov_ = wrap_covariates_names<_STATION_>(coeff_stations_cov);
    std::size_t q_S = names_stations_cov_.size();      //number of stations related covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_ = wrap_covariates_coefficients<_STATION_>(coeff_stations_cov);


    //  BASIS TYPES
    //response
    std::string basis_type_response_ = wrap_and_check_basis_type<_RESPONSE_>(basis_type_y_points);
    //response reconstruction weights
    std::string basis_type_rec_weights_response_ = wrap_and_check_basis_type<_REC_WEIGHTS_>(basis_type_rec_weights_y_points);
    //stationary
    std::vector<std::string> basis_types_stationary_cov_ = wrap_and_check_basis_type<_STATIONARY_>(basis_types_stationary_cov,q_C);
    //beta stationary cov 
    std::vector<std::string> basis_types_beta_stationary_cov_ = wrap_and_check_basis_type<_STATIONARY_>(basis_types_beta_stationary_cov,q_C);
    //events
    std::vector<std::string> basis_types_events_cov_ = wrap_and_check_basis_type<_EVENT_>(basis_types_events_cov,q_E);
    //beta events cov 
    std::vector<std::string> basis_types_beta_events_cov_ = wrap_and_check_basis_type<_EVENT_>(basis_types_beta_events_cov,q_E);
    //stations
    std::vector<std::string> basis_types_stations_cov_ = wrap_and_check_basis_type<_STATION_>(basis_types_stations_cov,q_S);
    //beta stations cov 
    std::vector<std::string> basis_types_beta_stations_cov_ = wrap_and_check_basis_type<_STATION_>(basis_types_beta_stations_cov,q_S);


    //  BASIS NUMBER AND DEGREE: checking matrix coefficients dimensions: rows: number of basis; cols: number of statistical units
    //response
    auto number_and_degree_basis_response_ = wrap_and_check_basis_number_and_degree<_RESPONSE_>(n_basis_y_points,degree_basis_y_points,knots_response_.size(),basis_type_response_);
    std::size_t number_basis_response_ = number_and_degree_basis_response_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::size_t degree_basis_response_ = number_and_degree_basis_response_[std::string{FDAGWR_FEATS::degree_basis_string}];
    check_dim_input<_RESPONSE_>(number_basis_response_,coefficients_response_.rows(),"response coefficients matrix rows");
    check_dim_input<_RESPONSE_>(number_of_statistical_units_,coefficients_response_.cols(),"response coefficients matrix columns");     
    //response reconstruction weights
    auto number_and_degree_basis_rec_weights_response_ = wrap_and_check_basis_number_and_degree<_REC_WEIGHTS_>(n_basis_rec_weights_y_points,degree_basis_rec_weights_y_points,knots_response_.size(),basis_type_rec_weights_response_);
    std::size_t number_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::size_t degree_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[std::string{FDAGWR_FEATS::degree_basis_string}];
    check_dim_input<_REC_WEIGHTS_>(number_basis_rec_weights_response_,coefficients_rec_weights_response_.rows(),"response reconstruction weights coefficients matrix rows");
    check_dim_input<_REC_WEIGHTS_>(number_of_statistical_units_,coefficients_rec_weights_response_.cols(),"response reconstruction weights coefficients matrix columns");     
    //stationary cov
    auto number_and_degree_basis_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_stationary_cov,degrees_basis_stationary_cov,knots_stationary_cov_.size(),q_C,basis_types_stationary_cov_);
    std::vector<std::size_t> number_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stationary cov
    auto number_and_degree_basis_beta_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_beta_stationary_cov,degrees_basis_beta_stationary_cov,knots_beta_stationary_cov_.size(),q_C,basis_types_beta_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    //events cov    
    auto number_and_degree_basis_events_cov_ = wrap_and_check_basis_number_and_degree<_EVENT_>(n_basis_events_cov,degrees_basis_events_cov,knots_events_cov_.size(),q_E,basis_types_events_cov_);
    std::vector<std::size_t> number_basis_events_cov_ = number_and_degree_basis_events_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_events_cov_ = number_and_degree_basis_events_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    for(std::size_t i = 0; i < q_E; ++i){   
        check_dim_input<_EVENT_>(number_basis_events_cov_[i],coefficients_events_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_EVENT_>(number_of_statistical_units_,coefficients_events_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta events cov
    auto number_and_degree_basis_beta_events_cov_ = wrap_and_check_basis_number_and_degree<_EVENT_>(n_basis_beta_events_cov,degrees_basis_beta_events_cov,knots_beta_events_cov_.size(),q_E,basis_types_beta_events_cov_);
    std::vector<std::size_t> number_basis_beta_events_cov_ = number_and_degree_basis_beta_events_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = number_and_degree_basis_beta_events_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    //stations cov
    auto number_and_degree_basis_stations_cov_ = wrap_and_check_basis_number_and_degree<_STATION_>(n_basis_stations_cov,degrees_basis_stations_cov,knots_stations_cov_.size(),q_S,basis_types_stations_cov_);
    std::vector<std::size_t> number_basis_stations_cov_ = number_and_degree_basis_stations_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_stations_cov_ = number_and_degree_basis_stations_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    for(std::size_t i = 0; i < q_S; ++i){   
        check_dim_input<_STATION_>(number_basis_stations_cov_[i],coefficients_stations_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATION_>(number_of_statistical_units_,coefficients_stations_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stations cov 
    auto number_and_degree_basis_beta_stations_cov_ = wrap_and_check_basis_number_and_degree<_STATION_>(n_basis_beta_stations_cov,degrees_basis_beta_stations_cov,knots_beta_stations_cov_.size(),q_S,basis_types_beta_stations_cov_);
    std::vector<std::size_t> number_basis_beta_stations_cov_ = number_and_degree_basis_beta_stations_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = number_and_degree_basis_beta_stations_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];


    //  DISTANCES
    //events    DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_events_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_events);
    auto coordinates_events_out_ = coordinates_events_;
    check_dim_input<_EVENT_>(number_of_statistical_units_,coordinates_events_.rows(),"coordinates matrix rows");
    check_dim_input<_EVENT_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_events_.cols(),"coordinates matrix columns");
    distance_matrix<_DISTANCE_> distances_events_cov_(std::move(coordinates_events_),number_threads);
    //stations  DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_stations_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_stations);
    auto coordinates_stations_out_ = coordinates_stations_;
    check_dim_input<_STATION_>(number_of_statistical_units_,coordinates_stations_.rows(),"coordinates matrix rows");
    check_dim_input<_STATION_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_stations_.cols(),"coordinates matrix columns");
    distance_matrix<_DISTANCE_> distances_stations_cov_(std::move(coordinates_stations_),number_threads);


    //  PENALIZATION TERMS: checking their consistency
    //stationary
    std::vector<double> lambda_stationary_cov_ = wrap_and_check_penalizations<_STATIONARY_>(penalization_stationary_cov,q_C);
    //events
    std::vector<double> lambda_events_cov_ = wrap_and_check_penalizations<_EVENT_>(penalization_events_cov,q_E);
    //stations
    std::vector<double> lambda_stations_cov_ = wrap_and_check_penalizations<_STATION_>(penalization_stations_cov,q_S);


    //  KERNEL BANDWITH
    //events
    double kernel_bandwith_events_cov_ = wrap_and_check_kernel_bandwith<_EVENT_>(kernel_bandwith_events);
    //stations
    double kernel_bandwith_stations_cov_ = wrap_and_check_kernel_bandwith<_STATION_>(kernel_bandwith_stations);

    ////////////////////////////////////////
    /////    END PARAMETERS WRAPPING   /////
    ////////////////////////////////////////



    ////////////////////////////////
    /////    OBJECT CREATION   /////
    ////////////////////////////////


    //DISTANCES
    //events
    distances_events_cov_.compute_distances();
    //stations
    distances_stations_cov_.compute_distances();


    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    //events (Theta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_E(knots_beta_events_cov_eigen_w_, 
                                                   degree_basis_beta_events_cov_, 
                                                   number_basis_beta_events_cov_, 
                                                   q_E);
    //stations (Psi)
    basis_systems< _DOMAIN_, bsplines_basis > bs_S(knots_beta_stations_cov_eigen_w_,  
                                                   degree_basis_beta_stations_cov_, 
                                                   number_basis_beta_stations_cov_, 
                                                   q_S);
    
    
    //PENALIZATION MATRICES
    //stationary
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_C(std::move(bs_C),lambda_stationary_cov_);
    std::size_t Lc = R_C.L();
    std::vector<std::size_t> Lc_j = R_C.Lj();
    //events
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_E(std::move(bs_E),lambda_events_cov_);
    std::size_t Le = R_E.L();
    std::vector<std::size_t> Le_j = R_E.Lj();
    //stations
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_S(std::move(bs_S),lambda_stations_cov_);
    std::size_t Ls = R_S.L();
    std::vector<std::size_t> Ls_j = R_S.Lj();


    //FD OBJECTS: RESPONSE and COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_(std::move(coefficients_response_),std::move(basis_response_));
    

    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    
    //stationary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_(coefficients_stationary_cov_,
                                                              q_C,
                                                              basis_types_stationary_cov_,
                                                              degree_basis_stationary_cov_,
                                                              number_basis_stationary_cov_,
                                                              knots_stationary_cov_eigen_w_,
                                                              basis_fac);
    
    //events covariates
    functional_data_covariates<_DOMAIN_,_EVENT_> x_E_fd_(coefficients_events_cov_,
                                                         q_E,
                                                         basis_types_events_cov_,
                                                         degree_basis_events_cov_,
                                                         number_basis_events_cov_,
                                                         knots_events_cov_eigen_w_,
                                                         basis_fac);
    
    //stations covariates
    functional_data_covariates<_DOMAIN_,_STATION_> x_S_fd_(coefficients_stations_cov_,
                                                           q_S,
                                                           basis_types_stations_cov_,
                                                           degree_basis_stations_cov_,
                                                           number_basis_stations_cov_,
                                                           knots_stations_cov_eigen_w_,
                                                           basis_fac);


    //FUNCTIONAL WEIGHT MATRIX
    //stationary
    functional_weight_matrix_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATIONARY_> W_C(rec_weights_y_fd_,
                                                                                                                                                    number_threads);
    W_C.compute_weights();                                                      
    //events
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_,_KERNEL_,_DISTANCE_> W_E(rec_weights_y_fd_,
                                                                                                                                                                       std::move(distances_events_cov_),
                                                                                                                                                                       kernel_bandwith_events_cov_,
                                                                                                                                                                       number_threads);
    W_E.compute_weights();                                                                         
    //stations
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_,_KERNEL_,_DISTANCE_> W_S(rec_weights_y_fd_,
                                                                                                                                                                         std::move(distances_stations_cov_),
                                                                                                                                                                         kernel_bandwith_stations_cov_,
                                                                                                                                                                         number_threads);
    W_S.compute_weights();


    ///////////////////////////////
    /////    FGWR ALGORITHM   /////
    ///////////////////////////////
    //wrapping all the functional elements in a functional_matrix

    //y: a column vector of dimension nx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_,number_threads);
    //phi: a sparse functional matrix nx(n*L), where L is the number of basis for the response
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> phi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_.fdata_basis(),number_of_statistical_units_,number_basis_response_);
    //c: a dense matrix of double (n*Ly) x 1 containing, one column below the other, the y basis expansion coefficients
    //already done at the beginning
    //basis used for doing response basis expansion
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //Xc: a functional matrix of dimension nxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_,number_threads);
    //Wc: a diagonal functional matrix of dimension nxn
    functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Wc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATIONARY_>(W_C,number_threads);
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);
    //Xe: a functional matrix of dimension nxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xe = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_EVENT_>(x_E_fd_,number_threads);
    //We: n diagonal functional matrices of dimension nxn
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > We = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_>(W_E,number_threads);
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> theta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_E);
    //Xs: a functional matrix of dimension nxqs
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xs = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATION_>(x_S_fd_,number_threads);
    //Ws: n diagonal functional matrices of dimension nxn
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Ws = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_>(W_S,number_threads);
    //psi: a sparse functional matrix of dimension qsxLs
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> psi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_S);


    //fwr algorithm
    auto fgwr_algo = fwr_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(y),
                                                                                   std::move(phi),
                                                                                   std::move(c),
                                                                                   number_basis_response_,
                                                                                   std::move(basis_y),
                                                                                   std::move(knots_smoothing),
                                                                                   std::move(Xc),
                                                                                   std::move(Wc),
                                                                                   std::move(R_C.PenalizationMatrix()),
                                                                                   std::move(omega),
                                                                                   q_C,
                                                                                   Lc,
                                                                                   Lc_j,
                                                                                   std::move(Xe),
                                                                                   std::move(We),
                                                                                   std::move(R_E.PenalizationMatrix()),
                                                                                   std::move(theta),
                                                                                   q_E,
                                                                                   Le,
                                                                                   Le_j,
                                                                                   std::move(Xs),
                                                                                   std::move(Ws),
                                                                                   std::move(R_S.PenalizationMatrix()),
                                                                                   std::move(psi),
                                                                                   q_S,
                                                                                   Ls,
                                                                                   Ls_j,
                                                                                   a,
                                                                                   b,
                                                                                   n_intervals,
                                                                                   abscissa_points_,
                                                                                   number_of_statistical_units_,
                                                                                   number_threads,
                                                                                   in_cascade_estimation);

    Rcout << "Model fitting" << std::endl;                                                                                    

    //computing the b
    fgwr_algo->compute();
    //evaluating the betas   
    fgwr_algo->evalBetas();

    Rcout << "Model fitted" << std::endl; 

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fgwr_algo->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 names_events_cov_,
                                                 basis_types_beta_events_cov_,
                                                 number_basis_beta_events_cov_,
                                                 knots_beta_events_cov_,
                                                 names_stations_cov_,
                                                 basis_types_beta_stations_cov_,
                                                 number_basis_beta_stations_cov_,
                                                 knots_beta_stations_cov_);
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fgwr_algo->betas(),
                                           abscissa_points_,
                                           names_stationary_cov_,
                                           {},
                                           names_events_cov_,
                                           names_stations_cov_);
    //elements for partial residuals
    Rcpp::List p_res = wrap_PRes_to_R_list(fgwr_algo->PRes());



    //returning element
    Rcpp::List l;
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _estimation_iter_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter};     //Exact or Cascade estimation
    std::string _bc_                = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_            = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _be_                = std::string{FDAGWR_B_NAMES::be};                                 //be
    std::string _beta_e_            = std::string{FDAGWR_BETAS_NAMES::beta_e};                         //beta_e
    std::string _bs_                = std::string{FDAGWR_B_NAMES::bs};                                 //bs
    std::string _beta_s_            = std::string{FDAGWR_BETAS_NAMES::beta_s};                         //beta_s
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _partial_residuals_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res};               //partial residuals 
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                        //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                     //response reconstruction weights
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};   //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};   //stationary betas
    std::string _cov_event_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_EVENT_>()};        //event-dependent training covariates
    std::string _beta_event_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_EVENT_>()};        //event-dependent betas
    std::string _cov_station_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates
    std::string _beta_station_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates    
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations
    std::string _cascade_estimate_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cascade_estimate};    //if using in cascade-estimation

    //regression model used 
    l[_model_name_] = std::string{algo_type<_FGWR_ALGO_>()};
    //estimation technique
    l[_estimation_iter_] = estimation_iter(in_cascade_estimation);
    //stationary covariate basis expansion coefficients for beta_c
    l[_bc_]  = b_coefficients[_bc_];
    //beta_c
    l[_beta_c_] = betas[_beta_c_];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[_be_]  = b_coefficients[_be_];
    //beta_e
    l[_beta_e_] = betas[_beta_e_];
    //station-dependent covariate basis expansion coefficients for beta_s
    l[_bs_]  = b_coefficients[_bs_];
    //beta_s
    l[_beta_s_] = betas[_beta_s_];
    //elements needed to perform prediction
    Rcpp::List elem_for_pred;       
    elem_for_pred[_partial_residuals_] = p_res;     //partial residuals, elements to reconstruct them
    Rcpp::List inputs_info;                         //containing training data information needed for prediction purposes

    //adding all the elements of the training set
    //input of y
    Rcpp::List response_input;
    response_input[_n_basis_]     = number_basis_response_;
    response_input[_t_basis_]     = basis_type_response_;
    response_input[_deg_basis_]   = degree_basis_response_;
    response_input[_knots_basis_] = knots_response_;
    response_input[_coeff_basis_] = Rcpp::wrap(coefficients_response_out_);
    inputs_info[_response_]       = response_input;
    //input of w for y  
    Rcpp::List response_rec_w_input;
    response_rec_w_input[_n_basis_]     = number_basis_rec_weights_response_;
    response_rec_w_input[_t_basis_]     = basis_type_rec_weights_response_;
    response_rec_w_input[_deg_basis_]   = degree_basis_rec_weights_response_;
    response_rec_w_input[_knots_basis_] = knots_response_;
    response_rec_w_input[_coeff_basis_] = Rcpp::wrap(coefficients_rec_weights_response_out_);
    inputs_info[_response_rec_w_]       = response_rec_w_input;
    //input of C
    Rcpp::List C_input;
    C_input[_q_]            = q_C;
    C_input[_n_basis_]      = number_basis_stationary_cov_;
    C_input[_t_basis_]      = basis_types_stationary_cov_;
    C_input[_deg_basis_]    = degree_basis_stationary_cov_;
    C_input[_knots_basis_]  = knots_stationary_cov_;
    C_input[_coeff_basis_]  = toRList(coefficients_stationary_cov_,false);
    inputs_info[_cov_stat_] = C_input;
    //input of Beta C   
    Rcpp::List beta_C_input;
    beta_C_input[_n_basis_]     = number_basis_beta_stationary_cov_;
    beta_C_input[_t_basis_]     = basis_types_beta_stationary_cov_;
    beta_C_input[_deg_basis_]   = degree_basis_beta_stationary_cov_;
    beta_C_input[_knots_basis_] = knots_beta_stationary_cov_;
    inputs_info[_beta_stat_]    = beta_C_input;
    //input of E
    Rcpp::List E_input;
    E_input[_q_]             = q_E;
    E_input[_n_basis_]       = number_basis_events_cov_;
    E_input[_t_basis_]       = basis_types_events_cov_;
    E_input[_deg_basis_]     = degree_basis_events_cov_;
    E_input[_knots_basis_]   = knots_events_cov_;
    E_input[_coeff_basis_]   = toRList(coefficients_events_cov_,false);
    E_input[_penalties_]     = lambda_events_cov_;
    E_input[_coords_]        = Rcpp::wrap(coordinates_events_out_);
    E_input[_bdw_ker_]       = kernel_bandwith_events;
    inputs_info[_cov_event_] = E_input;
    //input of Beta E   
    Rcpp::List beta_E_input;
    beta_E_input[_n_basis_]     = number_basis_beta_events_cov_;
    beta_E_input[_t_basis_]     = basis_types_beta_events_cov_;
    beta_E_input[_deg_basis_]   = degree_basis_beta_events_cov_;
    beta_E_input[_knots_basis_] = knots_beta_events_cov_;
    inputs_info[_beta_event_]   = beta_E_input;
    //input of S    
    Rcpp::List S_input;
    S_input[_q_]               = q_S;
    S_input[_n_basis_]         = number_basis_stations_cov_;
    S_input[_t_basis_]         = basis_types_stations_cov_;
    S_input[_deg_basis_]       = degree_basis_stations_cov_;
    S_input[_knots_basis_]     = knots_stations_cov_;
    S_input[_coeff_basis_]     = toRList(coefficients_stations_cov_,false);
    S_input[_penalties_]       = lambda_stations_cov_;
    S_input[_coords_]          = Rcpp::wrap(coordinates_stations_out_);
    S_input[_bdw_ker_]         = kernel_bandwith_stations;
    inputs_info[_cov_station_] = S_input;
    //input of Beta S
    Rcpp::List beta_S_input;
    beta_S_input[_n_basis_]     = number_basis_beta_stations_cov_;
    beta_S_input[_t_basis_]     = basis_types_beta_stations_cov_;
    beta_S_input[_deg_basis_]   = degree_basis_beta_stations_cov_;
    beta_S_input[_knots_basis_] = knots_beta_stations_cov_;
    inputs_info[_beta_station_] = beta_S_input;
    //domain
    inputs_info[_n_]              = number_of_statistical_units_;
    inputs_info[_a_]              = a;
    inputs_info[_b_]              = b;
    inputs_info[_abscissa_]         = abscissa_points_;
    inputs_info[_cascade_estimate_] = in_cascade_estimation;
    //adding all the elements of the training set to perform prediction
    elem_for_pred[_input_info_] = inputs_info;
    l[_elem_for_pred_]          = elem_for_pred;

    return l;
}


/*!
* @brief Function to perform predictions on new statistical units using a fitted Functional Multi-Source Geographically Weighted Regression SEC model. Non-stationary betas have to be recomputed in the new locations.
* @param coeff_stationary_cov_to_pred list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit to be predicted.
* @param coeff_events_cov_to_pred list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th event-dependent covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit to be predicted.
* @param coordinates_events_to_pred matrix of double containing the UTM coordinates of the event of new statistical units: each row represents a statistical unit to be predicted, each column a coordinate (2 columns).
* @param coeff_stations_cov_to_pred list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th station-dependent covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a new statistical unit.
* @param coordinates_stations_to_pred matrix of double containing the UTM coordinates of the station of new statistical units: each row represents a statistical unit to be predicted, each column a coordinate (2 columns).
* @param units_to_be_predicted number of units to be predicted
* @param abscissa_ev abscissa for which then evaluating the predicted reponse and betas, stationary and non-stationary, which have to be recomputed
* @param model_fitted: output of FMSGWR_SEC: an R list containing:
*         - "FGWR": string containing the type of fwr used ("FMSGWR_SEC")
*         - "EstimationTechnique": "Exact" if in_cascade_estimation false, "Cascade" if in_cascade_estimation true 
*         - "Bc": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "basis_coeff": a Lc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective element of basis_types_beta_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stationary_cov)
*         - "Beta_c": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Beta_eval": a vector of double containing the discrete evaluations of the stationary beta
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "Be": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Le_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_events_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_events_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_events_cov)
*         - "Beta_e": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)  
*         - "Bs": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Ls_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_stations_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stations_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stations_cov)
*         - "Beta_s": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "predictor_info": a list containing partial residuals and information of the fitted model to perform predictions for new statistical units:
*                             - "partial_res": a list containing information to compute the partial residuals:
*                                              - "c_tilde_hat": vector of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon (if in_cascade_estimation is true, contains only 0s).
*                                              - "A__": vector of matrices with the operator A_s for each statistical unit (if in_cascade_estimation is true, each matrix contains only 0s).
*                                              - "B__for_K": vector of matrices with the operator B_s used for the K_s_e(t) computation, for each statistical unit (if in_cascade_estimation is true, each matrix contains only 0s).
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response (element n_basis_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response. Possible values: "bsplines", "constant". (element basis_type_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response (element degree_basis_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response (element coeff_y_points).
*                                              - "ResponseReconstructionWeights": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response reconstruction weights (element n_basis_rec_weights_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response reconstruction weights. Possible values: "bsplines", "constant". (element basis_type_rec_weights_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response reconstruction weights (element degree_basis_rec_weights_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response reconstruction weights (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response reconstruction weights (element coeff_rec_weights_y_points).
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates": number of stationary covariates (length of coeff_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional stationary covariates (respective elements of n_basis_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional stationary covariates (respective elements of degrees_basis_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional stationary covariates (element knots_stationary_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional stationary covariates (respective elements of coeff_stationary_cov).
*                                              - "beta_Stationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element n_basis_beta_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element degrees_basis_beta_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the stationary covariates (element knots_beta_stationary_cov).                                                            
*                                              - "cov_Event": list:
*                                                            - "number_covariates": number of event-dependent covariates (length of coeff_events_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional event-dependent covariates (respective elements of n_basis_events_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional event-dependent covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_events_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional event-dependent covariates (respective elements of degrees_basis_events_cov).
*                                                            - "knots": knots used to make the basis expansion of functional event-dependent covariates (element knots_events_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional event-dependent covariates (respective elements of coeff_events_cov).
*                                                            - "penalizations": vector containing the penalizations of the event-dependent covariates (respective elements of penalization_events_cov)
*                                                            - "coordinates": UTM coordinates of the events of the training data (element coordinates_events).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the events (element kernel_bandwith_events).
*                                              - "beta_Event": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element n_basis_beta_events_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates. Possible values: "bsplines", "constant". (element basis_types_beta_events_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element degrees_basis_beta_events_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element knots_beta_events_cov).
*                                              - "cov_Station": list:
*                                                            - "number_covariates": number of station-dependent covariates (length of coeff_stations_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional station-dependent covariates (respective elements of n_basis_stations_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional station-dependent covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stations_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional station-dependent covariates (respective elements of degrees_basis_stations_cov).
*                                                            - "knots": knots used to make the basis expansion of functional station-dependent covariates (element knots_stations_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional station-dependent covariates (respective elements of coeff_stations_cov).
*                                                            - "penalizations": vector containing the penalizations of the station-dependent covariates (respective elements of penalization_stations_cov)
*                                                            - "coordinates": UTM coordinates of the stations of the training data (element coordinates_stations).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the stations (element kernel_bandwith_stations).
*                                              - "beta_Station": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element n_basis_beta_stations_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stations_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element degrees_basis_beta_stations_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element knots_beta_stations_cov).
*                                              - "a": domain left extreme  (element left_extreme_domain).
*                                              - "b": domain right extreme (element right_extreme_domain).
*                                              - "abscissa": abscissa for which the evaluations of the functional data are available (element t_points).
*                                              - "InCascadeEstimation": element in_cascade_estimation.
* @param n_knots_smoothing_pred number of knots used to smooth predicted response and non-stationary, obtaining basis expansion coefficients with respect to the training basis (default: 100).
* @param n_intervals_quadrature number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: 100).
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine.
* @return an R list containing:
*         - "FGWR_predictor": string containing the model used to predict ("predictor_FMSGWR_SEC")
*         - "EstimationTechnique": "Exact" if in_cascade_estimation in the fitted model false, "Cascade" if in_cascade_estimation in the fitted model true 
*         - "prediction": list containing:
*                         - "evaluation": list containing the evaluation of the prediction:
*                                          - "prediction_ev": list containing, for each unit to be predicted, the raw evaluations of the predicted response.
*                                          - "abscissa_ev": the abscissa points for which the prediction evaluation is available (element abscissa_ev).
*                         - "fd": list containing the prediction functional description:
*                                          - "prediction_basis_coeff": matrix containing the prediction basis expansion coefficients (each row a basis, each column a new statistical unit)
*                                          - "prediction_basis_type": basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_basis_num": number of basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_basis_deg": degree of basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_knots": knots used for the predicted response smoothing (n_knots_smoothing_pred equally spaced knots in the functional datum domain)
*         - "Bc_pred": list containing, for each stationary covariate:
*                      - "basis_coeff": coefficients of the basis expansion of the beta (from model_fitted).
*                      - "basis_num": number of basis used for the beta basis epxnasion (from model_fitted).
*                      - "basis_type": type of basis used for the beta basis expansion (from model_fitted).
*                      - "knots": knots used for the beta basis expansion (from model_fitted).
*         - "Beta_c_pred": list containing, for each stationary covariate:
*                           - "Beta_eval": evaluation of the beta along a grid.
*                           - "Abscissa": grid (element abscissa_ev).
*         - "Be_pred": list containing, for each event-dependent covariate:
*                      - "basis_coeff": list, one element for each unit to be predicted, with the recomputed coefficients of the basis expansion of the beta.
*                      - "basis_num": number of basis used for the beta basis expansion (from model_fitted).
*                      - "basis_type": type of basis used for the beta basis expansion (from model_fitted).
*                      - "knots": knots used for the beta basis expansion (from model_fitted).
*         - "Beta_e_pred": list containing, for each event-dependent covariate:
*                           - "Beta_eval": list containing, for each unit to be predicted, the evaluation of the beta along a grid.
*                           - "Abscissa": grid (element abscissa_ev).
*         - "Bs_pred": list containing, for each station-dependent covariate:
*                      - "basis_coeff": list, one element for each unit to be predicted, with the recomputed coefficients of the basis expansion of the beta.
*                      - "basis_num": number of basis used for the beta basis expansion (from model_fitted).
*                      - "basis_type": type of basis used for the beta basis expansion (from model_fitted).
*                      - "knots": knots used for the beta basis expansion (from model_fitted).
*         - "Beta_s_pred": list containing, for each station-dependent covariate:
*                           - "Beta_eval": list containing, for each unit to be predicted, the evaluation of the beta along a grid.
*                           - "Abscissa": grid (element abscissa_ev).
* @details NB: Covariates of units to be predicted have to be sampled in the same sample points for which the training data have been (t_points).
*              Covariates basis expansion for the units to be predicted has to be done with respect to the basis used for the covariates in the training set
*/
//
// [[Rcpp::export]]
Rcpp::List predict_FMSGWR_SEC(Rcpp::List coeff_stationary_cov_to_pred,
                              Rcpp::List coeff_events_cov_to_pred,
                              Rcpp::NumericMatrix coordinates_events_to_pred,   
                              Rcpp::List coeff_stations_cov_to_pred,
                              Rcpp::NumericMatrix coordinates_stations_to_pred,
                              int units_to_be_predicted,
                              Rcpp::NumericVector abscissa_ev,
                              Rcpp::List model_fitted,
                              int n_knots_smoothing_pred = 100,                              
                              int n_intervals_quadrature = 100,
                              Rcpp::Nullable<int> num_threads = R_NilValue)
{
    Rcout << "Functional Multi-Source Geographically Weighted Regression SEC predictor" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT


    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FMSGWR_SEC_;                         //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _EVENT_ = FDAGWR_COVARIATES_TYPES::EVENT;                        //enum for event covariates
    constexpr auto _STATION_ = FDAGWR_COVARIATES_TYPES::STATION;                    //enum for station covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    if(units_to_be_predicted <= 0){ Rcout << "Number of unit to be predicted has to be a positive number" << std::endl;}
    //checking that the model_fitted contains a fit from FMSGWR_ESC
    wrap_predict_input<_FGWR_ALGO_>(model_fitted);
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF KNOTS TO PERFORM SMOOTHING ON THE RESPONSE WITHOUT THE NON-STATIONARY COMPONENTS
    int n_knots_smoothing_y_new = wrap_and_check_n_knots_smoothing(n_knots_smoothing_pred);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA MIDPOINT QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);


    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    // NAME OF THE LIST ELEMENT COMING FROM THE FITTING MODEL FUNCTION
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _estimation_iter_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter};     //Exact or Cascade estimation
    std::string _bc_                = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_            = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _be_                = std::string{FDAGWR_B_NAMES::be};                                 //be
    std::string _beta_e_            = std::string{FDAGWR_BETAS_NAMES::beta_e};                         //beta_e
    std::string _bs_                = std::string{FDAGWR_B_NAMES::bs};                                 //bs
    std::string _beta_s_            = std::string{FDAGWR_BETAS_NAMES::beta_s};                         //beta_s
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _partial_residuals_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res};               //partial residuals 
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                        //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                     //response reconstruction weights
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};   //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};   //stationary betas
    std::string _cov_event_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_EVENT_>()};        //event-dependent training covariates
    std::string _beta_event_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_EVENT_>()};        //event-dependent betas
    std::string _cov_station_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates
    std::string _beta_station_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates    
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations
    std::string _cascade_estimate_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cascade_estimate};    //if using in cascade-estimation



    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[_elem_for_pred_];
    //list with partial residuals
    Rcpp::List partial_residuals = fitted_model[_partial_residuals_];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[_input_info_];
    //list with elements of the response
    Rcpp::List response_input            = training_input[_response_];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input      = training_input[_response_rec_w_];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[_cov_stat_];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[_beta_stat_];
    //list with elements of events-dependent covariates
    Rcpp::List events_cov_input          = training_input[_cov_event_];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_events_cov_input     = training_input[_beta_event_];
    //list with elements of stations-dependent covariates
    Rcpp::List stations_cov_input        = training_input[_cov_station_];
    //list with elements of the beta of stations-dependent covariates
    Rcpp::List beta_stations_cov_input   = training_input[_beta_station_];

    //ESTIMATION TECHNIQUE
    bool in_cascade_estimation = training_input[_cascade_estimate_];
    //DOMAIN INFORMATION
    std::size_t n_train = training_input[_n_];
    _FD_INPUT_TYPE_ a   = training_input[_a_];
    _FD_INPUT_TYPE_ b   = training_input[_b_];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);      //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[_abscissa_];              //abscissa point for which the training data are discretized
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[_n_basis_];
    std::string basis_type_response_   = response_input[_t_basis_];
    std::size_t degree_basis_response_ = response_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[_coeff_basis_]); 
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[_n_basis_];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[_t_basis_];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[_coeff_basis_]);  
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_       = stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(stationary_cov_input[_coeff_basis_]);
    //EVENTS COV    
    std::size_t q_E                                   = events_cov_input[_q_];
    std::vector<std::size_t> number_basis_events_cov_ = events_cov_input[_n_basis_];
    std::vector<std::string> basis_types_events_cov_  = events_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_events_cov_ = events_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_events_cov_       = events_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_events_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_ = wrap_covariates_coefficients<_EVENT_>(events_cov_input[_coeff_basis_]);
    std::vector<double> lambda_events_cov_ = events_cov_input[_penalties_];
    auto coordinates_events_               = reader_data<_DATA_TYPE_,_NAN_REM_>(events_cov_input[_coords_]);     
    double kernel_bandwith_events_cov_     = events_cov_input[_bdw_ker_];
    //STATIONS COV  
    std::size_t q_S                                     = stations_cov_input[_q_];
    std::vector<std::size_t> number_basis_stations_cov_ = stations_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stations_cov_  = stations_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stations_cov_ = stations_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stations_cov_       = stations_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stations_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_ = wrap_covariates_coefficients<_STATION_>(stations_cov_input[_coeff_basis_]);
    std::vector<double> lambda_stations_cov_ = stations_cov_input[_penalties_];
    auto coordinates_stations_               = reader_data<_DATA_TYPE_,_NAN_REM_>(stations_cov_input[_coords_]);
    double kernel_bandwith_stations_cov_     = stations_cov_input[_bdw_ker_];    
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[_bc_];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[_coeff_basis_]);  //sono tutte Lc_jx1
        Bc.push_back(Bc_i);}
    //EVENTS BETAS  
    std::vector<std::size_t> number_basis_beta_events_cov_ = beta_events_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_events_cov_  = beta_events_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = beta_events_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_events_cov_ = beta_events_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_events_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size()); 
    //saving the betas basis expansion coefficients for events-dependent covariates
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix>> Be; //vettore esterno: per ogni covariata E. Interno: per ogni unità di training
    Be.reserve(q_E);
    Rcpp::List Be_list = model_fitted[_be_];
    for(std::size_t i = 0; i < q_E; ++i){
        Rcpp::List Be_i_list = Be_list[i];
        auto Be_i = wrap_covariates_coefficients<_EVENT_>(Be_i_list[_coeff_basis_]);
        Be.push_back(Be_i);}
    //STATIONS BETAS    
    std::vector<std::size_t> number_basis_beta_stations_cov_ = beta_stations_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stations_cov_  = beta_stations_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = beta_stations_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stations_cov_ = beta_stations_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stations_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());
    //PARTIAL RESIDUALS
    auto c_tilde_hat = reader_data<_DATA_TYPE_,_NAN_REM_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat}]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> A_S_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_A__}]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> B_S_for_K_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_B__for_K}]);


    ////////////////////////////////////////
    /////   TRAINING OBJECT CREATION   /////
    ////////////////////////////////////////
    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    //events (Theta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_E(knots_beta_events_cov_eigen_w_, 
                                                   degree_basis_beta_events_cov_, 
                                                   number_basis_beta_events_cov_, 
                                                   q_E);
    //stations (Psi)
    basis_systems< _DOMAIN_, bsplines_basis > bs_S(knots_beta_stations_cov_eigen_w_,  
                                                   degree_basis_beta_stations_cov_, 
                                                   number_basis_beta_stations_cov_, 
                                                   q_S);


    //PENALIZATION MATRICES                                               
    //events
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_E(std::move(bs_E),lambda_events_cov_);
    std::size_t Le = R_E.L();
    std::vector<std::size_t> Le_j = R_E.Lj();
    //stations
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_S(std::move(bs_S),lambda_stations_cov_);
    std::size_t Ls = R_S.L();
    std::vector<std::size_t> Ls_j = R_S.Lj();
    
    //additional info stationary
    std::size_t Lc = std::reduce(number_basis_beta_stationary_cov_.cbegin(),number_basis_beta_stationary_cov_.cend(),static_cast<std::size_t>(0));
    std::vector<std::size_t> Lc_j = number_basis_beta_stationary_cov_;


    //MODEL FITTED COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y_train_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_y_train_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_train_(std::move(coefficients_response_),std::move(basis_y_train_));
    //sttaionary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_train_(coefficients_stationary_cov_,
                                                               q_C,
                                                               basis_types_stationary_cov_,
                                                               degree_basis_stationary_cov_,
                                                               number_basis_stationary_cov_,
                                                               knots_stationary_cov_eigen_w_,
                                                               basis_fac);
    //events covariates
    functional_data_covariates<_DOMAIN_,_EVENT_> x_E_fd_train_(coefficients_events_cov_,
                                                               q_E,
                                                               basis_types_events_cov_,
                                                               degree_basis_events_cov_,
                                                               number_basis_events_cov_,
                                                               knots_events_cov_eigen_w_,
                                                               basis_fac);
    
    //stations covariates
    functional_data_covariates<_DOMAIN_,_STATION_> x_S_fd_train_(coefficients_stations_cov_,
                                                                 q_S,
                                                                 basis_types_stations_cov_,
                                                                 degree_basis_stations_cov_,
                                                                 number_basis_stations_cov_,
                                                                 knots_stations_cov_eigen_w_,
                                                                 basis_fac);


    //wrapping all the functional elements in a functional_matrix
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> theta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_E);
    //psi: a sparse functional matrix of dimension qsxLs
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> psi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_S);
    //phi: a sparse functional matrix n_trainx(n_train*Ly), where L is the number of basis for the response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >; 
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> phi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(*basis_response_,n_train,number_basis_response_);
    //y_train: a column vector of dimension n_trainx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_train_,number_threads);
    //Xc_train: a functional matrix of dimension n_trainxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_train_,number_threads);
    //Xe_train: a functional matrix of dimension n_trainxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xe_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_EVENT_>(x_E_fd_train_,number_threads);
    //Xs_train: a functional matrix of dimension n_trainxqs
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xs_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATION_>(x_S_fd_train_,number_threads);


    //////////////////////////////////////////////
    ///// WRAPPING COVARIATES TO BE PREDICTED ////
    //////////////////////////////////////////////
    // stationary
    //covariates names
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_to_be_pred_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov_to_pred); 
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(units_to_be_predicted,coefficients_stationary_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //events
    //covariates names
    std::vector<std::string> names_events_cov_ = wrap_covariates_names<_EVENT_>(coeff_events_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_to_be_pred_ = wrap_covariates_coefficients<_EVENT_>(coeff_events_cov_to_pred); 
    for(std::size_t i = 0; i < q_E; ++i){   
        check_dim_input<_EVENT_>(number_basis_events_cov_[i],coefficients_events_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_EVENT_>(units_to_be_predicted,coefficients_events_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //stations
    //covariates names
    std::vector<std::string> names_stations_cov_ = wrap_covariates_names<_STATION_>(coeff_stations_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_to_be_pred_ = wrap_covariates_coefficients<_STATION_>(coeff_stations_cov_to_pred);
    for(std::size_t i = 0; i < q_S; ++i){   
        check_dim_input<_STATION_>(number_basis_stations_cov_[i],coefficients_stations_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATION_>(units_to_be_predicted,coefficients_stations_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    
    //TO BE PREDICTED COVARIATES  
    //stationary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_to_be_pred_(coefficients_stationary_cov_to_be_pred_,
                                                                         q_C,
                                                                         basis_types_stationary_cov_,
                                                                         degree_basis_stationary_cov_,
                                                                         number_basis_stationary_cov_,
                                                                         knots_stationary_cov_eigen_w_,
                                                                         basis_fac);
    //events covariates
    functional_data_covariates<_DOMAIN_,_EVENT_>   x_E_fd_to_be_pred_(coefficients_events_cov_to_be_pred_,
                                                                      q_E,
                                                                      basis_types_events_cov_,
                                                                      degree_basis_events_cov_,
                                                                      number_basis_events_cov_,
                                                                      knots_events_cov_eigen_w_,
                                                                      basis_fac);
    //stations covariates
    functional_data_covariates<_DOMAIN_,_STATION_> x_S_fd_to_be_pred_(coefficients_stations_cov_to_be_pred_,
                                                                      q_S,
                                                                      basis_types_stations_cov_,
                                                                      degree_basis_stations_cov_,
                                                                      number_basis_stations_cov_,
                                                                      knots_stations_cov_eigen_w_,
                                                                      basis_fac);
    //Xc_new: a functional matrix of dimension n_newxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_to_be_pred_,number_threads);                                                               
    //Xe_new: a functional matrix of dimension n_newxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xe_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_EVENT_>(x_E_fd_to_be_pred_,number_threads);
    //Xs_new: a functional matrix of dimension n_newxqs
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xs_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATION_>(x_S_fd_to_be_pred_,number_threads);
    //map containing the X
    std::map<std::string,functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>> X_new = {
        {std::string{covariate_type<_STATIONARY_>()},Xc_new},
        {std::string{covariate_type<_EVENT_>()},Xe_new},
        {std::string{covariate_type<_STATION_>()},Xs_new}};

    ////////////////////////////////////////
    /////////        CONSTRUCTING W   //////
    ////////////////////////////////////////
    //distances
    auto coordinates_events_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_events_to_pred);
    check_dim_input<_EVENT_>(units_to_be_predicted,coordinates_events_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_EVENT_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_events_to_pred_.cols(),"coordinates matrix columns");
    auto coordinates_stations_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_stations_to_pred);
    check_dim_input<_STATION_>(units_to_be_predicted,coordinates_stations_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_STATION_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_stations_to_pred_.cols(),"coordinates matrix columns");
    distance_matrix_pred<_DISTANCE_> distances_events_to_pred_(std::move(coordinates_events_),std::move(coordinates_events_to_pred_));
    distance_matrix_pred<_DISTANCE_> distances_stations_to_pred_(std::move(coordinates_stations_),std::move(coordinates_stations_to_pred_));
    distances_events_to_pred_.compute_distances();
    distances_stations_to_pred_.compute_distances();
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    //functional weight matrix
    //events
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_,_KERNEL_,_DISTANCE_> W_E_pred(rec_weights_y_fd_,
                                                                                                                                                                            std::move(distances_events_to_pred_),
                                                                                                                                                                            kernel_bandwith_events_cov_,
                                                                                                                                                                            number_threads,
                                                                                                                                                                            true);
    W_E_pred.compute_weights_pred();                                                                         
    //stations
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_,_KERNEL_,_DISTANCE_> W_S_pred(rec_weights_y_fd_,
                                                                                                                                                                              std::move(distances_stations_to_pred_),
                                                                                                                                                                              kernel_bandwith_stations_cov_,
                                                                                                                                                                              number_threads,
                                                                                                                                                                              true);
    W_S_pred.compute_weights_pred();
    //We_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > We_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_>(W_E_pred,number_threads);
    //Ws_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Ws_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_>(W_S_pred,number_threads);
    //map containing the W
    std::map<std::string,std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>>> W_new = {
        {std::string{covariate_type<_EVENT_>()},We_pred},
        {std::string{covariate_type<_STATION_>()},Ws_pred}};


    //fwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(Bc),
                                                                                                 std::move(Be),
                                                                                                 std::move(omega),
                                                                                                 q_C,
                                                                                                 Lc,
                                                                                                 Lc_j,
                                                                                                 std::move(theta),
                                                                                                 q_E,
                                                                                                 Le,
                                                                                                 Le_j,
                                                                                                 std::move(psi),
                                                                                                 q_S,
                                                                                                 Ls,
                                                                                                 Ls_j,
                                                                                                 std::move(phi),
                                                                                                 number_basis_response_,
                                                                                                 std::move(c_tilde_hat),
                                                                                                 std::move(A_S_i),
                                                                                                 std::move(B_S_for_K_i),
                                                                                                 std::move(y_train),
                                                                                                 std::move(Xc_train),
                                                                                                 std::move(Xe_train),
                                                                                                 std::move(R_E.PenalizationMatrix()),
                                                                                                 std::move(Xs_train),
                                                                                                 std::move(R_S.PenalizationMatrix()),
                                                                                                 a,
                                                                                                 b,
                                                                                                 n_intervals,
                                                                                                 n_train,
                                                                                                 number_threads,
                                                                                                 in_cascade_estimation);

    Rcout << "Prediction" << std::endl;                                                                                             

    //retrieve partial residuals
    fwr_predictor->computePartialResiduals();
    //compute the new b for the non-stationary covariates
    fwr_predictor->computeBNew(W_new);
    //compute the beta for stationary covariates
    fwr_predictor->computeStationaryBetas();            
    //compute the beta for non-stationary covariates
    fwr_predictor->computeNonStationaryBetas();   
    //perform prediction
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_pred = fwr_predictor->predict(X_new);
    //evaluating the betas   
    fwr_predictor->evalBetas(abscissa_points_ev_);
    //evaluating the prediction
    std::vector< std::vector< _FD_OUTPUT_TYPE_>> y_pred_ev = fwr_predictor->evalPred(y_pred,abscissa_points_ev_);
    //smoothing of the prediction
    auto y_pred_smooth_coeff = fwr_predictor->smoothPred(y_pred,*basis_pred,knots_smoothing_pred);

    Rcout << "Prediction done" << std::endl;

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fwr_predictor->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 names_events_cov_,
                                                 basis_types_beta_events_cov_,
                                                 number_basis_beta_events_cov_,
                                                 knots_beta_events_cov_,
                                                 names_stations_cov_,
                                                 basis_types_beta_stations_cov_,
                                                 number_basis_beta_stations_cov_,
                                                 knots_beta_stations_cov_);
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fwr_predictor->betas(),
                                           abscissa_points_ev_,
                                           names_stationary_cov_,
                                           {},
                                           names_events_cov_,
                                           names_stations_cov_);
    //predictions evaluations
    Rcpp::List y_pred_ev_R = wrap_prediction_to_R_list<_FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_>(y_pred_ev,
                                                                                          abscissa_points_ev_,
                                                                                          y_pred_smooth_coeff,
                                                                                          basis_type_response_,
                                                                                          number_basis_response_,
                                                                                          degree_basis_response_,
                                                                                          knots_smoothing_pred);

    //returning element                                       
    Rcpp::List l;
    //regression model used and estimation technique
    l[_model_name_]      = std::string{algo_type<_FGWR_ALGO_>()};
    l[_estimation_iter_] = estimation_iter(in_cascade_estimation);
    //predictions
    l[std::string{FDAGWR_HELPERS_for_PRED_NAMES::pred}] = y_pred_ev_R;
    //stationary covariate basis expansion coefficients for beta_c
    l[_bc_ + "_pred"]  = b_coefficients[_bc_];
    //beta_c
    l[_beta_c_ + "_pred"] = betas[_beta_c_];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[_be_ + "_pred"]  = b_coefficients[_be_];
    //beta_e
    l[_beta_e_ + "_pred"] = betas[_beta_e_];
    //station-dependent covariate basis expansion coefficients for beta_s
    l[_bs_ + "_pred"]  = b_coefficients[_bs_];
    //beta_s
    l[_beta_s_ + "_pred"] = betas[_beta_s_];

    return l;
}





/*!
* @brief Fitting a Functional Mixed Geographically Weighted Regression model. The covariates are functional objects, divided into
*        two categories: stationary covariates (C), constant over geographical space, and non-stationary covariates (NC), that vary depending on spatial coordinates. Regression coefficients are estimated 
*        in the following order: C, NC. The functional response is already reconstructed according to the method proposed by Bortolotti et Al. (2024) (link below)
* @param y_points matrix of double containing the raw response: each row represents a specific abscissa for which the response evaluation is available, each column a statistical unit. Response is a already reconstructed.
* @param t_points vector of double with the abscissa points with respect of the raw evaluations of y_points are available (length of t_points is equal to the number of rows of y_points).
* @param left_extreme_domain double indicating the left extreme of the functional data domain (not necessarily the smaller element in t_points).
* @param right_extreme_domain double indicating the right extreme of the functional data domain (not necessarily the biggest element in t_points).
* @param coeff_y_points matrix of double containing the coefficient of response's basis expansion: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param knots_y_points vector of double with the abscissa points with respect which the basis expansions of the response and response reconstruction weights are performed (all elements contained in [a,b]). 
* @param degree_basis_y_points non-negative integer: the degree of the basis used for the basis expansion of the (functional) response. Default explained below (can be NULL).
* @param n_basis_y_points positive integer: number of basis for the basis expansion of the (functional) response. It must match number of rows of coeff_y_points. Default explained below (can be NULL).
* @param coeff_rec_weights_y_points matrix of double containing the coefficients of the basis expansion of the weights to reconstruct the (functional) response: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param degree_basis_rec_weights_y_points non-negative integer: the degree of the basis used for response reconstruction weights. Default explained below (can be NULL).
* @param n_basis_rec_weights_y_points positive integer: number of basis for the basis expansion of response reconstruction weights. It must match number of rows of coeff_rec_weights_y_points. Default explained below (can be NULL).
* @param coeff_stationary_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                             The name of the i-th element is the name of the i-th stationary covariate (default: "reg.Ci" if no name present).
* @param basis_types_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariate basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate. Default explained below (can be NULL).
* @param n_basis_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate. It must match number of rows of the i-th element of coeff_stationary_cov. Default explained below (can be NULL).
* @param penalization_stationary_cov vector of non-negative double: element i-th is the penalization used for the i-th stationary covariate.
* @param knots_beta_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates functional regression coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate functional regression coefficients. Default explained below (can be NULL).
* @param n_basis_beta_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate functional regression coefficients. Default explained below (can be NULL).
* @param coeff_non_stationary_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th non-stationary covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                                 The name of the i-th element is the name of the i-th non-stationary covariate (default: "reg.NCi" if no name present).
* @param basis_types_non_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th non-stationary covariate basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_non_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the non-stationary covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_non_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th non-stationary covariate. Default explained below (can be NULL).
* @param n_basis_non_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th non-stationary covariate. It must match number of rows of the i-th element of coeff_non_stationary_cov. Default explained below (can be NULL).
* @param penalization_non_stationary_cov vector of non-negative double: element i-th is the penalization used for the i-th non-stationary covariate.
* @param coordinates_non_stationary matrix of double containing the UTM coordinates of the non-stationary site of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
* @param kernel_bandwith_non_stationary positive double indicating the bandwith of the gaussian kernel used to smooth the distances within non-stationary sites.
* @param knots_beta_non_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the non-stationary covariates functional regression coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_non_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th non-stationary covariate functional regression coefficient. Default explained below (can be NULL).
* @param n_basis_beta_non_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th non-stationary covariate functional regression coefficient. Default explained below (can be NULL).
* @param in_cascade_estimation bool: if false, an exact algorithm taking account for the interaction within non-stationary covariates is used to fit the model. Otherwise, the model is fitted in cascade. The first option is more precise, but way more computationally intensive.
* @param n_knots_smoothing number of knots used to perform the smoothing on the response obtained leaving out all the non-stationary components (default: 100).
* @param n_intervals_quadrature number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: 100).
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine.
* @param basis_type_y_points string containing the type of basis used for the functional response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_type_rec_weights_y_points string containing the type of basis used for the weights to reconstruct the functional response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariate functional regression coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_non_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariate functional regression coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @return an R list containing:
*         - "FGWR": string containing the type of fwr used ("FMGWR")
*         - "EstimationTechnique": "Exact" if in_cascade_estimation false, "Cascade" if in_cascade_estimation true 
*         - "Bc": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "basis_coeff": a Lc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective element of basis_types_beta_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stationary_cov)
*         - "Beta_c": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Beta_eval": a vector of double containing the discrete evaluations of the stationary beta
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "Bnc": a list containing, for each non-stationary covariate regression coefficent (each element is named with the element names in the list coeff_non_stationary_cov (default, if not given: "CovNC*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Lnc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_non_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_non_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_non_stationary_cov)
*         - "Beta_nc": a list containing, for each non-stationary covariate regression coefficent (each element is named with the element names in the list coeff_non_stationary_cov (default, if not given: "CovNC*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)  
*         - "predictor_info": a list containing partial residuals and information of the fitted model to perform predictions for new statistical units:
*                             - "partial_res": a list containing information to compute the partial residuals:
*                                              - "c_tilde_hat": vector of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon (if in_cascade_estimation is true, contains only 0s).
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response (element n_basis_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response. Possible values: "bsplines", "constant". (element basis_type_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response (element degree_basis_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response (element coeff_y_points).
*                                              - "ResponseReconstructionWeights": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response reconstruction weights (element n_basis_rec_weights_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response reconstruction weights. Possible values: "bsplines", "constant". (element basis_type_rec_weights_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response reconstruction weights (element degree_basis_rec_weights_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response reconstruction weights (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response reconstruction weights (element coeff_rec_weights_y_points).
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates": number of stationary covariates (length of coeff_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional stationary covariates (respective elements of n_basis_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional stationary covariates (respective elements of degrees_basis_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional stationary covariates (element knots_stationary_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional stationary covariates (respective elements of coeff_stationary_cov).
*                                              - "beta_Stationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element n_basis_beta_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element degrees_basis_beta_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the stationary covariates (element knots_beta_stationary_cov).                                                            
*                                              - "cov_NonStationary": list:
*                                                            - "number_covariates": number of non-stationary covariates (length of coeff_non_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional non-stationary covariates (respective elements of n_basis_non_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional non-stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_non_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional non-stationary covariates (respective elements of degrees_basis_non_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional non-stationary covariates (element knots_non_stationary_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional non-stationary covariates (respective elements of coeff_non_stationary_cov).
*                                                            - "penalizations": vector containing the penalizations of the non-stationary covariates (respective elements of penalization_non_stationary_cov)
*                                                            - "coordinates": UTM coordinates of the non-stationary sites of the training data (element coordinates_non_stationary).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the non-stationary sites (element kernel_bandwith_non_stationary).
*                                              - "beta_NonStationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element n_basis_beta_non_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_non_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element degrees_basis_beta_non_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element knots_beta_non_stationary_cov).
*                                              - "a": domain left extreme  (element left_extreme_domain).
*                                              - "b": domain right extreme (element right_extreme_domain).
*                                              - "abscissa": abscissa for which the evaluations of the functional data are available (element t_points).
*                                              - "InCascadeEstimation": element in_cascade_estimation.
* @details constant basis are used, for a covariate, if it resembles a scalar shape. It consists of a straight line with y-value equal to 1 all over the data domain.
*          Can be seen as a B-spline basis with degree 0, number of basis 1, using one knot, consequently having only one coefficient for the only basis for each statistical unit.
*          fdagwr sets all the feats accordingly if reads constant basis.
*          However, recall that the response is a functional datum, as the regressors coefficients. Since the package's basis variety could be hopefully enlarged in the future 
*          (for example, introducing Fourier basis for handling data that present periodical behaviors), the input parameters regarding basis types for response, response reconstruction
*          weights and regressors coefficients are left at the end of the input list, and defaulted as NULL. Consequently they will use a B-spline basis system, and should NOT use a constant basis,
*          Recall to perform externally the basis expansion before using the package, and afterwards passing basis types, degree and number and basis expansion coefficients and knots coherently
* @note a little excursion about degree and number of basis passed as input. For each specific covariate, or the response, if using B-spline basis, remember that number of knots = number of basis - degree + 1. 
*       By default, if passing NULL, fdagwr uses a cubic B-spline system of basis, the number of basis is computed coherently from the number of knots (that is the only mandatory input parameter).
*       Passing only the degree of the bsplines, the number of basis used will be set accordingly, and viceversa if passing only the number of basis. 
*       But, take care that the number of basis used has to match the number of rows of coefficients matrix (for EACH type of basis). If not, an exception is thrown. No problems arise if letting fdagwr defaulting the number of basis.
*       For response and response reconstruction weights, degree and number of basis consist of integer, and can be NULL. For all the regressors, and their coefficients, the inputs consist of vector of integers: 
*       if willing to pass a default parameter, all the vector has to be defaulted (if passing NULL, a vector with all 3 for the degrees is passed, for example)
* @link https://www.researchgate.net/publication/377251714_Weighted_Functional_Data_Analysis_for_the_Calibration_of_a_Ground_Motion_Model_in_Italy @endlink
*/
//
// [[Rcpp::export]]
Rcpp::List FMGWR(Rcpp::NumericMatrix y_points,
                 Rcpp::NumericVector t_points,
                 double left_extreme_domain,
                 double right_extreme_domain,
                 Rcpp::NumericMatrix coeff_y_points,
                 Rcpp::NumericVector knots_y_points,
                 Rcpp::Nullable<int> degree_basis_y_points,
                 Rcpp::Nullable<int> n_basis_y_points,
                 Rcpp::NumericMatrix coeff_rec_weights_y_points,
                 Rcpp::Nullable<int> degree_basis_rec_weights_y_points,
                 Rcpp::Nullable<int> n_basis_rec_weights_y_points,
                 Rcpp::List coeff_stationary_cov,
                 Rcpp::Nullable<Rcpp::CharacterVector> basis_types_stationary_cov,
                 Rcpp::NumericVector knots_stationary_cov,
                 Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_stationary_cov,
                 Rcpp::Nullable<Rcpp::IntegerVector> n_basis_stationary_cov,
                 Rcpp::NumericVector penalization_stationary_cov,
                 Rcpp::NumericVector knots_beta_stationary_cov,
                 Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_beta_stationary_cov,
                 Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_stationary_cov,
                 Rcpp::List coeff_non_stationary_cov,
                 Rcpp::Nullable<Rcpp::CharacterVector> basis_types_non_stationary_cov,
                 Rcpp::NumericVector knots_non_stationary_cov,
                 Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_non_stationary_cov,
                 Rcpp::Nullable<Rcpp::IntegerVector> n_basis_non_stationary_cov,
                 Rcpp::NumericVector penalization_non_stationary_cov,
                 Rcpp::NumericMatrix coordinates_non_stationary,
                 double kernel_bandwith_non_stationary,
                 Rcpp::NumericVector knots_beta_non_stationary_cov,
                 Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_beta_non_stationary_cov,
                 Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_non_stationary_cov,
                 bool in_cascade_estimation = false,                 
                 int n_knots_smoothing = 100,
                 int n_intervals_quadrature = 100,
                 Rcpp::Nullable<int> num_threads = R_NilValue,
                 std::string basis_type_y_points = "bsplines",
                 std::string basis_type_rec_weights_y_points = "bsplines",
                 Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue,
                 Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_non_stationary_cov = R_NilValue)
{
    Rcout << "Functional Mixed Geographically Weighted Regression" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT

    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FMGWR_;                              //fgwr type (estimating stationary -> event-dependent -> station-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _NON_STATIONARY_ = FDAGWR_COVARIATES_TYPES::NON_STATIONARY;      //enum for non stationary covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF KNOTS TO PERFORM SMOOTHING ON THE RESPONSE WITHOUT THE NON-STATIONARY COMPONENTS
    int n_knots_smoothing_y_new = wrap_and_check_n_knots_smoothing(n_knots_smoothing);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);


    //  RESPONSE
    //raw data
    auto response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(y_points);       //Eigen dense matrix type (auto is necessary )
    //number of statistical units
    std::size_t number_of_statistical_units_ = response_.cols();
    //coefficients matrix
    auto coefficients_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_y_points);
    auto coefficients_response_out_ = coefficients_response_;
    //c: a dense matrix of double (n*Ly) x 1 containing, one column below the other, the y basis expansion coefficients
    auto c = columnize_coeff_resp(coefficients_response_);
    //reconstruction weights coefficients matrix
    auto coefficients_rec_weights_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_rec_weights_y_points);
    auto coefficients_rec_weights_response_out_ = coefficients_rec_weights_response_;

    //  ABSCISSA POINTS of response
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = wrap_abscissas(t_points,left_extreme_domain,right_extreme_domain);
    // wrapper into eigen
    check_dim_input<_RESPONSE_>(response_.rows(), abscissa_points_.size(), "points for evaluation of raw data vector");   //check that size of abscissa points and number of evaluations of fd raw data coincide
    FDAGWR_TRAITS::Dense_Matrix abscissa_points_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(abscissa_points_.data(),abscissa_points_.size(),1);
    _FD_INPUT_TYPE_ a = left_extreme_domain;
    _FD_INPUT_TYPE_ b = right_extreme_domain;


    //  KNOTS (for basis expansion and for smoothing)
    //response
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = wrap_abscissas(knots_y_points,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    //stationary cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_ = wrap_abscissas(knots_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    //beta stationary cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = wrap_abscissas(knots_beta_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //non stationary cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_non_stationary_cov_ = wrap_abscissas(knots_non_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_non_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_non_stationary_cov_.data(),knots_non_stationary_cov_.size());
    //beta events cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_non_stationary_cov_ = wrap_abscissas(knots_beta_non_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_non_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_non_stationary_cov_.data(),knots_beta_non_stationary_cov_.size());
    //knots for performing smoothing (n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);


    //  COVARIATES names, coefficients and how many (q_), for every type
    //stationary 
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov);
    std::size_t q_C = names_stationary_cov_.size();    //number of stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov);    
    //non stationary
    std::vector<std::string> names_non_stationary_cov_ = wrap_covariates_names<_NON_STATIONARY_>(coeff_non_stationary_cov);
    std::size_t q_NC = names_non_stationary_cov_.size();        //number of events related covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_non_stationary_cov_ = wrap_covariates_coefficients<_NON_STATIONARY_>(coeff_non_stationary_cov);


    //  BASIS TYPES
    //response
    std::string basis_type_response_ = wrap_and_check_basis_type<_RESPONSE_>(basis_type_y_points);
    //response reconstruction weights
    std::string basis_type_rec_weights_response_ = wrap_and_check_basis_type<_REC_WEIGHTS_>(basis_type_rec_weights_y_points);
    //stationary
    std::vector<std::string> basis_types_stationary_cov_ = wrap_and_check_basis_type<_STATIONARY_>(basis_types_stationary_cov,q_C);
    //beta stationary cov 
    std::vector<std::string> basis_types_beta_stationary_cov_ = wrap_and_check_basis_type<_STATIONARY_>(basis_types_beta_stationary_cov,q_C);
    //events
    std::vector<std::string> basis_types_non_stationary_cov_ = wrap_and_check_basis_type<_NON_STATIONARY_>(basis_types_non_stationary_cov,q_NC);
    //beta events cov 
    std::vector<std::string> basis_types_beta_non_stationary_cov_ = wrap_and_check_basis_type<_NON_STATIONARY_>(basis_types_beta_non_stationary_cov,q_NC);


    //  BASIS NUMBER AND DEGREE: checking matrix coefficients dimensions: rows: number of basis; cols: number of statistical units
    //response
    auto number_and_degree_basis_response_ = wrap_and_check_basis_number_and_degree<_RESPONSE_>(n_basis_y_points,degree_basis_y_points,knots_response_.size(),basis_type_response_);
    std::size_t number_basis_response_ = number_and_degree_basis_response_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::size_t degree_basis_response_ = number_and_degree_basis_response_[std::string{FDAGWR_FEATS::degree_basis_string}];
    check_dim_input<_RESPONSE_>(number_basis_response_,coefficients_response_.rows(),"response coefficients matrix rows");
    check_dim_input<_RESPONSE_>(number_of_statistical_units_,coefficients_response_.cols(),"response coefficients matrix columns");     
    //response reconstruction weights
    auto number_and_degree_basis_rec_weights_response_ = wrap_and_check_basis_number_and_degree<_REC_WEIGHTS_>(n_basis_rec_weights_y_points,degree_basis_rec_weights_y_points,knots_response_.size(),basis_type_rec_weights_response_);
    std::size_t number_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::size_t degree_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[std::string{FDAGWR_FEATS::degree_basis_string}];
    check_dim_input<_REC_WEIGHTS_>(number_basis_rec_weights_response_,coefficients_rec_weights_response_.rows(),"response reconstruction weights coefficients matrix rows");
    check_dim_input<_REC_WEIGHTS_>(number_of_statistical_units_,coefficients_rec_weights_response_.cols(),"response reconstruction weights coefficients matrix columns");     
    //stationary cov
    auto number_and_degree_basis_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_stationary_cov,degrees_basis_stationary_cov,knots_stationary_cov_.size(),q_C,basis_types_stationary_cov_);
    std::vector<std::size_t> number_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stationary cov
    auto number_and_degree_basis_beta_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_beta_stationary_cov,degrees_basis_beta_stationary_cov,knots_beta_stationary_cov_.size(),q_C,basis_types_beta_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    //non stationary cov    
    auto number_and_degree_basis_non_stationary_cov_ = wrap_and_check_basis_number_and_degree<_NON_STATIONARY_>(n_basis_non_stationary_cov,degrees_basis_non_stationary_cov,knots_non_stationary_cov_.size(),q_NC,basis_types_non_stationary_cov_);
    std::vector<std::size_t> number_basis_non_stationary_cov_ = number_and_degree_basis_non_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_non_stationary_cov_ = number_and_degree_basis_non_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    for(std::size_t i = 0; i < q_NC; ++i){   
        check_dim_input<_NON_STATIONARY_>(number_basis_non_stationary_cov_[i],coefficients_non_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_NON_STATIONARY_>(number_of_statistical_units_,coefficients_non_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta non stationary cov
    auto number_and_degree_basis_beta_non_stationary_cov_ = wrap_and_check_basis_number_and_degree<_NON_STATIONARY_>(n_basis_beta_non_stationary_cov,degrees_basis_beta_non_stationary_cov,knots_beta_non_stationary_cov_.size(),q_NC,basis_types_beta_non_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_non_stationary_cov_ = number_and_degree_basis_beta_non_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_beta_non_stationary_cov_ = number_and_degree_basis_beta_non_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];


    //  DISTANCES
    //non stationary    DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_non_stationary_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_non_stationary);
    auto coordinates_non_stationary_out_ = coordinates_non_stationary_;
    check_dim_input<_NON_STATIONARY_>(number_of_statistical_units_,coordinates_non_stationary_.rows(),"coordinates matrix rows");
    check_dim_input<_NON_STATIONARY_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_non_stationary_.cols(),"coordinates matrix columns");
    distance_matrix<_DISTANCE_> distances_non_stationary_cov_(std::move(coordinates_non_stationary_),number_threads);


    //  PENALIZATION TERMS: checking their consistency
    //stationary
    std::vector<double> lambda_stationary_cov_ = wrap_and_check_penalizations<_STATIONARY_>(penalization_stationary_cov,q_C);
    //non stationary
    std::vector<double> lambda_non_stationary_cov_ = wrap_and_check_penalizations<_NON_STATIONARY_>(penalization_non_stationary_cov,q_NC);


    //  KERNEL BANDWITH
    //non stationary
    double kernel_bandwith_non_stationary_cov_ = wrap_and_check_kernel_bandwith<_NON_STATIONARY_>(kernel_bandwith_non_stationary);


    ////////////////////////////////////////
    /////    END PARAMETERS WRAPPING   /////
    ////////////////////////////////////////



    ////////////////////////////////
    /////    OBJECT CREATION   /////
    ////////////////////////////////


    //DISTANCES
    //non stationary
    distances_non_stationary_cov_.compute_distances();



    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    //non stationary (Eta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_NC(knots_beta_non_stationary_cov_eigen_w_, 
                                                    degree_basis_beta_non_stationary_cov_, 
                                                    number_basis_beta_non_stationary_cov_, 
                                                    q_NC);

    
    
    //PENALIZATION MATRICES
    //stationary
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_C(std::move(bs_C),lambda_stationary_cov_);
    std::size_t Lc = R_C.L();
    std::vector<std::size_t> Lc_j = R_C.Lj();
    //non stationary
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_NC(std::move(bs_NC),lambda_non_stationary_cov_);
    std::size_t Lnc = R_NC.L();
    std::vector<std::size_t> Lnc_j = R_NC.Lj();



    //FD OBJECTS: RESPONSE and COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_(std::move(coefficients_response_),std::move(basis_response_));
    
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    
    //stationary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_(coefficients_stationary_cov_,
                                                              q_C,
                                                              basis_types_stationary_cov_,
                                                              degree_basis_stationary_cov_,
                                                              number_basis_stationary_cov_,
                                                              knots_stationary_cov_eigen_w_,
                                                              basis_fac);
    
    //non stationary covariates
    functional_data_covariates<_DOMAIN_,_NON_STATIONARY_> x_NC_fd_(coefficients_non_stationary_cov_,
                                                                   q_NC,
                                                                   basis_types_non_stationary_cov_,
                                                                   degree_basis_non_stationary_cov_,
                                                                   number_basis_non_stationary_cov_,
                                                                   knots_non_stationary_cov_eigen_w_,
                                                                   basis_fac);
    

    //FUNCTIONAL WEIGHT MATRIX
    //stationary
    functional_weight_matrix_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATIONARY_> W_C(rec_weights_y_fd_,
                                                                                                                                                    number_threads);
    W_C.compute_weights();                                                      
    //non stationary
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_,_KERNEL_,_DISTANCE_> W_NC(rec_weights_y_fd_,
                                                                                                                                                                                 std::move(distances_non_stationary_cov_),
                                                                                                                                                                                 kernel_bandwith_non_stationary_cov_,
                                                                                                                                                                                 number_threads);
    W_NC.compute_weights();                                                                         



    ///////////////////////////////
    /////    FGWR ALGORITHM   /////
    ///////////////////////////////
    //wrapping all the functional elements in a functional_matrix

    //y: a column vector of dimension nx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_,number_threads);
    //phi: a sparse functional matrix nx(n*L), where L is the number of basis for the response
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> phi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_.fdata_basis(),number_of_statistical_units_,number_basis_response_);
    //c: a dense matrix of double (n*Ly) x 1 containing, one column below the other, the y basis expansion coefficients
    //already done at the beginning
    //basis used for doing response basis expansion
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //Xc: a functional matrix of dimension nxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_,number_threads);
    //Wc: a diagonal functional matrix of dimension nxn
    functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Wc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATIONARY_>(W_C,number_threads);
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);
    //Xnc: a functional matrix of dimension nxqnc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xnc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_NON_STATIONARY_>(x_NC_fd_,number_threads);
    //Wnc: n diagonal functional matrices of dimension nxn
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Wnc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_>(W_NC,number_threads);
    //eta: a sparse functional matrix of dimension qncxLnc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> eta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_NC);


    //fwr algorithm
    auto fgwr_algo = fwr_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(y),
                                                                                   std::move(phi),
                                                                                   std::move(c),
                                                                                   number_basis_response_,
                                                                                   std::move(basis_y),
                                                                                   std::move(knots_smoothing),
                                                                                   std::move(Xc),
                                                                                   std::move(Wc),
                                                                                   std::move(R_C.PenalizationMatrix()),
                                                                                   std::move(omega),
                                                                                   q_C,
                                                                                   Lc,
                                                                                   Lc_j,
                                                                                   std::move(Xnc),
                                                                                   std::move(Wnc),
                                                                                   std::move(R_NC.PenalizationMatrix()),
                                                                                   std::move(eta),
                                                                                   q_NC,
                                                                                   Lnc,
                                                                                   Lnc_j,
                                                                                   a,
                                                                                   b,
                                                                                   n_intervals,
                                                                                   abscissa_points_,
                                                                                   number_of_statistical_units_,
                                                                                   number_threads,
                                                                                   in_cascade_estimation);

    Rcout << "Model fitting" << std::endl;                                                                                    
                                                                                   
    //computing the b
    fgwr_algo->compute();
    //evaluating the betas   
    fgwr_algo->evalBetas();

    Rcout << "Model fitted" << std::endl; 

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fgwr_algo->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 names_non_stationary_cov_,
                                                 basis_types_beta_non_stationary_cov_,
                                                 number_basis_beta_non_stationary_cov_,
                                                 knots_beta_non_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {});
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fgwr_algo->betas(),
                                           abscissa_points_,
                                           names_stationary_cov_,
                                           names_non_stationary_cov_,
                                           {},
                                           {});
    //elements for partial residuals
    Rcpp::List p_res = wrap_PRes_to_R_list(fgwr_algo->PRes());

    //returning element
    Rcpp::List l;
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _estimation_iter_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter};     //Exact or Cascade estimation
    std::string _bc_                = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_            = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _bnc_               = std::string{FDAGWR_B_NAMES::bnc};                                //bnc
    std::string _beta_nc_           = std::string{FDAGWR_BETAS_NAMES::beta_nc};                        //beta_nc
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _partial_residuals_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res};               //partial residuals 
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                          //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                       //response reconstruction weights
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};     //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};     //stationary betas
    std::string _cov_no_stat_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_NON_STATIONARY_>()}; //event-dependent training covariates
    std::string _beta_no_stat_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_NON_STATIONARY_>()}; //event-dependent betas
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations
    std::string _cascade_estimate_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cascade_estimate};    //if using in cascade-estimation

    
    //regression model used
    l[_model_name_] = std::string{algo_type<_FGWR_ALGO_>()};
    //estimation technique
    l[_estimation_iter_] = estimation_iter(in_cascade_estimation);
    //stationary covariate basis expansion coefficients for beta_c
    l[_bc_] = b_coefficients[_bc_];
    //beta_c
    l[_beta_c_] = betas[_beta_c_];
    //non stationary covariate basis expansion coefficients for beta_nc
    l[_bnc_] = b_coefficients[_bnc_];
    //beta_nc
    l[_beta_nc_] = betas[_beta_nc_];
    //elements needed to perform prediction
    Rcpp::List elem_for_pred;
    elem_for_pred[_partial_residuals_] = p_res;    //partial residuals, elements to reconstruct them
    Rcpp::List inputs_info;                        //containing training data information needed to prediction purposes
    
    //adding all the elements of the training set
    //input of y
    Rcpp::List response_input;
    response_input[_n_basis_]     = number_basis_response_;
    response_input[_t_basis_]     = basis_type_response_;
    response_input[_deg_basis_]   = degree_basis_response_;
    response_input[_knots_basis_] = knots_response_;
    response_input[_coeff_basis_] = Rcpp::wrap(coefficients_response_out_);
    inputs_info[_response_]       = response_input;
    //input of w for y  
    Rcpp::List response_rec_w_input;
    response_rec_w_input[_n_basis_]     = number_basis_rec_weights_response_;
    response_rec_w_input[_t_basis_]     = basis_type_rec_weights_response_;
    response_rec_w_input[_deg_basis_]   = degree_basis_rec_weights_response_;
    response_rec_w_input[_knots_basis_] = knots_response_;
    response_rec_w_input[_coeff_basis_] = Rcpp::wrap(coefficients_rec_weights_response_out_);
    inputs_info[_response_rec_w_]       = response_rec_w_input;
    //input of C
    Rcpp::List C_input;
    C_input[_q_]            = q_C;
    C_input[_n_basis_]      = number_basis_stationary_cov_;
    C_input[_t_basis_]      = basis_types_stationary_cov_;
    C_input[_deg_basis_]    = degree_basis_stationary_cov_;
    C_input[_knots_basis_]  = knots_stationary_cov_;
    C_input[_coeff_basis_]  = toRList(coefficients_stationary_cov_,false);
    inputs_info[_cov_stat_] = C_input;
    //input of Beta C   
    Rcpp::List beta_C_input;
    beta_C_input[_n_basis_]     = number_basis_beta_stationary_cov_;
    beta_C_input[_t_basis_]     = basis_types_beta_stationary_cov_;
    beta_C_input[_deg_basis_]   = degree_basis_beta_stationary_cov_;
    beta_C_input[_knots_basis_] = knots_beta_stationary_cov_;
    inputs_info[_beta_stat_]    = beta_C_input;
    //input of NC
    Rcpp::List NC_input;
    NC_input[_q_]              = q_NC;
    NC_input[_n_basis_]        = number_basis_non_stationary_cov_;
    NC_input[_t_basis_]        = basis_types_non_stationary_cov_;
    NC_input[_deg_basis_]      = degree_basis_non_stationary_cov_;
    NC_input[_knots_basis_]    = knots_non_stationary_cov_;
    NC_input[_coeff_basis_]    = toRList(coefficients_non_stationary_cov_,false);
    NC_input[_penalties_]      = lambda_non_stationary_cov_;
    NC_input[_coords_]         = Rcpp::wrap(coordinates_non_stationary_out_);
    NC_input[_bdw_ker_]        = kernel_bandwith_non_stationary;
    inputs_info[_cov_no_stat_] = NC_input;
    //input of Beta NC   
    Rcpp::List beta_NC_input;
    beta_NC_input[_n_basis_]     = number_basis_beta_non_stationary_cov_;
    beta_NC_input[_t_basis_]     = basis_types_beta_non_stationary_cov_;
    beta_NC_input[_deg_basis_]   = degree_basis_beta_non_stationary_cov_;
    beta_NC_input[_knots_basis_] = knots_beta_non_stationary_cov_;
    inputs_info[_beta_no_stat_]  = beta_NC_input;
    //domain
    inputs_info[_n_] = number_of_statistical_units_;
    inputs_info[_a_] = a;
    inputs_info[_b_] = b;
    inputs_info[_abscissa_]         = abscissa_points_;
    inputs_info[_cascade_estimate_] = in_cascade_estimation;
    //adding all the elements to perform prediction
    elem_for_pred[_input_info_] = inputs_info;
    l[_elem_for_pred_]          = elem_for_pred;

    return l;
}


/*!
* @brief Function to perform predictions on new statistical units using a fitted Functional Mixed Geographically Weighted Regression model. Non-stationary betas have to be recomputed in the new locations.
* @param coeff_stationary_cov_to_pred list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit to be predicted.
* @param coeff_non_stationary_cov_to_pred list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th non-stationary covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit to be predicted.
* @param coordinates_non_stationary_to_pred matrix of double containing the UTM coordinates of the non-stationary site of new statistical units: each row represents a statistical unit to be predicted, each column a coordinate (2 columns).
* @param units_to_be_predicted number of units to be predicted
* @param abscissa_ev abscissa for which then evaluating the predicted reponse and betas, stationary and non-stationary, which have to be recomputed
* @param model_fitted: output of FMGWR: an R list containing:
*         - "FGWR": string containing the type of fwr used ("FMGWR")
*         - "EstimationTechnique": "Exact" if in_cascade_estimation false, "Cascade" if in_cascade_estimation true 
*         - "Bc": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "basis_coeff": a Lc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective element of basis_types_beta_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stationary_cov)
*         - "Beta_c": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Beta_eval": a vector of double containing the discrete evaluations of the stationary beta
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "Bnc": a list containing, for each non-stationary covariate regression coefficent (each element is named with the element names in the list coeff_non_stationary_cov (default, if not given: "CovNC*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Lnc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_non_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_non_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_non_stationary_cov)
*         - "Beta_nc": a list containing, for each non-stationary covariate regression coefficent (each element is named with the element names in the list coeff_non_stationary_cov (default, if not given: "CovNC*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)  
*         - "predictor_info": a list containing partial residuals and information of the fitted model to perform predictions for new statistical units:
*                             - "partial_res": a list containing information to compute the partial residuals:
*                                              - "c_tilde_hat": vector of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon (if in_cascade_estimation is true, contains only 0s).
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response (element n_basis_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response. Possible values: "bsplines", "constant". (element basis_type_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response (element degree_basis_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response (element coeff_y_points).
*                                              - "ResponseReconstructionWeights": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response reconstruction weights (element n_basis_rec_weights_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response reconstruction weights. Possible values: "bsplines", "constant". (element basis_type_rec_weights_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response reconstruction weights (element degree_basis_rec_weights_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response reconstruction weights (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response reconstruction weights (element coeff_rec_weights_y_points).
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates": number of stationary covariates (length of coeff_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional stationary covariates (respective elements of n_basis_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional stationary covariates (respective elements of degrees_basis_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional stationary covariates (element knots_stationary_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional stationary covariates (respective elements of coeff_stationary_cov).
*                                              - "beta_Stationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element n_basis_beta_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element degrees_basis_beta_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the stationary covariates (element knots_beta_stationary_cov).                                                            
*                                              - "cov_NonStationary": list:
*                                                            - "number_covariates": number of non-stationary covariates (length of coeff_non_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional non-stationary covariates (respective elements of n_basis_non_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional non-stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_non_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional non-stationary covariates (respective elements of degrees_basis_non_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional non-stationary covariates (element knots_non_stationary_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional non-stationary covariates (respective elements of coeff_non_stationary_cov).
*                                                            - "penalizations": vector containing the penalizations of the non-stationary covariates (respective elements of penalization_non_stationary_cov)
*                                                            - "coordinates": UTM coordinates of the non-stationary sites of the training data (element coordinates_non_stationary).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the non-stationary sites (element kernel_bandwith_non_stationary).
*                                              - "beta_NonStationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element n_basis_beta_non_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_non_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element degrees_basis_beta_non_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element knots_beta_non_stationary_cov).
*                                              - "a": domain left extreme  (element left_extreme_domain).
*                                              - "b": domain right extreme (element right_extreme_domain).
*                                              - "abscissa": abscissa for which the evaluations of the functional data are available (element t_points).
*                                              - "InCascadeEstimation": element in_cascade_estimation.
* @param n_knots_smoothing_pred number of knots used to smooth predicted response and non-stationary, obtaining basis expansion coefficients with respect to the training basis (default: 100).
* @param n_intervals_quadrature number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: 100).
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine.
* @return an R list containing:
*         - "FGWR_predictor": string containing the model used to predict ("predictor_FMGWR")
*         - "EstimationTechnique": "Exact" if in_cascade_estimation in the fitted model false, "Cascade" if in_cascade_estimation in the fitted model true 
*         - "prediction": list containing:
*                         - "evaluation": list containing the evaluation of the prediction:
*                                          - "prediction_ev": list containing, for each unit to be predicted, the raw evaluations of the predicted response.
*                                          - "abscissa_ev": the abscissa points for which the prediction evaluation is available (element abscissa_ev).
*                         - "fd": list containing the prediction functional description:
*                                          - "prediction_basis_coeff": matrix containing the prediction basis expansion coefficients (each row a basis, each column a new statistical unit)
*                                          - "prediction_basis_type": basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_basis_num": number of basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_basis_deg": degree of basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_knots": knots used for the predicted response smoothing (n_knots_smoothing_pred equally spaced knots in the functional datum domain)
*         - "Bc_pred": list containing, for each stationary covariate:
*                      - "basis_coeff": coefficients of the basis expansion of the beta (from model_fitted).
*                      - "basis_num": number of basis used for the beta basis epxnasion (from model_fitted).
*                      - "basis_type": type of basis used for the beta basis expansion (from model_fitted).
*                      - "knots": knots used for the beta basis expansion (from model_fitted).
*         - "Beta_c_pred": list containing, for each stationary covariate:
*                           - "Beta_eval": evaluation of the beta along a grid.
*                           - "Abscissa": grid (element abscissa_ev).
*         - "Bnc_pred": list containing, for each non-stationary covariate:
*                      - "basis_coeff": list, one element for each unit to be predicted, with the recomputed coefficients of the basis expansion of the beta.
*                      - "basis_num": number of basis used for the beta basis expansion (from model_fitted).
*                      - "basis_type": type of basis used for the beta basis expansion (from model_fitted).
*                      - "knots": knots used for the beta basis expansion (from model_fitted).
*         - "Beta_nc_pred": list containing, for each non-stationary covariate:
*                           - "Beta_eval": list containing, for each unit to be predicted, the evaluation of the beta along a grid.
*                           - "Abscissa": grid (element abscissa_ev).
* @details NB: Covariates of units to be predicted have to be sampled in the same sample points for which the training data have been (t_points).
*              Covariates basis expansion for the units to be predicted has to be done with respect to the basis used for the covariates in the training set
*/
//
// [[Rcpp::export]]
Rcpp::List predict_FMGWR(Rcpp::List coeff_stationary_cov_to_pred,
                         Rcpp::List coeff_non_stationary_cov_to_pred,
                         Rcpp::NumericMatrix coordinates_non_stationary_to_pred,   
                         int units_to_be_predicted,
                         Rcpp::NumericVector abscissa_ev,
                         Rcpp::List model_fitted,
                         int n_knots_smoothing_pred = 100,
                         int n_intervals_quadrature = 100,
                         Rcpp::Nullable<int> num_threads = R_NilValue)
{
    Rcout << "Functional Mixed Geographically Weighted Regression predictor" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT


    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FMGWR_;                              //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _NON_STATIONARY_ = FDAGWR_COVARIATES_TYPES::NON_STATIONARY;      //enum for non-stationary covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    if(units_to_be_predicted <= 0){ Rcout << "Number of unit to be predicted has to be a positive number" << std::endl;}
    //checking that the model_fitted contains a fit from FMSGWR_ESC
    wrap_predict_input<_FGWR_ALGO_>(model_fitted);
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF KNOTS TO PERFORM SMOOTHING ON THE RESPONSE WITHOUT THE NON-STATIONARY COMPONENTS
    int n_knots_smoothing_y_new = wrap_and_check_n_knots_smoothing(n_knots_smoothing_pred);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);



    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _estimation_iter_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter};     //Exact or Cascade estimation
    std::string _bc_                = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_            = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _bnc_               = std::string{FDAGWR_B_NAMES::bnc};                                //bnc
    std::string _beta_nc_           = std::string{FDAGWR_BETAS_NAMES::beta_nc};                        //beta_nc
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _partial_residuals_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res};               //partial residuals 
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                          //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                       //response reconstruction weights
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};     //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};     //stationary betas
    std::string _cov_no_stat_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_NON_STATIONARY_>()}; //event-dependent training covariates
    std::string _beta_no_stat_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_NON_STATIONARY_>()}; //event-dependent betas
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations
    std::string _cascade_estimate_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cascade_estimate};    //if using in cascade-estimation


    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[_elem_for_pred_];
    //list with partial residuals
    Rcpp::List partial_residuals = fitted_model[_partial_residuals_];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[_input_info_];
    //list with elements of the response
    Rcpp::List response_input            = training_input[_response_];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input      = training_input[_response_rec_w_];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[_cov_stat_];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[_beta_stat_];
    //list with elements of events-dependent covariates
    Rcpp::List non_stationary_cov_input          = training_input[_cov_no_stat_];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_non_stationary_cov_input     = training_input[_beta_no_stat_];

    //ESTIMATION TECHNIQUE
    bool in_cascade_estimation = training_input[_cascade_estimate_];
    //DOMAIN INFORMATION
    std::size_t n_train = training_input[_n_];
    _FD_INPUT_TYPE_ a   = training_input[_a_];
    _FD_INPUT_TYPE_ b   = training_input[_b_];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);      //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[_abscissa_];              //abscissa point for which the training data are discretized
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[_n_basis_];
    std::string basis_type_response_   = response_input[_t_basis_];
    std::size_t degree_basis_response_ = response_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[_coeff_basis_]); 
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[_n_basis_];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[_t_basis_];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[_coeff_basis_]);  
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_       = stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(stationary_cov_input[_coeff_basis_]);
    //NON STATIONARY COV    
    std::size_t q_NC                                          = non_stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_non_stationary_cov_ = non_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_non_stationary_cov_  = non_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_non_stationary_cov_ = non_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_non_stationary_cov_       = non_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_non_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_non_stationary_cov_.data(),knots_non_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_non_stationary_cov_ = wrap_covariates_coefficients<_NON_STATIONARY_>(non_stationary_cov_input[_coeff_basis_]);
    std::vector<double> lambda_non_stationary_cov_ = non_stationary_cov_input[_penalties_];
    auto coordinates_non_stationary_               = reader_data<_DATA_TYPE_,_NAN_REM_>(non_stationary_cov_input[_coords_]);     
    double kernel_bandwith_non_stationary_cov_     = non_stationary_cov_input[_bdw_ker_];  
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[_bc_];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[_coeff_basis_]);  //sono tutte Lc_jx1
        Bc.push_back(Bc_i);}
    //NON-STATIONAY BETAS  
    std::vector<std::size_t> number_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_non_stationary_cov_  = beta_non_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_non_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_non_stationary_cov_.data(),knots_beta_non_stationary_cov_.size()); 
    //PARTIAL RESIDUALS
    auto c_tilde_hat = reader_data<_DATA_TYPE_,_NAN_REM_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat}]);

    ////////////////////////////////////////
    /////   TRAINING OBJECT CREATION   /////
    ////////////////////////////////////////
    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    //non-stationary (Eta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_NC(knots_beta_non_stationary_cov_eigen_w_, 
                                                    degree_basis_beta_non_stationary_cov_, 
                                                    number_basis_beta_non_stationary_cov_, 
                                                    q_NC);


    //PENALIZATION MATRICES                                               
    //non-stationary
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_NC(std::move(bs_NC),lambda_non_stationary_cov_);
    std::size_t Lnc = R_NC.L();
    std::vector<std::size_t> Lnc_j = R_NC.Lj();
    //additional info stationary
    std::size_t Lc = std::reduce(number_basis_beta_stationary_cov_.cbegin(),number_basis_beta_stationary_cov_.cend(),static_cast<std::size_t>(0));
    std::vector<std::size_t> Lc_j = number_basis_beta_stationary_cov_;


    //MODEL FITTED COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y_train_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_y_train_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_train_(std::move(coefficients_response_),std::move(basis_y_train_));
    //sttaionary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_train_(coefficients_stationary_cov_,
                                                               q_C,
                                                               basis_types_stationary_cov_,
                                                               degree_basis_stationary_cov_,
                                                               number_basis_stationary_cov_,
                                                               knots_stationary_cov_eigen_w_,
                                                               basis_fac);
    //events covariates
    functional_data_covariates<_DOMAIN_,_NON_STATIONARY_> x_NC_fd_train_(coefficients_non_stationary_cov_,
                                                                        q_NC,
                                                                        basis_types_non_stationary_cov_,
                                                                        degree_basis_non_stationary_cov_,
                                                                        number_basis_non_stationary_cov_,
                                                                        knots_non_stationary_cov_eigen_w_,
                                                                        basis_fac);


    //wrapping all the functional elements in a functional_matrix
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> eta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_NC);
    //phi: a sparse functional matrix n_trainx(n_train*Ly), where L is the number of basis for the response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >; 
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> phi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(*basis_response_,n_train,number_basis_response_);
    //y_train: a column vector of dimension n_trainx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_train_,number_threads);
    //Xc_train: a functional matrix of dimension n_trainxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_train_,number_threads);
    //Xnc_train: a functional matrix of dimension n_trainxqnc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xnc_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_NON_STATIONARY_>(x_NC_fd_train_,number_threads);



    //////////////////////////////////////////////
    ///// WRAPPING COVARIATES TO BE PREDICTED ////
    //////////////////////////////////////////////
    // stationary
    //covariates names
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_to_be_pred_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov_to_pred); 
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(units_to_be_predicted,coefficients_stationary_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //non stationary
    //covariates names
    std::vector<std::string> names_non_stationary_cov_ = wrap_covariates_names<_NON_STATIONARY_>(coeff_non_stationary_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_non_stationary_cov_to_be_pred_ = wrap_covariates_coefficients<_NON_STATIONARY_>(coeff_non_stationary_cov_to_pred); 
    for(std::size_t i = 0; i < q_NC; ++i){   
        check_dim_input<_NON_STATIONARY_>(number_basis_non_stationary_cov_[i],coefficients_non_stationary_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_NON_STATIONARY_>(units_to_be_predicted,coefficients_non_stationary_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}

    //TO BE PREDICTED COVARIATES  
    //stationary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_to_be_pred_(coefficients_stationary_cov_to_be_pred_,
                                                                         q_C,
                                                                         basis_types_stationary_cov_,
                                                                         degree_basis_stationary_cov_,
                                                                         number_basis_stationary_cov_,
                                                                         knots_stationary_cov_eigen_w_,
                                                                         basis_fac);
    //non-stationary covariates
    functional_data_covariates<_DOMAIN_,_NON_STATIONARY_>   x_NC_fd_to_be_pred_(coefficients_non_stationary_cov_to_be_pred_,
                                                                                q_NC,
                                                                                basis_types_non_stationary_cov_,
                                                                                degree_basis_non_stationary_cov_,
                                                                                number_basis_non_stationary_cov_,
                                                                                knots_non_stationary_cov_eigen_w_,
                                                                                basis_fac);

    //Xc_new: a functional matrix of dimension n_newxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_to_be_pred_,number_threads);                                                               
    //Xnc_new: a functional matrix of dimension n_newxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xnc_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_NON_STATIONARY_>(x_NC_fd_to_be_pred_,number_threads);
    //map containing the X
    std::map<std::string,functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>> X_new = {
        {std::string{covariate_type<_STATIONARY_>()},    Xc_new},
        {std::string{covariate_type<_NON_STATIONARY_>()},Xnc_new}};

    ////////////////////////////////////////
    /////////        CONSTRUCTING W   //////
    ////////////////////////////////////////
    //distances
    auto coordinates_non_stationary_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_non_stationary_to_pred);
    check_dim_input<_NON_STATIONARY_>(units_to_be_predicted,coordinates_non_stationary_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_NON_STATIONARY_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_non_stationary_to_pred_.cols(),"coordinates matrix columns");
    distance_matrix_pred<_DISTANCE_> distances_non_stationary_to_pred_(std::move(coordinates_non_stationary_),std::move(coordinates_non_stationary_to_pred_));
    distances_non_stationary_to_pred_.compute_distances();
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    //functional weight matrix
    //non-stationary
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_,_KERNEL_,_DISTANCE_> W_NC_pred(rec_weights_y_fd_,
                                                                                                                                                                                      std::move(distances_non_stationary_to_pred_),
                                                                                                                                                                                      kernel_bandwith_non_stationary_cov_,
                                                                                                                                                                                      number_threads,
                                                                                                                                                                                      true);
    W_NC_pred.compute_weights_pred();  
    //Wnc_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Wnc_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_>(W_NC_pred,number_threads);
    //map containing the W
    std::map<std::string,std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>>> W_new = {
        {std::string{covariate_type<_NON_STATIONARY_>()},Wnc_pred}};


    //fgwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(Bc),
                                                                                                 std::move(omega),
                                                                                                 q_C,
                                                                                                 Lc,
                                                                                                 Lc_j,
                                                                                                 std::move(eta),
                                                                                                 q_NC,
                                                                                                 Lnc,
                                                                                                 Lnc_j,
                                                                                                 std::move(phi),
                                                                                                 number_basis_response_,
                                                                                                 std::move(c_tilde_hat),
                                                                                                 std::move(y_train),
                                                                                                 std::move(Xc_train),
                                                                                                 std::move(Xnc_train),
                                                                                                 std::move(R_NC.PenalizationMatrix()),
                                                                                                 a,
                                                                                                 b,
                                                                                                 n_intervals,
                                                                                                 n_train,
                                                                                                 number_threads,
                                                                                                 in_cascade_estimation);

    Rcout << "Prediction" << std::endl;                                                                                             

    //retrieve partial residuals
    fwr_predictor->computePartialResiduals();
    //compute the new b for the non-stationary covariates
    fwr_predictor->computeBNew(W_new);
    //compute the beta for stationary covariates
    fwr_predictor->computeStationaryBetas();            
    //compute the beta for non-stationary covariates
    fwr_predictor->computeNonStationaryBetas();   
    //perform prediction
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_pred = fwr_predictor->predict(X_new);
    //evaluating the betas   
    fwr_predictor->evalBetas(abscissa_points_ev_);
    //evaluating the prediction
    std::vector< std::vector< _FD_OUTPUT_TYPE_>> y_pred_ev = fwr_predictor->evalPred(y_pred,abscissa_points_ev_);
    //smoothing of the prediction
    auto y_pred_smooth_coeff = fwr_predictor->smoothPred(y_pred,*basis_pred,knots_smoothing_pred);

    Rcout << "Prediction done" << std::endl;  

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fwr_predictor->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 names_non_stationary_cov_,
                                                 basis_types_beta_non_stationary_cov_,
                                                 number_basis_beta_non_stationary_cov_,
                                                 knots_beta_non_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {});
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fwr_predictor->betas(),
                                           abscissa_points_ev_,
                                           names_stationary_cov_,
                                           names_non_stationary_cov_,
                                           {},
                                           {});
    //predictions evaluations
    Rcpp::List y_pred_ev_R = wrap_prediction_to_R_list<_FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_>(y_pred_ev,
                                                                                          abscissa_points_ev_,
                                                                                          y_pred_smooth_coeff,
                                                                                          basis_type_response_,
                                                                                          number_basis_response_,
                                                                                          degree_basis_response_,
                                                                                          knots_smoothing_pred);

    //returning element                                       
    Rcpp::List l;
    //predictor
    l[_model_name_ + "_predictor"] = "predictor_" + std::string{algo_type<_FGWR_ALGO_>()};
    l[_estimation_iter_] = estimation_iter(in_cascade_estimation);
    //predictions
    l[std::string{FDAGWR_HELPERS_for_PRED_NAMES::pred}] = y_pred_ev_R;
    //stationary covariate basis expansion coefficients for beta_c
    l[_bc_ + "_pred"]  = b_coefficients[_bc_];
    //beta_c
    l[_beta_c_ + "_pred"] = betas[_beta_c_];
    //event-dependent covariate basis expansion coefficients for beta_nc
    l[_bnc_ + "_pred"]  = b_coefficients[_bnc_];
    //beta_nc
    l[_beta_nc_ + "_pred"] = betas[_beta_nc_];

    return l;
}





/*!
* @brief Fitting a Functional Geographically Weighted Regression model. The covariates are functional objects, non-stationary covariates (NC), that vary depending on spatial coordinates.
*        The functional response is already reconstructed according to the method proposed by Bortolotti et Al. (2024) (link below)
* @param y_points matrix of double containing the raw response: each row represents a specific abscissa for which the response evaluation is available, each column a statistical unit. Response is a already reconstructed.
* @param t_points vector of double with the abscissa points with respect of the raw evaluations of y_points are available (length of t_points is equal to the number of rows of y_points).
* @param left_extreme_domain double indicating the left extreme of the functional data domain (not necessarily the smaller element in t_points).
* @param right_extreme_domain double indicating the right extreme of the functional data domain (not necessarily the biggest element in t_points).
* @param coeff_y_points matrix of double containing the coefficient of response's basis expansion: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param knots_y_points vector of double with the abscissa points with respect which the basis expansions of the response and response reconstruction weights are performed (all elements contained in [a,b]). 
* @param degree_basis_y_points non-negative integer: the degree of the basis used for the basis expansion of the (functional) response. Default explained below (can be NULL).
* @param n_basis_y_points positive integer: number of basis for the basis expansion of the (functional) response. It must match number of rows of coeff_y_points. Default explained below (can be NULL).
* @param coeff_rec_weights_y_points matrix of double containing the coefficients of the basis expansion of the weights to reconstruct the (functional) response: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param degree_basis_rec_weights_y_points non-negative integer: the degree of the basis used for response reconstruction weights. Default explained below (can be NULL).
* @param n_basis_rec_weights_y_points positive integer: number of basis for the basis expansion of response reconstruction weights. It must match number of rows of coeff_rec_weights_y_points. Default explained below (can be NULL).
* @param coeff_non_stationary_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th non-stationary covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                                 The name of the i-th element is the name of the i-th non-stationary covariate (default: "reg.NCi" if no name present).
* @param basis_types_non_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th non-stationary covariate basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_non_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the non-stationary covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_non_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th non-stationary covariate. Default explained below (can be NULL).
* @param n_basis_non_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th non-stationary covariate. It must match number of rows of the i-th element of coeff_non_stationary_cov. Default explained below (can be NULL).
* @param penalization_non_stationary_cov vector of non-negative double: element i-th is the penalization used for the i-th non-stationary covariate.
* @param coordinates_non_stationary matrix of double containing the UTM coordinates of the non-stationary site of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
* @param kernel_bandwith_non_stationary positive double indicating the bandwith of the gaussian kernel used to smooth the distances within non-stationary sites.
* @param knots_beta_non_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the non-stationary covariates functional regression coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_non_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th non-stationary covariate functional regression coefficient. Default explained below (can be NULL).
* @param n_basis_beta_non_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th non-stationary covariate functional regression coefficient. Default explained below (can be NULL).
* @param n_intervals_quadrature number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: 100).
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine.
* @param basis_type_y_points string containing the type of basis used for the functional response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_type_rec_weights_y_points string containing the type of basis used for the weights to reconstruct the functional response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_non_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariate functional regression coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @return an R list containing:
*         - "FGWR": string containing the type of fwr used ("FGWR")
*         - "Bnc": a list containing, for each non-stationary covariate regression coefficent (each element is named with the element names in the list coeff_non_stationary_cov (default, if not given: "CovNC*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Lnc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_non_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_non_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_non_stationary_cov)
*         - "Beta_nc": a list containing, for each non-stationary covariate regression coefficent (each element is named with the element names in the list coeff_non_stationary_cov (default, if not given: "CovNC*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)  
*         - "predictor_info": a list containing information of the fitted model to perform predictions for new statistical units:
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response (element n_basis_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response. Possible values: "bsplines", "constant". (element basis_type_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response (element degree_basis_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response (element coeff_y_points).
*                                              - "ResponseReconstructionWeights": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response reconstruction weights (element n_basis_rec_weights_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response reconstruction weights. Possible values: "bsplines", "constant". (element basis_type_rec_weights_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response reconstruction weights (element degree_basis_rec_weights_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response reconstruction weights (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response reconstruction weights (element coeff_rec_weights_y_points).                                                           
*                                              - "cov_NonStationary": list:
*                                                            - "number_covariates": number of non-stationary covariates (length of coeff_non_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional non-stationary covariates (respective elements of n_basis_non_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional non-stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_non_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional non-stationary covariates (respective elements of degrees_basis_non_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional non-stationary covariates (element knots_non_stationary_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional non-stationary covariates (respective elements of coeff_non_stationary_cov).
*                                                            - "penalizations": vector containing the penalizations of the non-stationary covariates (respective elements of penalization_non_stationary_cov)
*                                                            - "coordinates": UTM coordinates of the non-stationary sites of the training data (element coordinates_non_stationary).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the non-stationary sites (element kernel_bandwith_non_stationary).
*                                              - "beta_NonStationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element n_basis_beta_non_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_non_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element degrees_basis_beta_non_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element knots_beta_non_stationary_cov).
*                                              - "a": domain left extreme  (element left_extreme_domain).
*                                              - "b": domain right extreme (element right_extreme_domain).
*                                              - "abscissa": abscissa for which the evaluations of the functional data are available (element t_points).
* @details constant basis are used, for a covariate, if it resembles a scalar shape. It consists of a straight line with y-value equal to 1 all over the data domain.
*          Can be seen as a B-spline basis with degree 0, number of basis 1, using one knot, consequently having only one coefficient for the only basis for each statistical unit.
*          fdagwr sets all the feats accordingly if reads constant basis.
*          However, recall that the response is a functional datum, as the regressors coefficients. Since the package's basis variety could be hopefully enlarged in the future 
*          (for example, introducing Fourier basis for handling data that present periodical behaviors), the input parameters regarding basis types for response, response reconstruction
*          weights and regressors coefficients are left at the end of the input list, and defaulted as NULL. Consequently they will use a B-spline basis system, and should NOT use a constant basis,
*          Recall to perform externally the basis expansion before using the package, and afterwards passing basis types, degree and number and basis expansion coefficients and knots coherently
* @note a little excursion about degree and number of basis passed as input. For each specific covariate, or the response, if using B-spline basis, remember that number of knots = number of basis - degree + 1. 
*       By default, if passing NULL, fdagwr uses a cubic B-spline system of basis, the number of basis is computed coherently from the number of knots (that is the only mandatory input parameter).
*       Passing only the degree of the bsplines, the number of basis used will be set accordingly, and viceversa if passing only the number of basis. 
*       But, take care that the number of basis used has to match the number of rows of coefficients matrix (for EACH type of basis). If not, an exception is thrown. No problems arise if letting fdagwr defaulting the number of basis.
*       For response and response reconstruction weights, degree and number of basis consist of integer, and can be NULL. For all the regressors, and their coefficients, the inputs consist of vector of integers: 
*       if willing to pass a default parameter, all the vector has to be defaulted (if passing NULL, a vector with all 3 for the degrees is passed, for example)
* @link https://www.researchgate.net/publication/377251714_Weighted_Functional_Data_Analysis_for_the_Calibration_of_a_Ground_Motion_Model_in_Italy @endlink
*/
//
// [[Rcpp::export]]
Rcpp::List FGWR(Rcpp::NumericMatrix y_points,
                Rcpp::NumericVector t_points,
                double left_extreme_domain,
                double right_extreme_domain,
                Rcpp::NumericMatrix coeff_y_points,
                Rcpp::NumericVector knots_y_points,
                Rcpp::Nullable<int> degree_basis_y_points,
                Rcpp::Nullable<int> n_basis_y_points,
                Rcpp::NumericMatrix coeff_rec_weights_y_points,
                Rcpp::Nullable<int> degree_basis_rec_weights_y_points,
                Rcpp::Nullable<int> n_basis_rec_weights_y_points,
                Rcpp::List coeff_non_stationary_cov,
                Rcpp::Nullable<Rcpp::CharacterVector> basis_types_non_stationary_cov,
                Rcpp::NumericVector knots_non_stationary_cov,
                Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_non_stationary_cov,
                Rcpp::Nullable<Rcpp::IntegerVector> n_basis_non_stationary_cov,
                Rcpp::NumericVector penalization_non_stationary_cov,
                Rcpp::NumericMatrix coordinates_non_stationary,
                double kernel_bandwith_non_stationary,
                Rcpp::NumericVector knots_beta_non_stationary_cov,
                Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_beta_non_stationary_cov,
                Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_non_stationary_cov,
                int n_intervals_quadrature = 100,
                Rcpp::Nullable<int> num_threads = R_NilValue,
                std::string basis_type_y_points = "bsplines",
                std::string basis_type_rec_weights_y_points = "bsplines",
                Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue,
                Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_non_stationary_cov = R_NilValue)
{
    Rcout << "Functional Geographically Weighted Regression" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT

    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FGWR_;                               //fgwr type (estimating stationary -> event-dependent -> station-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _NON_STATIONARY_ = FDAGWR_COVARIATES_TYPES::NON_STATIONARY;      //enum for non stationary covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);


    //  RESPONSE
    //raw data
    auto response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(y_points);       //Eigen dense matrix type (auto is necessary )
    //number of statistical units
    std::size_t number_of_statistical_units_ = response_.cols();
    //coefficients matrix
    auto coefficients_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_y_points);
    auto coefficients_response_out_ = coefficients_response_;
    //reconstruction weights coefficients matrix
    auto coefficients_rec_weights_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_rec_weights_y_points);
    auto coefficients_rec_weights_response_out_ = coefficients_rec_weights_response_;

    //  ABSCISSA POINTS of response
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = wrap_abscissas(t_points,left_extreme_domain,right_extreme_domain);
    // wrapper into eigen
    check_dim_input<_RESPONSE_>(response_.rows(), abscissa_points_.size(), "points for evaluation of raw data vector");   //check that size of abscissa points and number of evaluations of fd raw data coincide
    FDAGWR_TRAITS::Dense_Matrix abscissa_points_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(abscissa_points_.data(),abscissa_points_.size(),1);
    _FD_INPUT_TYPE_ a = left_extreme_domain;
    _FD_INPUT_TYPE_ b = right_extreme_domain;


    //  KNOTS (for basis expansion and for smoothing)
    //response
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = wrap_abscissas(knots_y_points,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    //non stationary cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_non_stationary_cov_ = wrap_abscissas(knots_non_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_non_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_non_stationary_cov_.data(),knots_non_stationary_cov_.size());
    //beta events cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_non_stationary_cov_ = wrap_abscissas(knots_beta_non_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_non_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_non_stationary_cov_.data(),knots_beta_non_stationary_cov_.size());


    //  COVARIATES names, coefficients and how many (q_), for every type   
    //non stationary
    std::vector<std::string> names_non_stationary_cov_ = wrap_covariates_names<_NON_STATIONARY_>(coeff_non_stationary_cov);
    std::size_t q_NC = names_non_stationary_cov_.size();        //number of events related covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_non_stationary_cov_ = wrap_covariates_coefficients<_NON_STATIONARY_>(coeff_non_stationary_cov);


    //  BASIS TYPES
    //response
    std::string basis_type_response_ = wrap_and_check_basis_type<_RESPONSE_>(basis_type_y_points);
    //response reconstruction weights
    std::string basis_type_rec_weights_response_ = wrap_and_check_basis_type<_REC_WEIGHTS_>(basis_type_rec_weights_y_points);
    //non stationary
    std::vector<std::string> basis_types_non_stationary_cov_ = wrap_and_check_basis_type<_NON_STATIONARY_>(basis_types_non_stationary_cov,q_NC);
    //beta non stationary cov 
    std::vector<std::string> basis_types_beta_non_stationary_cov_ = wrap_and_check_basis_type<_NON_STATIONARY_>(basis_types_beta_non_stationary_cov,q_NC);


    //  BASIS NUMBER AND DEGREE: checking matrix coefficients dimensions: rows: number of basis; cols: number of statistical units
    //response
    auto number_and_degree_basis_response_ = wrap_and_check_basis_number_and_degree<_RESPONSE_>(n_basis_y_points,degree_basis_y_points,knots_response_.size(),basis_type_response_);
    std::size_t number_basis_response_ = number_and_degree_basis_response_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::size_t degree_basis_response_ = number_and_degree_basis_response_[std::string{FDAGWR_FEATS::degree_basis_string}];
    check_dim_input<_RESPONSE_>(number_basis_response_,coefficients_response_.rows(),"response coefficients matrix rows");
    check_dim_input<_RESPONSE_>(number_of_statistical_units_,coefficients_response_.cols(),"response coefficients matrix columns");     
    //response reconstruction weights
    auto number_and_degree_basis_rec_weights_response_ = wrap_and_check_basis_number_and_degree<_REC_WEIGHTS_>(n_basis_rec_weights_y_points,degree_basis_rec_weights_y_points,knots_response_.size(),basis_type_rec_weights_response_);
    std::size_t number_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::size_t degree_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[std::string{FDAGWR_FEATS::degree_basis_string}];
    check_dim_input<_REC_WEIGHTS_>(number_basis_rec_weights_response_,coefficients_rec_weights_response_.rows(),"response reconstruction weights coefficients matrix rows");
    check_dim_input<_REC_WEIGHTS_>(number_of_statistical_units_,coefficients_rec_weights_response_.cols(),"response reconstruction weights coefficients matrix columns");     
    //non stationary cov    
    auto number_and_degree_basis_non_stationary_cov_ = wrap_and_check_basis_number_and_degree<_NON_STATIONARY_>(n_basis_non_stationary_cov,degrees_basis_non_stationary_cov,knots_non_stationary_cov_.size(),q_NC,basis_types_non_stationary_cov_);
    std::vector<std::size_t> number_basis_non_stationary_cov_ = number_and_degree_basis_non_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_non_stationary_cov_ = number_and_degree_basis_non_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    for(std::size_t i = 0; i < q_NC; ++i){   
        check_dim_input<_NON_STATIONARY_>(number_basis_non_stationary_cov_[i],coefficients_non_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_NON_STATIONARY_>(number_of_statistical_units_,coefficients_non_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta non stationary cov
    auto number_and_degree_basis_beta_non_stationary_cov_ = wrap_and_check_basis_number_and_degree<_NON_STATIONARY_>(n_basis_beta_non_stationary_cov,degrees_basis_beta_non_stationary_cov,knots_beta_non_stationary_cov_.size(),q_NC,basis_types_beta_non_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_non_stationary_cov_ = number_and_degree_basis_beta_non_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_beta_non_stationary_cov_ = number_and_degree_basis_beta_non_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];


    //  DISTANCES
    //non stationary    DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_non_stationary_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_non_stationary);
    auto coordinates_non_stationary_out_ = coordinates_non_stationary_;
    check_dim_input<_NON_STATIONARY_>(number_of_statistical_units_,coordinates_non_stationary_.rows(),"coordinates matrix rows");
    check_dim_input<_NON_STATIONARY_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_non_stationary_.cols(),"coordinates matrix columns");
    distance_matrix<_DISTANCE_> distances_non_stationary_cov_(std::move(coordinates_non_stationary_),number_threads);


    //  PENALIZATION TERMS: checking their consistency
    //non stationary
    std::vector<double> lambda_non_stationary_cov_ = wrap_and_check_penalizations<_NON_STATIONARY_>(penalization_non_stationary_cov,q_NC);


    //  KERNEL BANDWITH
    //non stationary
    double kernel_bandwith_non_stationary_cov_ = wrap_and_check_kernel_bandwith<_NON_STATIONARY_>(kernel_bandwith_non_stationary);


    ////////////////////////////////////////
    /////    END PARAMETERS WRAPPING   /////
    ////////////////////////////////////////



    ////////////////////////////////
    /////    OBJECT CREATION   /////
    ////////////////////////////////


    //DISTANCES
    //non stationary
    distances_non_stationary_cov_.compute_distances();



    //BASIS SYSTEMS FOR THE BETAS
    //non stationary (Eta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_NC(knots_beta_non_stationary_cov_eigen_w_, 
                                                    degree_basis_beta_non_stationary_cov_, 
                                                    number_basis_beta_non_stationary_cov_, 
                                                    q_NC);

    
    
    //PENALIZATION MATRICES
    //non stationary
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_NC(std::move(bs_NC),lambda_non_stationary_cov_);
    std::size_t Lnc = R_NC.L();
    std::vector<std::size_t> Lnc_j = R_NC.Lj();



    //FD OBJECTS: RESPONSE and COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_(std::move(coefficients_response_),std::move(basis_response_));
    
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    
    //non stationary covariates
    functional_data_covariates<_DOMAIN_,_NON_STATIONARY_> x_NC_fd_(coefficients_non_stationary_cov_,
                                                                   q_NC,
                                                                   basis_types_non_stationary_cov_,
                                                                   degree_basis_non_stationary_cov_,
                                                                   number_basis_non_stationary_cov_,
                                                                   knots_non_stationary_cov_eigen_w_,
                                                                   basis_fac);
    

    //FUNCTIONAL WEIGHT MATRIX                                                     
    //non stationary
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_,_KERNEL_,_DISTANCE_> W_NC(rec_weights_y_fd_,
                                                                                                                                                                                 std::move(distances_non_stationary_cov_),
                                                                                                                                                                                 kernel_bandwith_non_stationary_cov_,
                                                                                                                                                                                 number_threads);
    W_NC.compute_weights();                                                                         



    ///////////////////////////////
    /////    FGWR ALGORITHM   /////
    ///////////////////////////////
    //wrapping all the functional elements in a functional_matrix

    //y: a column vector of dimension nx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_,number_threads);
    //Xnc: a functional matrix of dimension nxqnc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xnc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_NON_STATIONARY_>(x_NC_fd_,number_threads);
    //Wnc: n diagonal functional matrices of dimension nxn
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Wnc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_>(W_NC,number_threads);
    //eta: a sparse functional matrix of dimension qncxLnc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> eta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_NC);


    //fgwr algorithm
    auto fgwr_algo = fwr_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(y),
                                                                                   std::move(Xnc),
                                                                                   std::move(Wnc),
                                                                                   std::move(R_NC.PenalizationMatrix()),
                                                                                   std::move(eta),
                                                                                   q_NC,
                                                                                   Lnc,
                                                                                   Lnc_j,
                                                                                   a,
                                                                                   b,
                                                                                   n_intervals,
                                                                                   abscissa_points_,
                                                                                   number_of_statistical_units_,
                                                                                   number_threads);

    Rcout << "Model fitting" << std::endl;                                                                                     
   
    //computing the b
    fgwr_algo->compute();
    //evaluating the betas   
    fgwr_algo->evalBetas();

    Rcout << "Model fitted" << std::endl;    

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fgwr_algo->bCoefficients(),
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 names_non_stationary_cov_,
                                                 basis_types_beta_non_stationary_cov_,
                                                 number_basis_beta_non_stationary_cov_,
                                                 knots_beta_non_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {});
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fgwr_algo->betas(),
                                           abscissa_points_,
                                           {},
                                           names_non_stationary_cov_,
                                           {},
                                           {});


    //returning element
    Rcpp::List l;
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _bnc_               = std::string{FDAGWR_B_NAMES::bnc};                                 //bc
    std::string _beta_nc_           = std::string{FDAGWR_BETAS_NAMES::beta_nc};                         //beta_c
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                            //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                         //response reconstruction weights
    std::string _cov_no_stat_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_NON_STATIONARY_>()};   //stationary training covariates
    std::string _beta_no_stat_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_NON_STATIONARY_>()};   //stationary betas  
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations

    //regression model used 
    l[_model_name_] = std::string{algo_type<_FGWR_ALGO_>()};
    //non-stationary covariate basis expansion coefficients for beta_nc
    l[_bnc_]  = b_coefficients[_bnc_];
    //beta_nc
    l[_beta_nc_] = betas[_beta_nc_];
    //elements needed to perform prediction
    Rcpp::List elem_for_pred;       
    Rcpp::List inputs_info;                         //containing training data information needed for prediction purposes

    //adding all the elements of the training set
    //input of y
    Rcpp::List response_input;
    response_input[_n_basis_]     = number_basis_response_;
    response_input[_t_basis_]     = basis_type_response_;
    response_input[_deg_basis_]   = degree_basis_response_;
    response_input[_knots_basis_] = knots_response_;
    response_input[_coeff_basis_] = Rcpp::wrap(coefficients_response_out_);
    inputs_info[_response_]       = response_input;
    //input of w for y  
    Rcpp::List response_rec_w_input;
    response_rec_w_input[_n_basis_]     = number_basis_rec_weights_response_;
    response_rec_w_input[_t_basis_]     = basis_type_rec_weights_response_;
    response_rec_w_input[_deg_basis_]   = degree_basis_rec_weights_response_;
    response_rec_w_input[_knots_basis_] = knots_response_;
    response_rec_w_input[_coeff_basis_] = Rcpp::wrap(coefficients_rec_weights_response_out_);
    inputs_info[_response_rec_w_]       = response_rec_w_input;
    //input of NC
    Rcpp::List NC_input;
    NC_input[_q_]              = q_NC;
    NC_input[_n_basis_]        = number_basis_non_stationary_cov_;
    NC_input[_t_basis_]        = basis_types_non_stationary_cov_;
    NC_input[_deg_basis_]      = degree_basis_non_stationary_cov_;
    NC_input[_knots_basis_]    = knots_non_stationary_cov_;
    NC_input[_coeff_basis_]    = toRList(coefficients_non_stationary_cov_,false);
    NC_input[_penalties_]      = lambda_non_stationary_cov_;
    NC_input[_coords_]         = Rcpp::wrap(coordinates_non_stationary_out_);
    NC_input[_bdw_ker_]        = kernel_bandwith_non_stationary;
    inputs_info[_cov_no_stat_] = NC_input;
    //input of Beta NC   
    Rcpp::List beta_NC_input;
    beta_NC_input[_n_basis_]     = number_basis_beta_non_stationary_cov_;
    beta_NC_input[_t_basis_]     = basis_types_beta_non_stationary_cov_;
    beta_NC_input[_deg_basis_]   = degree_basis_beta_non_stationary_cov_;
    beta_NC_input[_knots_basis_] = knots_beta_non_stationary_cov_;
    inputs_info[_beta_no_stat_]  = beta_NC_input;
    //domain
    inputs_info[_n_]           = number_of_statistical_units_;
    inputs_info[_a_]           = a;
    inputs_info[_b_]           = b;
    inputs_info[_abscissa_]    = abscissa_points_;
    //adding all the elements of the training set to perform prediction
    elem_for_pred[_input_info_] = inputs_info;
    l[_elem_for_pred_]          = elem_for_pred;

    return l;
}


/*!
* @brief Function to perform predictions on new statistical units using a fitted Functional Geographically Weighted Regression model. Non-stationary betas have to be recomputed in the new locations.
* @param coeff_non_stationary_cov_to_pred list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th non-stationary covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit to be predicted.
* @param coordinates_non_stationary_to_pred matrix of double containing the UTM coordinates of the non-stationary site of new statistical units: each row represents a statistical unit to be predicted, each column a coordinate (2 columns).
* @param units_to_be_predicted number of units to be predicted
* @param abscissa_ev abscissa for which then evaluating the predicted reponse and betas, stationary and non-stationary, which have to be recomputed
* @param model_fitted: output of FGWR: an R list containing:
*         - "FGWR": string containing the type of fwr used ("FGWR")
*         - "Bnc": a list containing, for each non-stationary covariate regression coefficent (each element is named with the element names in the list coeff_non_stationary_cov (default, if not given: "CovNC*")) a list with:
*                 - "basis_coeff": a list, containg, for each unit, a Lnc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_non_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_non_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_non_stationary_cov)
*         - "Beta_nc": a list containing, for each non-stationary covariate regression coefficent (each element is named with the element names in the list coeff_non_stationary_cov (default, if not given: "CovNC*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)  
*         - "predictor_info": a list containing information of the fitted model to perform predictions for new statistical units:
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response (element n_basis_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response. Possible values: "bsplines", "constant". (element basis_type_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response (element degree_basis_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response (element coeff_y_points).
*                                              - "ResponseReconstructionWeights": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response reconstruction weights (element n_basis_rec_weights_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response reconstruction weights. Possible values: "bsplines", "constant". (element basis_type_rec_weights_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response reconstruction weights (element degree_basis_rec_weights_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response reconstruction weights (element knots_y_points).
*                                                            - "basis_coeff": matrix containing the coefficients of the basis expansion of the functional response reconstruction weights (element coeff_rec_weights_y_points).                                                           
*                                              - "cov_NonStationary": list:
*                                                            - "number_covariates": number of non-stationary covariates (length of coeff_non_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional non-stationary covariates (respective elements of n_basis_non_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional non-stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_non_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional non-stationary covariates (respective elements of degrees_basis_non_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional non-stationary covariates (element knots_non_stationary_cov).
*                                                            - "basis_coeff": list containing the matrices with the coefficients of the basis expansion of the functional non-stationary covariates (respective elements of coeff_non_stationary_cov).
*                                                            - "penalizations": vector containing the penalizations of the non-stationary covariates (respective elements of penalization_non_stationary_cov)
*                                                            - "coordinates": UTM coordinates of the non-stationary sites of the training data (element coordinates_non_stationary).
*                                                            - "kernel_bwd": bandwith of the gaussian kernel used to smooth distances of the non-stationary sites (element kernel_bandwith_non_stationary).
*                                              - "beta_NonStationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element n_basis_beta_non_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_non_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element degrees_basis_beta_non_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the non-stationary covariates (element knots_beta_non_stationary_cov).
*                                              - "a": domain left extreme  (element left_extreme_domain).
*                                              - "b": domain right extreme (element right_extreme_domain).
*                                              - "abscissa": abscissa for which the evaluations of the functional data are available (element t_points).
* @param n_knots_smoothing_pred number of knots used to smooth predicted response and non-stationary, obtaining basis expansion coefficients with respect to the training basis (default: 100).
* @param n_intervals_quadrature number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: 100).
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine.
* @return an R list containing:
*         - "FGWR_predictor": string containing the model used to predict ("predictor_FGWR")
*         - "prediction": list containing:
*                         - "evaluation": list containing the evaluation of the prediction:
*                                          - "prediction_ev": list containing, for each unit to be predicted, the raw evaluations of the predicted response.
*                                          - "abscissa_ev": the abscissa points for which the prediction evaluation is available (element abscissa_ev).
*                         - "fd": list containing the prediction functional description:
*                                          - "prediction_basis_coeff": matrix containing the prediction basis expansion coefficients (each row a basis, each column a new statistical unit)
*                                          - "prediction_basis_type": basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_basis_num": number of basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_basis_deg": degree of basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_knots": knots used for the predicted response smoothing (n_knots_smoothing_pred equally spaced knots in the functional datum domain)
*         - "Bnc_pred": list containing, for each non-stationary covariate:
*                      - "basis_coeff": list, one element for each unit to be predicted, with the recomputed coefficients of the basis expansion of the beta.
*                      - "basis_num": number of basis used for the beta basis expansion (from model_fitted).
*                      - "basis_type": type of basis used for the beta basis expansion (from model_fitted).
*                      - "knots": knots used for the beta basis expansion (from model_fitted).
*         - "Beta_nc_pred": list containing, for each non-stationary covariate:
*                           - "Beta_eval": list containing, for each unit to be predicted, the evaluation of the beta along a grid.
*                           - "Abscissa": grid (element abscissa_ev).
* @details NB: Covariates of units to be predicted have to be sampled in the same sample points for which the training data have been (t_points).
*              Covariates basis expansion for the units to be predicted has to be done with respect to the basis used for the covariates in the training set
*/
//
// [[Rcpp::export]]
Rcpp::List predict_FGWR(Rcpp::List coeff_non_stationary_cov_to_pred,
                        Rcpp::NumericMatrix coordinates_non_stationary_to_pred,   
                        int units_to_be_predicted,
                        Rcpp::NumericVector abscissa_ev,
                        Rcpp::List model_fitted,
                        int n_knots_smoothing_pred = 100,
                        int n_intervals_quadrature = 100,
                        Rcpp::Nullable<int> num_threads = R_NilValue)
{
    Rcout << "Functional Geographically Weighted Regression predictor" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT


    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FGWR_;                               //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _NON_STATIONARY_ = FDAGWR_COVARIATES_TYPES::NON_STATIONARY;      //enum for non-stationary covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    if(units_to_be_predicted <= 0){ Rcout << "Number of unit to be predicted has to be a positive number" << std::endl;}
    //checking that the model_fitted contains a fit from FMSGWR_ESC
    wrap_predict_input<_FGWR_ALGO_>(model_fitted);
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF KNOTS TO PERFORM SMOOTHING ON THE RESPONSE WITHOUT THE NON-STATIONARY COMPONENTS
    int n_knots_smoothing_y_new = wrap_and_check_n_knots_smoothing(n_knots_smoothing_pred);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA MIDPOINT QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);


    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _bnc_               = std::string{FDAGWR_B_NAMES::bnc};                                //bc
    std::string _beta_nc_           = std::string{FDAGWR_BETAS_NAMES::beta_nc};                        //beta_c
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                            //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                         //response reconstruction weights
    std::string _cov_no_stat_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_NON_STATIONARY_>()};   //stationary training covariates
    std::string _beta_no_stat_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_NON_STATIONARY_>()};   //stationary betas  
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations

    //list with the fitted model
    Rcpp::List fitted_model   = model_fitted[_elem_for_pred_];
    //lists with the input of the training
    Rcpp::List training_input = fitted_model[_input_info_];
    //list with elements of the response
    Rcpp::List response_input       = training_input[_response_];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input = training_input[_response_rec_w_];
    //list with elements of events-dependent covariates
    Rcpp::List non_stationary_cov_input      = training_input[_cov_no_stat_];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_non_stationary_cov_input = training_input[_beta_no_stat_];


    //DOMAIN INFORMATION
    std::size_t n_train = training_input[_n_];
    _FD_INPUT_TYPE_ a   = training_input[_a_];
    _FD_INPUT_TYPE_ b   = training_input[_b_];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);     //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[_abscissa_];             //abscissa point for which the training data are discretized
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[_n_basis_];
    std::string basis_type_response_   = response_input[_t_basis_];
    std::size_t degree_basis_response_ = response_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[_coeff_basis_]);  
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[_n_basis_];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[_t_basis_];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[_coeff_basis_]);  
    //NON STATIONARY COV    
    std::size_t q_NC                                          = non_stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_non_stationary_cov_ = non_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_non_stationary_cov_  = non_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_non_stationary_cov_ = non_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_non_stationary_cov_       = non_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_non_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_non_stationary_cov_.data(),knots_non_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_non_stationary_cov_ = wrap_covariates_coefficients<_NON_STATIONARY_>(non_stationary_cov_input[_coeff_basis_]);
    std::vector<double> lambda_non_stationary_cov_ = non_stationary_cov_input[_penalties_];
    auto coordinates_non_stationary_               = reader_data<_DATA_TYPE_,_NAN_REM_>(non_stationary_cov_input[_coords_]);     
    double kernel_bandwith_non_stationary_cov_     = non_stationary_cov_input[_bdw_ker_];  
    //NON-STATIONAY BETAS  
    std::vector<std::size_t> number_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_non_stationary_cov_  = beta_non_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_non_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_non_stationary_cov_.data(),knots_beta_non_stationary_cov_.size()); 


    ////////////////////////////////////////
    /////   TRAINING OBJECT CREATION   /////
    ////////////////////////////////////////
    //BASIS SYSTEMS FOR THE BETAS
    //non-stationary (Eta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_NC(knots_beta_non_stationary_cov_eigen_w_, 
                                                    degree_basis_beta_non_stationary_cov_, 
                                                    number_basis_beta_non_stationary_cov_, 
                                                    q_NC);


    //PENALIZATION MATRICES                                               
    //non-stationary
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_NC(std::move(bs_NC),lambda_non_stationary_cov_);
    std::size_t Lnc = R_NC.L();
    std::vector<std::size_t> Lnc_j = R_NC.Lj();



    //MODEL FITTED RESPONSE and COVARIATES
    //response
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y_train_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_y_train_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_train_(std::move(coefficients_response_),std::move(basis_y_train_));
    //events covariates
    functional_data_covariates<_DOMAIN_,_NON_STATIONARY_> x_NC_fd_train_(coefficients_non_stationary_cov_,
                                                                        q_NC,
                                                                        basis_types_non_stationary_cov_,
                                                                        degree_basis_non_stationary_cov_,
                                                                        number_basis_non_stationary_cov_,
                                                                        knots_non_stationary_cov_eigen_w_,
                                                                        basis_fac);


    //wrapping all the functional elements in a functional_matrix
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> eta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_NC);
    //y_train: a column vector of dimension n_trainx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_train_,number_threads);
    //Xnc_train: a functional matrix of dimension n_trainxqnc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xnc_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_NON_STATIONARY_>(x_NC_fd_train_,number_threads);



    //////////////////////////////////////////////
    ///// WRAPPING COVARIATES TO BE PREDICTED ////
    //////////////////////////////////////////////
    // stationary
    //non stationary
    //covariates names
    std::vector<std::string> names_non_stationary_cov_ = wrap_covariates_names<_NON_STATIONARY_>(coeff_non_stationary_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_non_stationary_cov_to_be_pred_ = wrap_covariates_coefficients<_NON_STATIONARY_>(coeff_non_stationary_cov_to_pred); 
    for(std::size_t i = 0; i < q_NC; ++i){   
        check_dim_input<_NON_STATIONARY_>(number_basis_non_stationary_cov_[i],coefficients_non_stationary_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_NON_STATIONARY_>(units_to_be_predicted,coefficients_non_stationary_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}

    //TO BE PREDICTED COVARIATES  
    //non-stationary covariates
    functional_data_covariates<_DOMAIN_,_NON_STATIONARY_>   x_NC_fd_to_be_pred_(coefficients_non_stationary_cov_to_be_pred_,
                                                                                q_NC,
                                                                                basis_types_non_stationary_cov_,
                                                                                degree_basis_non_stationary_cov_,
                                                                                number_basis_non_stationary_cov_,
                                                                                knots_non_stationary_cov_eigen_w_,
                                                                                basis_fac);

                                                            
    //Xnc_new: a functional matrix of dimension n_newxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xnc_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_NON_STATIONARY_>(x_NC_fd_to_be_pred_,number_threads);
    //map containing the X
    std::map<std::string,functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>> X_new = {
        {std::string{covariate_type<_NON_STATIONARY_>()},Xnc_new}};

    ////////////////////////////////////////
    /////////        CONSTRUCTING W   //////
    ////////////////////////////////////////
    //distances
    auto coordinates_non_stationary_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_non_stationary_to_pred);
    check_dim_input<_NON_STATIONARY_>(units_to_be_predicted,coordinates_non_stationary_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_NON_STATIONARY_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_non_stationary_to_pred_.cols(),"coordinates matrix columns");
    distance_matrix_pred<_DISTANCE_> distances_non_stationary_to_pred_(std::move(coordinates_non_stationary_),std::move(coordinates_non_stationary_to_pred_));
    distances_non_stationary_to_pred_.compute_distances();
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    //functional weight matrix
    //non-stationary
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_,_KERNEL_,_DISTANCE_> W_NC_pred(rec_weights_y_fd_,
                                                                                                                                                                                      std::move(distances_non_stationary_to_pred_),
                                                                                                                                                                                      kernel_bandwith_non_stationary_cov_,
                                                                                                                                                                                      number_threads,
                                                                                                                                                                                      true);
    W_NC_pred.compute_weights_pred();  
    //Wnc_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Wnc_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_>(W_NC_pred,number_threads);
    //map containing the W
    std::map<std::string,std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>>> W_new = {
        {std::string{covariate_type<_NON_STATIONARY_>()},Wnc_pred}};


    //fwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(eta),
                                                                                                 q_NC,
                                                                                                 Lnc,
                                                                                                 Lnc_j,
                                                                                                 std::move(y_train),
                                                                                                 std::move(Xnc_train),
                                                                                                 std::move(R_NC.PenalizationMatrix()),
                                                                                                 a,
                                                                                                 b,
                                                                                                 n_intervals,
                                                                                                 n_train,
                                                                                                 number_threads);

    Rcout << "Prediction" << std::endl;
                                                                                                 
    //compute the new b for the non-stationary covariates
    fwr_predictor->computeBNew(W_new);          
    //compute the beta for non-stationary covariates
    fwr_predictor->computeNonStationaryBetas();   
    //perform prediction
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_pred = fwr_predictor->predict(X_new);
    //evaluating the betas   
    fwr_predictor->evalBetas(abscissa_points_ev_);
    //evaluating the prediction
    std::vector< std::vector< _FD_OUTPUT_TYPE_>> y_pred_ev = fwr_predictor->evalPred(y_pred,abscissa_points_ev_);
    //smoothing of the prediction
    auto y_pred_smooth_coeff = fwr_predictor->smoothPred(y_pred,*basis_pred,knots_smoothing_pred);

    Rcout << "Prediction done" << std::endl;

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fwr_predictor->bCoefficients(),
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 names_non_stationary_cov_,
                                                 basis_types_beta_non_stationary_cov_,
                                                 number_basis_beta_non_stationary_cov_,
                                                 knots_beta_non_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {});
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fwr_predictor->betas(),
                                           abscissa_points_ev_,
                                           {},
                                           names_non_stationary_cov_,
                                           {},
                                           {});
    //predictions evaluations
    Rcpp::List y_pred_ev_R = wrap_prediction_to_R_list<_FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_>(y_pred_ev,
                                                                                          abscissa_points_ev_,
                                                                                          y_pred_smooth_coeff,
                                                                                          basis_type_response_,
                                                                                          number_basis_response_,
                                                                                          degree_basis_response_,
                                                                                          knots_smoothing_pred);

    //returning element                                       
    Rcpp::List l;
    //predictor
    l[_model_name_ + "_predictor"] = "predictor_" + std::string{algo_type<_FGWR_ALGO_>()};
    //predictions
    l[std::string{FDAGWR_HELPERS_for_PRED_NAMES::pred}] = y_pred_ev_R;
    //event-dependent covariate basis expansion coefficients for beta_e
    l[_bnc_ + "_pred"]  = b_coefficients[_bnc_];
    //beta_e
    l[_beta_nc_ + "_pred"] = betas[_beta_nc_];

    return l;
}





/*!
* @brief Fitting a Functional Weighted Regression model. The covariates are functional objects, stationary covariates (C), constant over geographical space. 
*        The functional response is already reconstructed according to the method proposed by Bortolotti et Al. (2024) (link below)
* @param y_points matrix of double containing the raw response: each row represents a specific abscissa for which the response evaluation is available, each column a statistical unit. Response is a already reconstructed.
* @param t_points vector of double with the abscissa points with respect of the raw evaluations of y_points are available (length of t_points is equal to the number of rows of y_points).
* @param left_extreme_domain double indicating the left extreme of the functional data domain (not necessarily the smaller element in t_points).
* @param right_extreme_domain double indicating the right extreme of the functional data domain (not necessarily the biggest element in t_points).
* @param coeff_y_points matrix of double containing the coefficient of response's basis expansion: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param knots_y_points vector of double with the abscissa points with respect which the basis expansions of the response and response reconstruction weights are performed (all elements contained in [a,b]). 
* @param degree_basis_y_points non-negative integer: the degree of the basis used for the basis expansion of the (functional) response. Default explained below (can be NULL).
* @param n_basis_y_points positive integer: number of basis for the basis expansion of the (functional) response. It must match number of rows of coeff_y_points. Default explained below (can be NULL).
* @param coeff_rec_weights_y_points matrix of double containing the coefficients of the basis expansion of the weights to reconstruct the (functional) response: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param degree_basis_rec_weights_y_points non-negative integer: the degree of the basis used for response reconstruction weights. Default explained below (can be NULL).
* @param n_basis_rec_weights_y_points positive integer: number of basis for the basis expansion of response reconstruction weights. It must match number of rows of coeff_rec_weights_y_points. Default explained below (can be NULL).
* @param coeff_stationary_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                             The name of the i-th element is the name of the i-th stationary covariate (default: "reg.Ci" if no name present).
* @param basis_types_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariate basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate. Default explained below (can be NULL).
* @param n_basis_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate. It must match number of rows of the i-th element of coeff_stationary_cov. Default explained below (can be NULL).
* @param penalization_stationary_cov vector of non-negative double: element i-th is the penalization used for the i-th stationary covariate.
* @param knots_beta_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates functional regression coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate functional regression coefficients. Default explained below (can be NULL).
* @param n_basis_beta_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate functional regression coefficients. Default explained below (can be NULL).
* @param n_intervals_quadrature number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: 100).
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine.
* @param basis_type_y_points string containing the type of basis used for the functional response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_type_rec_weights_y_points string containing the type of basis used for the weights to reconstruct the functional response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariate functional regression coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @return an R list containing:
*         - "FGWR": string containing the type of fwr used ("FWR")
*         - "Bc": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "basis_coeff": a Lc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective element of basis_types_beta_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stationary_cov)
*         - "Beta_c": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Beta_eval": a vector of double containing the discrete evaluations of the stationary beta
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "predictor_info": a list containing information of the fitted model to perform predictions for new statistical units:
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response (element n_basis_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response. Possible values: "bsplines", "constant". (element basis_type_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response (element degree_basis_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response (element knots_y_points).
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates": number of stationary covariates (length of coeff_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional stationary covariates (respective elements of n_basis_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional stationary covariates (respective elements of degrees_basis_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional stationary covariates (element knots_stationary_cov).
*                                              - "beta_Stationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element n_basis_beta_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element degrees_basis_beta_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the stationary covariates (element knots_beta_stationary_cov).                                                            
*                                              - "a": domain left extreme  (element left_extreme_domain).
*                                              - "b": domain right extreme (element right_extreme_domain).
*                                              - "abscissa": abscissa for which the evaluations of the functional data are available (element t_points).
* @details constant basis are used, for a covariate, if it resembles a scalar shape. It consists of a straight line with y-value equal to 1 all over the data domain.
*          Can be seen as a B-spline basis with degree 0, number of basis 1, using one knot, consequently having only one coefficient for the only basis for each statistical unit.
*          fdagwr sets all the feats accordingly if reads constant basis.
*          However, recall that the response is a functional datum, as the regressors coefficients. Since the package's basis variety could be hopefully enlarged in the future 
*          (for example, introducing Fourier basis for handling data that present periodical behaviors), the input parameters regarding basis types for response, response reconstruction
*          weights and regressors coefficients are left at the end of the input list, and defaulted as NULL. Consequently they will use a B-spline basis system, and should NOT use a constant basis,
*          Recall to perform externally the basis expansion before using the package, and afterwards passing basis types, degree and number and basis expansion coefficients and knots coherently
* @note a little excursion about degree and number of basis passed as input. For each specific covariate, or the response, if using B-spline basis, remember that number of knots = number of basis - degree + 1. 
*       By default, if passing NULL, fdagwr uses a cubic B-spline system of basis, the number of basis is computed coherently from the number of knots (that is the only mandatory input parameter).
*       Passing only the degree of the bsplines, the number of basis used will be set accordingly, and viceversa if passing only the number of basis. 
*       But, take care that the number of basis used has to match the number of rows of coefficients matrix (for EACH type of basis). If not, an exception is thrown. No problems arise if letting fdagwr defaulting the number of basis.
*       For response and response reconstruction weights, degree and number of basis consist of integer, and can be NULL. For all the regressors, and their coefficients, the inputs consist of vector of integers: 
*       if willing to pass a default parameter, all the vector has to be defaulted (if passing NULL, a vector with all 3 for the degrees is passed, for example)
* @link https://www.researchgate.net/publication/377251714_Weighted_Functional_Data_Analysis_for_the_Calibration_of_a_Ground_Motion_Model_in_Italy @endlink
*/
//
// [[Rcpp::export]]
Rcpp::List FWR(Rcpp::NumericMatrix y_points,
                Rcpp::NumericVector t_points,
                double left_extreme_domain,
                double right_extreme_domain,
                Rcpp::NumericMatrix coeff_y_points,
                Rcpp::NumericVector knots_y_points,
                Rcpp::Nullable<int> degree_basis_y_points,
                Rcpp::Nullable<int> n_basis_y_points,
                Rcpp::NumericMatrix coeff_rec_weights_y_points,
                Rcpp::Nullable<int> degree_basis_rec_weights_y_points,
                Rcpp::Nullable<int> n_basis_rec_weights_y_points,
                Rcpp::List coeff_stationary_cov,
                Rcpp::Nullable<Rcpp::CharacterVector> basis_types_stationary_cov,
                Rcpp::NumericVector knots_stationary_cov,
                Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_stationary_cov,
                Rcpp::Nullable<Rcpp::IntegerVector> n_basis_stationary_cov,
                Rcpp::NumericVector penalization_stationary_cov,
                Rcpp::NumericVector knots_beta_stationary_cov,
                Rcpp::Nullable<Rcpp::IntegerVector> degrees_basis_beta_stationary_cov,
                Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_stationary_cov,
                int n_intervals_quadrature = 100,
                Rcpp::Nullable<int> num_threads = R_NilValue,
                std::string basis_type_y_points = "bsplines",
                std::string basis_type_rec_weights_y_points = "bsplines",
                Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue)
{
    Rcout << "Functional Weighted Regression" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT

    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FWR_;                                //fgwr type (estimating stationary)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA MIDPOINT QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);


    //  RESPONSE
    //raw data
    auto response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(y_points);       //Eigen dense matrix type (auto is necessary )
    //number of statistical units
    std::size_t number_of_statistical_units_ = response_.cols();
    //coefficients matrix
    auto coefficients_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_y_points);
    //reconstruction weights coefficients matrix
    auto coefficients_rec_weights_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_rec_weights_y_points);
    auto coefficients_rec_weights_response_out_ = coefficients_rec_weights_response_;

    //  ABSCISSA POINTS of response
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = wrap_abscissas(t_points,left_extreme_domain,right_extreme_domain);
    // wrapper into eigen
    check_dim_input<_RESPONSE_>(response_.rows(), abscissa_points_.size(), "points for evaluation of raw data vector");   //check that size of abscissa points and number of evaluations of fd raw data coincide
    FDAGWR_TRAITS::Dense_Matrix abscissa_points_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(abscissa_points_.data(),abscissa_points_.size(),1);
    _FD_INPUT_TYPE_ a = left_extreme_domain;
    _FD_INPUT_TYPE_ b = right_extreme_domain;


    //  KNOTS (for basis expansion and for smoothing)
    //response
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = wrap_abscissas(knots_y_points,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    //stationary cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_ = wrap_abscissas(knots_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    //beta stationary cov
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = wrap_abscissas(knots_beta_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());

    //  COVARIATES names, coefficients and how many (q_), for every type
    //stationary 
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov);
    std::size_t q_C = names_stationary_cov_.size();    //number of stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov);    


    //  BASIS TYPES
    //response
    std::string basis_type_response_ = wrap_and_check_basis_type<_RESPONSE_>(basis_type_y_points);
    //response reconstruction weights
    std::string basis_type_rec_weights_response_ = wrap_and_check_basis_type<_REC_WEIGHTS_>(basis_type_rec_weights_y_points);
    //stationary
    std::vector<std::string> basis_types_stationary_cov_ = wrap_and_check_basis_type<_STATIONARY_>(basis_types_stationary_cov,q_C);
    //beta stationary cov 
    std::vector<std::string> basis_types_beta_stationary_cov_ = wrap_and_check_basis_type<_STATIONARY_>(basis_types_beta_stationary_cov,q_C);


    //  BASIS NUMBER AND DEGREE: checking matrix coefficients dimensions: rows: number of basis; cols: number of statistical units
    //response
    auto number_and_degree_basis_response_ = wrap_and_check_basis_number_and_degree<_RESPONSE_>(n_basis_y_points,degree_basis_y_points,knots_response_.size(),basis_type_response_);
    std::size_t number_basis_response_ = number_and_degree_basis_response_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::size_t degree_basis_response_ = number_and_degree_basis_response_[std::string{FDAGWR_FEATS::degree_basis_string}];
    check_dim_input<_RESPONSE_>(number_basis_response_,coefficients_response_.rows(),"response coefficients matrix rows");
    check_dim_input<_RESPONSE_>(number_of_statistical_units_,coefficients_response_.cols(),"response coefficients matrix columns");     
    //response reconstruction weights
    auto number_and_degree_basis_rec_weights_response_ = wrap_and_check_basis_number_and_degree<_REC_WEIGHTS_>(n_basis_rec_weights_y_points,degree_basis_rec_weights_y_points,knots_response_.size(),basis_type_rec_weights_response_);
    std::size_t number_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::size_t degree_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[std::string{FDAGWR_FEATS::degree_basis_string}];
    check_dim_input<_REC_WEIGHTS_>(number_basis_rec_weights_response_,coefficients_rec_weights_response_.rows(),"response reconstruction weights coefficients matrix rows");
    check_dim_input<_REC_WEIGHTS_>(number_of_statistical_units_,coefficients_rec_weights_response_.cols(),"response reconstruction weights coefficients matrix columns");     
    //stationary cov
    auto number_and_degree_basis_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_stationary_cov,degrees_basis_stationary_cov,knots_stationary_cov_.size(),q_C,basis_types_stationary_cov_);
    std::vector<std::size_t> number_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stationary cov
    auto number_and_degree_basis_beta_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_beta_stationary_cov,degrees_basis_beta_stationary_cov,knots_beta_stationary_cov_.size(),q_C,basis_types_beta_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[std::string{FDAGWR_FEATS::n_basis_string}];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[std::string{FDAGWR_FEATS::degree_basis_string}];


    //  PENALIZATION TERMS: checking their consistency
    //stationary
    std::vector<double> lambda_stationary_cov_ = wrap_and_check_penalizations<_STATIONARY_>(penalization_stationary_cov,q_C);

    ////////////////////////////////////////
    /////    END PARAMETERS WRAPPING   /////
    ////////////////////////////////////////



    ////////////////////////////////
    /////    OBJECT CREATION   /////
    ////////////////////////////////


    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    
    
    //PENALIZATION MATRICES
    //stationary
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_C(std::move(bs_C),lambda_stationary_cov_);
    std::size_t Lc = R_C.L();
    std::vector<std::size_t> Lc_j = R_C.Lj();


    //FD OBJECTS: RESPONSE and COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_(std::move(coefficients_response_),std::move(basis_response_));
    
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    
    //stationary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_(coefficients_stationary_cov_,
                                                              q_C,
                                                              basis_types_stationary_cov_,
                                                              degree_basis_stationary_cov_,
                                                              number_basis_stationary_cov_,
                                                              knots_stationary_cov_eigen_w_,
                                                              basis_fac);
    
    //FUNCTIONAL WEIGHT MATRIX
    //stationary
    functional_weight_matrix_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATIONARY_> W_C(rec_weights_y_fd_,
                                                                                                                                                    number_threads);
    W_C.compute_weights();                                                      


    ///////////////////////////////
    /////    FGWR ALGORITHM   /////
    ///////////////////////////////
    //wrapping all the functional elements in a functional_matrix

    //y: a column vector of dimension nx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_,number_threads);
    //Xc: a functional matrix of dimension nxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_,number_threads);
    //Wc: a diagonal functional matrix of dimension nxn
    functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Wc = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATIONARY_>(W_C,number_threads);
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);


    //fgwr algorithm
    auto fgwr_algo = fwr_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(y),                                                                       
                                                                                   std::move(Xc),
                                                                                   std::move(Wc),
                                                                                   std::move(R_C.PenalizationMatrix()),
                                                                                   std::move(omega),
                                                                                   q_C,
                                                                                   Lc,
                                                                                   Lc_j,
                                                                                   a,
                                                                                   b,
                                                                                   n_intervals,
                                                                                   abscissa_points_,
                                                                                   number_of_statistical_units_,
                                                                                   number_threads);

    Rcout << "Model fitting" << std::endl;                                                                                     
                                                                                 
    //computing the b
    fgwr_algo->compute();
    //evaluating the betas   
    fgwr_algo->evalBetas();

    Rcout << "Model fitted" << std::endl; 

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fgwr_algo->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {});
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fgwr_algo->betas(),
                                           abscissa_points_,
                                           names_stationary_cov_,
                                           {},
                                           {},
                                           {});

    
    //returning element
    Rcpp::List l;
    //names main outputs
    std::string _model_name_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _bc_            = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_        = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _elem_for_pred_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _input_info_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                        //response
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};   //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};   //stationary betas  
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    //domain
    std::string _n_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations

    //regression model used 
    l[_model_name_] = std::string{algo_type<_FGWR_ALGO_>()};
    //non-stationary covariate basis expansion coefficients for beta_nc
    l[_bc_]  = b_coefficients[_bc_];
    //beta_nc
    l[_beta_c_] = betas[_beta_c_];
    //elements needed to perform prediction
    Rcpp::List elem_for_pred;       
    Rcpp::List inputs_info;                         //containing training data information needed for prediction purposes

    //adding all the elements of the training set
    //input of y
    Rcpp::List response_input;
    response_input[_n_basis_]     = number_basis_response_;
    response_input[_t_basis_]     = basis_type_response_;
    response_input[_deg_basis_]   = degree_basis_response_;
    response_input[_knots_basis_] = knots_response_;
    inputs_info[_response_]       = response_input;
    //input of C
    Rcpp::List C_input;
    C_input[_q_]            = q_C;
    C_input[_n_basis_]      = number_basis_stationary_cov_;
    C_input[_t_basis_]      = basis_types_stationary_cov_;
    C_input[_deg_basis_]    = degree_basis_stationary_cov_;
    C_input[_knots_basis_]  = knots_stationary_cov_;
    inputs_info[_cov_stat_] = C_input;
    //input of Beta C   
    Rcpp::List beta_C_input;
    beta_C_input[_n_basis_]     = number_basis_beta_stationary_cov_;
    beta_C_input[_t_basis_]     = basis_types_beta_stationary_cov_;
    beta_C_input[_deg_basis_]   = degree_basis_beta_stationary_cov_;
    beta_C_input[_knots_basis_] = knots_beta_stationary_cov_;
    inputs_info[_beta_stat_]     = beta_C_input;
    //domain
    inputs_info[_n_]           = number_of_statistical_units_;
    inputs_info[_a_]           = a;
    inputs_info[_b_]           = b;
    inputs_info[_abscissa_]    = abscissa_points_;
    //adding all the elements of the training set to perform prediction
    elem_for_pred[_input_info_] = inputs_info;
    l[_elem_for_pred_]          = elem_for_pred;

    return l;
}


/*!
* @brief Function to perform predictions on new statistical units using a fitted Functional Weighted Regression model.
* @param coeff_stationary_cov_to_pred list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit to be predicted.
* @param units_to_be_predicted number of units to be predicted
* @param abscissa_ev abscissa for which then evaluating the predicted reponse and betas, stationary and non-stationary, which have to be recomputed
* @param model_fitted: output of FWR: an R list containing:
*         - "FGWR": string containing the type of fwr used ("FWR")
*         - "Bc": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "basis_coeff": a Lc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective element of basis_types_beta_stationary_cov)
*                 - "basis_num": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stationary_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stationary_cov)
*         - "Beta_c": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Beta_eval": a vector of double containing the discrete evaluations of the stationary beta
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "predictor_info": a list containing information of the fitted model to perform predictions for new statistical units:
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional response (element n_basis_y_points).
*                                                            - "basis_type": basis used to make the basis expansion of the functional response. Possible values: "bsplines", "constant". (element basis_type_y_points).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional response (element degree_basis_y_points).
*                                                            - "knots": knots used to make the basis expansion of the functional response (element knots_y_points).
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates": number of stationary covariates (length of coeff_stationary_cov).
*                                                            - "basis_num": vector with the numbers of basis used to make the basis expansion of the functional stationary covariates (respective elements of n_basis_stationary_cov).
*                                                            - "basis_type": vector with type of basis used to make the basis expansion of the functional stationary covariates. Possible values: "bsplines", "constant". (respective elements of basis_types_stationary_cov).
*                                                            - "basis_deg": vector with the degree of basis used to make the basis expansion of functional stationary covariates (respective elements of degrees_basis_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of functional stationary covariates (element knots_stationary_cov).
*                                              - "beta_Stationary": list:
*                                                            - "basis_num": number of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element n_basis_beta_stationary_cov).
*                                                            - "basis_type": basis used to make the basis expansion of the functional regression coefficients of the stationary covariates. Possible values: "bsplines", "constant". (element basis_types_beta_stationary_cov).
*                                                            - "basis_deg": degree of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (element degrees_basis_beta_stationary_cov).
*                                                            - "knots": knots used to make the basis expansion of the functional regression coefficients of the stationary covariates (element knots_beta_stationary_cov).                                                            
*                                              - "a": domain left extreme  (element left_extreme_domain).
*                                              - "b": domain right extreme (element right_extreme_domain).
*                                              - "abscissa": abscissa for which the evaluations of the functional data are available (element t_points).
* @param n_knots_smoothing_pred number of knots used to smooth predicted response and non-stationary, obtaining basis expansion coefficients with respect to the training basis (default: 100).
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine.
* @return an R list containing:
*         - "FGWR_predictor": string containing the model used to predict ("predictor_FWR")
*         - "prediction": list containing:
*                         - "evaluation": list containing the evaluation of the prediction:
*                                          - "prediction_ev": list containing, for each unit to be predicted, the raw evaluations of the predicted response.
*                                          - "abscissa_ev": the abscissa points for which the prediction evaluation is available (element abscissa_ev).
*                         - "fd": list containing the prediction functional description:
*                                          - "prediction_basis_coeff": matrix containing the prediction basis expansion coefficients (each row a basis, each column a new statistical unit)
*                                          - "prediction_basis_type": basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_basis_num": number of basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_basis_deg": degree of basis used for the predicted response basis expansion (from model_fitted)
*                                          - "prediction_knots": knots used for the predicted response smoothing (n_knots_smoothing_pred equally spaced knots in the functional datum domain)
*         - "Bc_pred": list containing, for each stationary covariate:
*                      - "basis_coeff": coefficients of the basis expansion of the beta (from model_fitted).
*                      - "basis_num": number of basis used for the beta basis epxnasion (from model_fitted).
*                      - "basis_type": type of basis used for the beta basis expansion (from model_fitted).
*                      - "knots": knots used for the beta basis expansion (from model_fitted).
*         - "Beta_c_pred": list containing, for each stationary covariate:
*                           - "Beta_eval": evaluation of the beta along a grid.
*                           - "Abscissa": grid (element abscissa_ev).
* @details NB: Covariates of units to be predicted have to be sampled in the same sample points for which the training data have been (t_points).
*              Covariates basis expansion for the units to be predicted has to be done with respect to the basis used for the covariates in the training set
*/
//
// [[Rcpp::export]]
Rcpp::List predict_FWR(Rcpp::List coeff_stationary_cov_to_pred,
                        int units_to_be_predicted,
                        Rcpp::NumericVector abscissa_ev,
                        Rcpp::List model_fitted,
                        int n_knots_smoothing_pred = 100,    
                        Rcpp::Nullable<int> num_threads = R_NilValue)
{
    Rcout << "Functional Weighted Regression predictor" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT


    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FWR_;                                //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    if(units_to_be_predicted <= 0){ Rcout << "Number of unit to be predicted has to be a positive number" << std::endl;}
    //checking that the model_fitted contains a fit from FMSGWR_ESC
    wrap_predict_input<_FGWR_ALGO_>(model_fitted);
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF KNOTS TO PERFORM SMOOTHING ON THE RESPONSE WITHOUT THE NON-STATIONARY COMPONENTS
    int n_knots_smoothing_y_new = wrap_and_check_n_knots_smoothing(n_knots_smoothing_pred);


    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    //names main outputs
    std::string _model_name_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _bc_            = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_        = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _elem_for_pred_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _input_info_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                        //response
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};   //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};   //stationary betas  
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    //domain
    std::string _n_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations

    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[_elem_for_pred_];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[_input_info_];
    //list with elements of the response
    Rcpp::List response_input            = training_input[_response_];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[_cov_stat_];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[_beta_stat_];

    //DOMAIN INFORMATION
    std::size_t n_train = training_input[_n_];
    _FD_INPUT_TYPE_ a   = training_input[_a_];
    _FD_INPUT_TYPE_ b   = training_input[_b_];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);     //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[_abscissa_];             //abscissa point for which the training data are discretized
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[_n_basis_];
    std::string basis_type_response_   = response_input[_t_basis_];
    std::size_t degree_basis_response_ = response_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_ = stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size()); 
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[_bc_];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[_coeff_basis_]);  //sono tutte Lc_jx1
        Bc.push_back(Bc_i);}


    ////////////////////////////////////////
    /////   TRAINING OBJECT CREATION   /////
    ////////////////////////////////////////
    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    std::size_t Lc = std::reduce(number_basis_beta_stationary_cov_.cbegin(),number_basis_beta_stationary_cov_.cend(),static_cast<std::size_t>(0));
    std::vector<std::size_t> Lc_j = number_basis_beta_stationary_cov_;

    //wrapping all the functional elements in a functional_matrix
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);



    //////////////////////////////////////////////
    ///// WRAPPING COVARIATES TO BE PREDICTED ////
    //////////////////////////////////////////////
    // stationary
    //covariates names
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_to_be_pred_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov_to_pred); 
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(units_to_be_predicted,coefficients_stationary_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}

    //TO BE PREDICTED COVARIATES  
    //stationary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_to_be_pred_(coefficients_stationary_cov_to_be_pred_,
                                                                         q_C,
                                                                         basis_types_stationary_cov_,
                                                                         degree_basis_stationary_cov_,
                                                                         number_basis_stationary_cov_,
                                                                         knots_stationary_cov_eigen_w_,
                                                                         basis_fac);

    //Xc_new: a functional matrix of dimension n_newxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_to_be_pred_,number_threads);                                                               
    //map containing the X
    std::map<std::string,functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>> X_new = {{std::string{covariate_type<_STATIONARY_>()},Xc_new}};



    //fwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(Bc),
                                                                                                 std::move(omega),
                                                                                                 q_C,
                                                                                                 Lc,
                                                                                                 Lc_j,
                                                                                                 a,
                                                                                                 b,
                                                                                                 n_train,
                                                                                                 number_threads);

    Rcout << "Prediction" << std::endl;                                                                                             

    //compute the beta for stationary covariates
    fwr_predictor->computeStationaryBetas();            
    //perform prediction
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_pred = fwr_predictor->predict(X_new);
    //evaluating the betas   
    fwr_predictor->evalBetas(abscissa_points_ev_);
    //evaluating the prediction
    std::vector< std::vector< _FD_OUTPUT_TYPE_>> y_pred_ev = fwr_predictor->evalPred(y_pred,abscissa_points_ev_);
    //smoothing of the prediction
    auto y_pred_smooth_coeff = fwr_predictor->smoothPred<_DOMAIN_>(y_pred,*basis_pred,knots_smoothing_pred);

    Rcout << "Prediction done" << std::endl;  

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fwr_predictor->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {});
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fwr_predictor->betas(),
                                           abscissa_points_ev_,
                                           names_stationary_cov_,
                                           {},
                                           {},
                                           {});
    //predictions evaluations
    Rcpp::List y_pred_ev_R = wrap_prediction_to_R_list<_FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_>(y_pred_ev,
                                                                                          abscissa_points_ev_,
                                                                                          y_pred_smooth_coeff,
                                                                                          basis_type_response_,
                                                                                          number_basis_response_,
                                                                                          degree_basis_response_,
                                                                                          knots_smoothing_pred);

    //returning element                                       
    Rcpp::List l;
    //predictor
    l[_model_name_ + "_predictor"] = "predictor_" + std::string{algo_type<_FGWR_ALGO_>()};
    //predictions
    l[std::string{FDAGWR_HELPERS_for_PRED_NAMES::pred}] = y_pred_ev_R;
    //stationary covariate basis expansion coefficients for beta_c
    l[_bc_ + "_pred"]  = b_coefficients[_bc_];
    //beta_c
    l[_beta_c_ + "_pred"] = betas[_beta_c_];

    return l;
}









//
// [[Rcpp::export]]
Rcpp::List tune_new_betas_FMSGWR_ESC(Rcpp::NumericMatrix coordinates_events_to_pred,   
                                     Rcpp::NumericMatrix coordinates_stations_to_pred,
                                     int units_to_be_predicted,
                                     Rcpp::NumericVector abscissa_ev,
                                     Rcpp::List model_fitted,
                                     int n_intervals_quadrature = 100,
                                     Rcpp::Nullable<int> num_threads = R_NilValue)
{
    Rcout << "Functional Multi-Source Geographically Weighted Regression ESC new betas tuning" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT


    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FMSGWR_ESC_;                         //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _EVENT_ = FDAGWR_COVARIATES_TYPES::EVENT;                        //enum for event covariates
    constexpr auto _STATION_ = FDAGWR_COVARIATES_TYPES::STATION;                    //enum for station covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    if(units_to_be_predicted <= 0){ Rcout << "Number of unit to be predicted has to be a positive number" << std::endl;}
    //checking that the model_fitted contains a fit from FMSGWR_ESC
    wrap_predict_input<_FGWR_ALGO_>(model_fitted);
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA MIDPOINT QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);



    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    // NAME OF THE LIST ELEMENT COMING FROM THE FITTING MODEL FUNCTION
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _estimation_iter_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter};     //Exact or Cascade estimation
    std::string _bc_                = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_            = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _be_                = std::string{FDAGWR_B_NAMES::be};                                 //be
    std::string _beta_e_            = std::string{FDAGWR_BETAS_NAMES::beta_e};                         //beta_e
    std::string _bs_                = std::string{FDAGWR_B_NAMES::bs};                                 //bs
    std::string _beta_s_            = std::string{FDAGWR_BETAS_NAMES::beta_s};                         //beta_s
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _partial_residuals_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res};               //partial residuals 
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                        //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                     //response reconstruction weights
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};   //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};   //stationary betas
    std::string _cov_event_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_EVENT_>()};        //event-dependent training covariates
    std::string _beta_event_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_EVENT_>()};        //event-dependent betas
    std::string _cov_station_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates
    std::string _beta_station_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates    
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations
    std::string _cascade_estimate_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cascade_estimate};         //if using in cascade-estimation


    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[_elem_for_pred_];
    //list with partial residuals
    Rcpp::List partial_residuals = fitted_model[_partial_residuals_];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[_input_info_];
    //list with elements of the response
    Rcpp::List response_input            = training_input[_response_];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input      = training_input[_response_rec_w_];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[_cov_stat_];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[_beta_stat_];
    //list with elements of events-dependent covariates
    Rcpp::List events_cov_input          = training_input[_cov_event_];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_events_cov_input     = training_input[_beta_event_];
    //list with elements of stations-dependent covariates
    Rcpp::List stations_cov_input        = training_input[_cov_station_];
    //list with elements of the beta of stations-dependent covariates
    Rcpp::List beta_stations_cov_input   = training_input[_beta_station_];

    //ESTIMATION TECHNIQUE
    bool in_cascade_estimation = training_input[_cascade_estimate_];
    //DOMAIN INFORMATION
    std::size_t n_train = training_input[_n_];
    _FD_INPUT_TYPE_ a   = training_input[_a_];
    _FD_INPUT_TYPE_ b   = training_input[_b_];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);     //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_    = training_input[_abscissa_];          //abscissa point for which the training data are discretized
    //RESPONSE
    std::size_t number_basis_response_ = response_input[_n_basis_];
    std::string basis_type_response_   = response_input[_t_basis_];
    std::size_t degree_basis_response_ = response_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[_coeff_basis_]); 
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[_n_basis_];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[_t_basis_];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[_coeff_basis_]);  
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_       = stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(stationary_cov_input[_coeff_basis_]);
    //EVENTS COV    
    std::size_t q_E                                   = events_cov_input[_q_];
    std::vector<std::size_t> number_basis_events_cov_ = events_cov_input[_n_basis_];
    std::vector<std::string> basis_types_events_cov_  = events_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_events_cov_ = events_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_events_cov_       = events_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_events_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_ = wrap_covariates_coefficients<_EVENT_>(events_cov_input[_coeff_basis_]);
    std::vector<double> lambda_events_cov_ = events_cov_input[_penalties_];
    auto coordinates_events_               = reader_data<_DATA_TYPE_,_NAN_REM_>(events_cov_input[_coords_]);     
    double kernel_bandwith_events_cov_     = events_cov_input[_bdw_ker_];
    //STATIONS COV  
    std::size_t q_S                                     = stations_cov_input[_q_];
    std::vector<std::size_t> number_basis_stations_cov_ = stations_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stations_cov_  = stations_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stations_cov_ = stations_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stations_cov_       = stations_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stations_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_ = wrap_covariates_coefficients<_STATION_>(stations_cov_input[_coeff_basis_]);
    std::vector<double> lambda_stations_cov_ = stations_cov_input[_penalties_];
    auto coordinates_stations_               = reader_data<_DATA_TYPE_,_NAN_REM_>(stations_cov_input[_coords_]);
    double kernel_bandwith_stations_cov_     = stations_cov_input[_bdw_ker_];    
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[_bc_];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[_coeff_basis_]);  //Lc_j x 1
        Bc.push_back(Bc_i);}
    //EVENTS BETAS  
    std::vector<std::size_t> number_basis_beta_events_cov_ = beta_events_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_events_cov_  = beta_events_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = beta_events_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_events_cov_ = beta_events_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_events_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size()); 
    //STATIONS BETAS    
    std::vector<std::size_t> number_basis_beta_stations_cov_ = beta_stations_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stations_cov_  = beta_stations_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = beta_stations_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stations_cov_ = beta_stations_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stations_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());
    //saving the betas basis expansion coefficients for station-dependent covariates
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix>> Bs; //vettore esterno: per ogni covariata S. Interno: per ogni unità di training
    Bs.reserve(q_S);
    Rcpp::List Bs_list = model_fitted[_bs_];
    for(std::size_t i = 0; i < q_S; ++i){
        Rcpp::List Bs_i_list = Bs_list[i];
        auto Bs_i = wrap_covariates_coefficients<_STATION_>(Bs_i_list[_coeff_basis_]);  //Ls_j x 1
        Bs.push_back(Bs_i);}
    //PARTIAL RESIDUALS
    auto c_tilde_hat = reader_data<_DATA_TYPE_,_NAN_REM_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat}]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> A_E_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_A__}]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> B_E_for_K_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_B__for_K}]);

    //covariates names
    // stationary
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(model_fitted[_bc_]);
    //events
    std::vector<std::string> names_events_cov_ = wrap_covariates_names<_EVENT_>(model_fitted[_be_]);
    //stations
    std::vector<std::string> names_stations_cov_ = wrap_covariates_names<_STATION_>(model_fitted[_bs_]);

    ////////////////////////////////////////
    /////   TRAINING OBJECT CREATION   /////
    ////////////////////////////////////////
    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    //events (Theta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_E(knots_beta_events_cov_eigen_w_, 
                                                   degree_basis_beta_events_cov_, 
                                                   number_basis_beta_events_cov_, 
                                                   q_E);
    //stations (Psi)
    basis_systems< _DOMAIN_, bsplines_basis > bs_S(knots_beta_stations_cov_eigen_w_,  
                                                   degree_basis_beta_stations_cov_, 
                                                   number_basis_beta_stations_cov_, 
                                                   q_S);


    //PENALIZATION MATRICES                                               
    //events
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_E(std::move(bs_E),lambda_events_cov_);
    std::size_t Le = R_E.L();
    std::vector<std::size_t> Le_j = R_E.Lj();
    //stations
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_S(std::move(bs_S),lambda_stations_cov_);
    std::size_t Ls = R_S.L();
    std::vector<std::size_t> Ls_j = R_S.Lj();
    
    //additional info stationary
    std::size_t Lc = std::reduce(number_basis_beta_stationary_cov_.cbegin(),number_basis_beta_stationary_cov_.cend(),static_cast<std::size_t>(0));
    std::vector<std::size_t> Lc_j = number_basis_beta_stationary_cov_;


    //MODEL FITTED COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y_train_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_y_train_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_train_(std::move(coefficients_response_),std::move(basis_y_train_));
    //sttaionary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_train_(coefficients_stationary_cov_,
                                                                    q_C,
                                                                    basis_types_stationary_cov_,
                                                                    degree_basis_stationary_cov_,
                                                                    number_basis_stationary_cov_,
                                                                    knots_stationary_cov_eigen_w_,
                                                                    basis_fac);
    //events covariates
    functional_data_covariates<_DOMAIN_,_EVENT_> x_E_fd_train_(coefficients_events_cov_,
                                                               q_E,
                                                               basis_types_events_cov_,
                                                               degree_basis_events_cov_,
                                                               number_basis_events_cov_,
                                                               knots_events_cov_eigen_w_,
                                                               basis_fac);
    
    //stations covariates
    functional_data_covariates<_DOMAIN_,_STATION_> x_S_fd_train_(coefficients_stations_cov_,
                                                                 q_S,
                                                                 basis_types_stations_cov_,
                                                                 degree_basis_stations_cov_,
                                                                 number_basis_stations_cov_,
                                                                 knots_stations_cov_eigen_w_,
                                                                 basis_fac);


    //wrapping all the functional elements in a functional_matrix
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> theta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_E);
    //psi: a sparse functional matrix of dimension qsxLs
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> psi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_S);
    //phi: a sparse functional matrix n_trainx(n_train*Ly), where L is the number of basis for the response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >; 
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> phi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(*basis_response_,n_train,number_basis_response_);
    //y_train: a column vector of dimension n_trainx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_train_,number_threads);
    //Xc_train: a functional matrix of dimension n_trainxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_train_,number_threads);
    //Xe_train: a functional matrix of dimension n_trainxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xe_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_EVENT_>(x_E_fd_train_,number_threads);
    //Xs_train: a functional matrix of dimension n_trainxqs
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xs_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATION_>(x_S_fd_train_,number_threads);



    ////////////////////////////////////////
    /////////        CONSTRUCTING W   //////
    ////////////////////////////////////////
    //distances
    auto coordinates_events_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_events_to_pred);
    check_dim_input<_EVENT_>(units_to_be_predicted,coordinates_events_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_EVENT_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_events_to_pred_.cols(),"coordinates matrix columns");
    auto coordinates_stations_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_stations_to_pred);
    check_dim_input<_STATION_>(units_to_be_predicted,coordinates_stations_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_STATION_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_stations_to_pred_.cols(),"coordinates matrix columns");
    distance_matrix_pred<_DISTANCE_> distances_events_to_pred_(std::move(coordinates_events_),std::move(coordinates_events_to_pred_));
    distance_matrix_pred<_DISTANCE_> distances_stations_to_pred_(std::move(coordinates_stations_),std::move(coordinates_stations_to_pred_));
    distances_events_to_pred_.compute_distances();
    distances_stations_to_pred_.compute_distances();
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    //functional weight matrix
    //events
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_,_KERNEL_,_DISTANCE_> W_E_pred(rec_weights_y_fd_,
                                                                                                                                                                            std::move(distances_events_to_pred_),
                                                                                                                                                                            kernel_bandwith_events_cov_,
                                                                                                                                                                            number_threads,
                                                                                                                                                                            true);
    W_E_pred.compute_weights_pred();                                                                         
    //stations
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_,_KERNEL_,_DISTANCE_> W_S_pred(rec_weights_y_fd_,
                                                                                                                                                                              std::move(distances_stations_to_pred_),
                                                                                                                                                                              kernel_bandwith_stations_cov_,
                                                                                                                                                                              number_threads,
                                                                                                                                                                              true);
    W_S_pred.compute_weights_pred();
    //We_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > We_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_>(W_E_pred,number_threads);
    //Ws_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Ws_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_>(W_S_pred,number_threads);
    //map containing the W
    std::map<std::string,std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>>> W_new = {
        {std::string{covariate_type<_EVENT_>()},  We_pred},
        {std::string{covariate_type<_STATION_>()},Ws_pred}};


    //fwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(Bc),
                                                                                                 std::move(Bs),
                                                                                                 std::move(omega),
                                                                                                 q_C,
                                                                                                 Lc,
                                                                                                 Lc_j,
                                                                                                 std::move(theta),
                                                                                                 q_E,
                                                                                                 Le,
                                                                                                 Le_j,
                                                                                                 std::move(psi),
                                                                                                 q_S,
                                                                                                 Ls,
                                                                                                 Ls_j,
                                                                                                 std::move(phi),
                                                                                                 number_basis_response_,
                                                                                                 std::move(c_tilde_hat),
                                                                                                 std::move(A_E_i),
                                                                                                 std::move(B_E_for_K_i),
                                                                                                 std::move(y_train),
                                                                                                 std::move(Xc_train),
                                                                                                 std::move(Xe_train),
                                                                                                 std::move(R_E.PenalizationMatrix()),
                                                                                                 std::move(Xs_train),
                                                                                                 std::move(R_S.PenalizationMatrix()),
                                                                                                 a,
                                                                                                 b,
                                                                                                 n_intervals,
                                                                                                 n_train,
                                                                                                 number_threads,
                                                                                                 in_cascade_estimation);

    Rcout << "Betas tuning" << std::endl;
    //retrieve partial residuals
    fwr_predictor->computePartialResiduals();
    //compute the new b for the non-stationary covariates
    fwr_predictor->computeBNew(W_new);
    //compute the beta for stationary covariates
    fwr_predictor->computeStationaryBetas();            
    //compute the beta for non-stationary covariates
    fwr_predictor->computeNonStationaryBetas();   
    //evaluating the betas   
    fwr_predictor->evalBetas(abscissa_points_ev_);
    Rcout << "Betas tuned" << std::endl;


    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fwr_predictor->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 names_events_cov_,
                                                 basis_types_beta_events_cov_,
                                                 number_basis_beta_events_cov_,
                                                 knots_beta_events_cov_,
                                                 names_stations_cov_,
                                                 basis_types_beta_stations_cov_,
                                                 number_basis_beta_stations_cov_,
                                                 knots_beta_stations_cov_);
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fwr_predictor->betas(),
                                           abscissa_points_ev_,
                                           names_stationary_cov_,
                                           {},
                                           names_events_cov_,
                                           names_stations_cov_);

    //returning element                                       
    Rcpp::List l;
    //predictor
    l[_model_name_ + "_predictor"] = "predictor_" + std::string{algo_type<_FGWR_ALGO_>()};
    l[_estimation_iter_]           = estimation_iter(in_cascade_estimation);
    //stationary covariate basis expansion coefficients for beta_c
    l[_bc_ + "_pred"]     = b_coefficients[_bc_];
    //beta_c
    l[_beta_c_ + "_pred"] = betas[_beta_c_];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[_be_ + "_pred"]     = b_coefficients[_be_];
    //beta_e
    l[_beta_e_ + "_pred"] = betas[_beta_e_];
    //station-dependent covariate basis expansion coefficients for beta_s
    l[_bs_ + "_pred"]     = b_coefficients[_bs_];
    //beta_s
    l[_beta_s_ + "_pred"] = betas[_beta_s_];

    return l;
}


//
// [[Rcpp::export]]
Rcpp::List tune_new_betas_FMSGWR_SEC(Rcpp::NumericMatrix coordinates_events_to_pred,   
                                     Rcpp::NumericMatrix coordinates_stations_to_pred,
                                     int units_to_be_predicted,
                                     Rcpp::NumericVector abscissa_ev,
                                     Rcpp::List model_fitted,                              
                                     int n_intervals_quadrature = 100,
                                     Rcpp::Nullable<int> num_threads = R_NilValue)
{
    Rcout << "Functional Multi-Source Geographically Weighted Regression SEC new betas tuning" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT


    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FMSGWR_SEC_;                         //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _EVENT_ = FDAGWR_COVARIATES_TYPES::EVENT;                        //enum for event covariates
    constexpr auto _STATION_ = FDAGWR_COVARIATES_TYPES::STATION;                    //enum for station covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    if(units_to_be_predicted <= 0){ Rcout << "Number of unit to be predicted has to be a positive number" << std::endl;}
    //checking that the model_fitted contains a fit from FMSGWR_ESC
    wrap_predict_input<_FGWR_ALGO_>(model_fitted);
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA MIDPOINT QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);


    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    // NAME OF THE LIST ELEMENT COMING FROM THE FITTING MODEL FUNCTION
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _estimation_iter_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter};     //Exact or Cascade estimation
    std::string _bc_                = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_            = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _be_                = std::string{FDAGWR_B_NAMES::be};                                 //be
    std::string _beta_e_            = std::string{FDAGWR_BETAS_NAMES::beta_e};                         //beta_e
    std::string _bs_                = std::string{FDAGWR_B_NAMES::bs};                                 //bs
    std::string _beta_s_            = std::string{FDAGWR_BETAS_NAMES::beta_s};                         //beta_s
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _partial_residuals_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res};               //partial residuals 
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                        //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                     //response reconstruction weights
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};   //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};   //stationary betas
    std::string _cov_event_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_EVENT_>()};        //event-dependent training covariates
    std::string _beta_event_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_EVENT_>()};        //event-dependent betas
    std::string _cov_station_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates
    std::string _beta_station_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates    
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations
    std::string _cascade_estimate_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cascade_estimate};    //if using in cascade-estimation



    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[_elem_for_pred_];
    //list with partial residuals
    Rcpp::List partial_residuals = fitted_model[_partial_residuals_];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[_input_info_];
    //list with elements of the response
    Rcpp::List response_input            = training_input[_response_];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input      = training_input[_response_rec_w_];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[_cov_stat_];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[_beta_stat_];
    //list with elements of events-dependent covariates
    Rcpp::List events_cov_input          = training_input[_cov_event_];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_events_cov_input     = training_input[_beta_event_];
    //list with elements of stations-dependent covariates
    Rcpp::List stations_cov_input        = training_input[_cov_station_];
    //list with elements of the beta of stations-dependent covariates
    Rcpp::List beta_stations_cov_input   = training_input[_beta_station_];

    //ESTIMATION TECHNIQUE
    bool in_cascade_estimation = training_input[_cascade_estimate_];
    //DOMAIN INFORMATION
    std::size_t n_train = training_input[_n_];
    _FD_INPUT_TYPE_ a   = training_input[_a_];
    _FD_INPUT_TYPE_ b   = training_input[_b_];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);      //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[_abscissa_];              //abscissa point for which the training data are discretized
    //RESPONSE
    std::size_t number_basis_response_ = response_input[_n_basis_];
    std::string basis_type_response_   = response_input[_t_basis_];
    std::size_t degree_basis_response_ = response_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[_coeff_basis_]); 
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[_n_basis_];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[_t_basis_];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[_coeff_basis_]);  
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_       = stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(stationary_cov_input[_coeff_basis_]);
    //EVENTS COV    
    std::size_t q_E                                   = events_cov_input[_q_];
    std::vector<std::size_t> number_basis_events_cov_ = events_cov_input[_n_basis_];
    std::vector<std::string> basis_types_events_cov_  = events_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_events_cov_ = events_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_events_cov_       = events_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_events_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_ = wrap_covariates_coefficients<_EVENT_>(events_cov_input[_coeff_basis_]);
    std::vector<double> lambda_events_cov_ = events_cov_input[_penalties_];
    auto coordinates_events_               = reader_data<_DATA_TYPE_,_NAN_REM_>(events_cov_input[_coords_]);     
    double kernel_bandwith_events_cov_     = events_cov_input[_bdw_ker_];
    //STATIONS COV  
    std::size_t q_S                                     = stations_cov_input[_q_];
    std::vector<std::size_t> number_basis_stations_cov_ = stations_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stations_cov_  = stations_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stations_cov_ = stations_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stations_cov_       = stations_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stations_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_ = wrap_covariates_coefficients<_STATION_>(stations_cov_input[_coeff_basis_]);
    std::vector<double> lambda_stations_cov_ = stations_cov_input[_penalties_];
    auto coordinates_stations_               = reader_data<_DATA_TYPE_,_NAN_REM_>(stations_cov_input[_coords_]);
    double kernel_bandwith_stations_cov_     = stations_cov_input[_bdw_ker_];    
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[_bc_];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[_coeff_basis_]);  //sono tutte Lc_jx1
        Bc.push_back(Bc_i);}
    //EVENTS BETAS  
    std::vector<std::size_t> number_basis_beta_events_cov_ = beta_events_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_events_cov_  = beta_events_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = beta_events_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_events_cov_ = beta_events_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_events_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size()); 
    //saving the betas basis expansion coefficients for events-dependent covariates
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix>> Be; //vettore esterno: per ogni covariata E. Interno: per ogni unità di training
    Be.reserve(q_E);
    Rcpp::List Be_list = model_fitted[_be_];
    for(std::size_t i = 0; i < q_E; ++i){
        Rcpp::List Be_i_list = Be_list[i];
        auto Be_i = wrap_covariates_coefficients<_EVENT_>(Be_i_list[_coeff_basis_]);
        Be.push_back(Be_i);}
    //STATIONS BETAS    
    std::vector<std::size_t> number_basis_beta_stations_cov_ = beta_stations_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stations_cov_  = beta_stations_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = beta_stations_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stations_cov_ = beta_stations_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stations_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());
    //PARTIAL RESIDUALS
    auto c_tilde_hat = reader_data<_DATA_TYPE_,_NAN_REM_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat}]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> A_S_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_A__}]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> B_S_for_K_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_B__for_K}]);

    //covariates names
    // stationary
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(model_fitted[_bc_]);
    //events
    std::vector<std::string> names_events_cov_ = wrap_covariates_names<_EVENT_>(model_fitted[_be_]);
    //stations
    std::vector<std::string> names_stations_cov_ = wrap_covariates_names<_STATION_>(model_fitted[_bs_]);

    ////////////////////////////////////////
    /////   TRAINING OBJECT CREATION   /////
    ////////////////////////////////////////
    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    //events (Theta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_E(knots_beta_events_cov_eigen_w_, 
                                                   degree_basis_beta_events_cov_, 
                                                   number_basis_beta_events_cov_, 
                                                   q_E);
    //stations (Psi)
    basis_systems< _DOMAIN_, bsplines_basis > bs_S(knots_beta_stations_cov_eigen_w_,  
                                                   degree_basis_beta_stations_cov_, 
                                                   number_basis_beta_stations_cov_, 
                                                   q_S);


    //PENALIZATION MATRICES                                               
    //events
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_E(std::move(bs_E),lambda_events_cov_);
    std::size_t Le = R_E.L();
    std::vector<std::size_t> Le_j = R_E.Lj();
    //stations
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_S(std::move(bs_S),lambda_stations_cov_);
    std::size_t Ls = R_S.L();
    std::vector<std::size_t> Ls_j = R_S.Lj();
    
    //additional info stationary
    std::size_t Lc = std::reduce(number_basis_beta_stationary_cov_.cbegin(),number_basis_beta_stationary_cov_.cend(),static_cast<std::size_t>(0));
    std::vector<std::size_t> Lc_j = number_basis_beta_stationary_cov_;


    //MODEL FITTED COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y_train_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_y_train_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_train_(std::move(coefficients_response_),std::move(basis_y_train_));
    //sttaionary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_train_(coefficients_stationary_cov_,
                                                               q_C,
                                                               basis_types_stationary_cov_,
                                                               degree_basis_stationary_cov_,
                                                               number_basis_stationary_cov_,
                                                               knots_stationary_cov_eigen_w_,
                                                               basis_fac);
    //events covariates
    functional_data_covariates<_DOMAIN_,_EVENT_> x_E_fd_train_(coefficients_events_cov_,
                                                               q_E,
                                                               basis_types_events_cov_,
                                                               degree_basis_events_cov_,
                                                               number_basis_events_cov_,
                                                               knots_events_cov_eigen_w_,
                                                               basis_fac);
    
    //stations covariates
    functional_data_covariates<_DOMAIN_,_STATION_> x_S_fd_train_(coefficients_stations_cov_,
                                                                 q_S,
                                                                 basis_types_stations_cov_,
                                                                 degree_basis_stations_cov_,
                                                                 number_basis_stations_cov_,
                                                                 knots_stations_cov_eigen_w_,
                                                                 basis_fac);


    //wrapping all the functional elements in a functional_matrix
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> theta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_E);
    //psi: a sparse functional matrix of dimension qsxLs
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> psi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_S);
    //phi: a sparse functional matrix n_trainx(n_train*Ly), where L is the number of basis for the response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >; 
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> phi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(*basis_response_,n_train,number_basis_response_);
    //y_train: a column vector of dimension n_trainx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_train_,number_threads);
    //Xc_train: a functional matrix of dimension n_trainxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_train_,number_threads);
    //Xe_train: a functional matrix of dimension n_trainxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xe_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_EVENT_>(x_E_fd_train_,number_threads);
    //Xs_train: a functional matrix of dimension n_trainxqs
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xs_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATION_>(x_S_fd_train_,number_threads);



    ////////////////////////////////////////
    /////////        CONSTRUCTING W   //////
    ////////////////////////////////////////
    //distances
    auto coordinates_events_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_events_to_pred);
    check_dim_input<_EVENT_>(units_to_be_predicted,coordinates_events_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_EVENT_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_events_to_pred_.cols(),"coordinates matrix columns");
    auto coordinates_stations_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_stations_to_pred);
    check_dim_input<_STATION_>(units_to_be_predicted,coordinates_stations_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_STATION_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_stations_to_pred_.cols(),"coordinates matrix columns");
    distance_matrix_pred<_DISTANCE_> distances_events_to_pred_(std::move(coordinates_events_),std::move(coordinates_events_to_pred_));
    distance_matrix_pred<_DISTANCE_> distances_stations_to_pred_(std::move(coordinates_stations_),std::move(coordinates_stations_to_pred_));
    distances_events_to_pred_.compute_distances();
    distances_stations_to_pred_.compute_distances();
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    //functional weight matrix
    //events
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_,_KERNEL_,_DISTANCE_> W_E_pred(rec_weights_y_fd_,
                                                                                                                                                                            std::move(distances_events_to_pred_),
                                                                                                                                                                            kernel_bandwith_events_cov_,
                                                                                                                                                                            number_threads,
                                                                                                                                                                            true);
    W_E_pred.compute_weights_pred();                                                                         
    //stations
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_,_KERNEL_,_DISTANCE_> W_S_pred(rec_weights_y_fd_,
                                                                                                                                                                              std::move(distances_stations_to_pred_),
                                                                                                                                                                              kernel_bandwith_stations_cov_,
                                                                                                                                                                              number_threads,
                                                                                                                                                                              true);
    W_S_pred.compute_weights_pred();
    //We_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > We_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_EVENT_>(W_E_pred,number_threads);
    //Ws_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Ws_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_STATION_>(W_S_pred,number_threads);
    //map containing the W
    std::map<std::string,std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>>> W_new = {
        {std::string{covariate_type<_EVENT_>()},We_pred},
        {std::string{covariate_type<_STATION_>()},Ws_pred}};


    //fwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(Bc),
                                                                                                 std::move(Be),
                                                                                                 std::move(omega),
                                                                                                 q_C,
                                                                                                 Lc,
                                                                                                 Lc_j,
                                                                                                 std::move(theta),
                                                                                                 q_E,
                                                                                                 Le,
                                                                                                 Le_j,
                                                                                                 std::move(psi),
                                                                                                 q_S,
                                                                                                 Ls,
                                                                                                 Ls_j,
                                                                                                 std::move(phi),
                                                                                                 number_basis_response_,
                                                                                                 std::move(c_tilde_hat),
                                                                                                 std::move(A_S_i),
                                                                                                 std::move(B_S_for_K_i),
                                                                                                 std::move(y_train),
                                                                                                 std::move(Xc_train),
                                                                                                 std::move(Xe_train),
                                                                                                 std::move(R_E.PenalizationMatrix()),
                                                                                                 std::move(Xs_train),
                                                                                                 std::move(R_S.PenalizationMatrix()),
                                                                                                 a,
                                                                                                 b,
                                                                                                 n_intervals,
                                                                                                 n_train,
                                                                                                 number_threads,
                                                                                                 in_cascade_estimation);

    Rcout << "Betas tuning" << std::endl;
    
    //retrieve partial residuals
    fwr_predictor->computePartialResiduals();
    //compute the new b for the non-stationary covariates
    fwr_predictor->computeBNew(W_new);
    //compute the beta for stationary covariates
    fwr_predictor->computeStationaryBetas();            
    //compute the beta for non-stationary covariates
    fwr_predictor->computeNonStationaryBetas();   
    //evaluating the betas   
    fwr_predictor->evalBetas(abscissa_points_ev_);

    Rcout << "Betas tuned" << std::endl;

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fwr_predictor->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 names_events_cov_,
                                                 basis_types_beta_events_cov_,
                                                 number_basis_beta_events_cov_,
                                                 knots_beta_events_cov_,
                                                 names_stations_cov_,
                                                 basis_types_beta_stations_cov_,
                                                 number_basis_beta_stations_cov_,
                                                 knots_beta_stations_cov_);
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fwr_predictor->betas(),
                                           abscissa_points_ev_,
                                           names_stationary_cov_,
                                           {},
                                           names_events_cov_,
                                           names_stations_cov_);


    //returning element                                       
    Rcpp::List l;
    //regression model used and estimation technique
    l[_model_name_]      = std::string{algo_type<_FGWR_ALGO_>()};
    l[_estimation_iter_] = estimation_iter(in_cascade_estimation);
    //stationary covariate basis expansion coefficients for beta_c
    l[_bc_ + "_pred"]  = b_coefficients[_bc_];
    //beta_c
    l[_beta_c_ + "_pred"] = betas[_beta_c_];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[_be_ + "_pred"]  = b_coefficients[_be_];
    //beta_e
    l[_beta_e_ + "_pred"] = betas[_beta_e_];
    //station-dependent covariate basis expansion coefficients for beta_s
    l[_bs_ + "_pred"]  = b_coefficients[_bs_];
    //beta_s
    l[_beta_s_ + "_pred"] = betas[_beta_s_];

    return l;
}


//
// [[Rcpp::export]]
Rcpp::List tune_new_betas_FMGWR(Rcpp::NumericMatrix coordinates_non_stationary_to_pred,   
                                int units_to_be_predicted,
                                Rcpp::NumericVector abscissa_ev,
                                Rcpp::List model_fitted,
                                int n_intervals_quadrature = 100,
                                Rcpp::Nullable<int> num_threads = R_NilValue)
{
    Rcout << "Functional Mixed Geographically Weighted Regression new betas tuning" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT


    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FMGWR_;                              //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _NON_STATIONARY_ = FDAGWR_COVARIATES_TYPES::NON_STATIONARY;      //enum for non-stationary covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    if(units_to_be_predicted <= 0){ Rcout << "Number of unit to be predicted has to be a positive number" << std::endl;}
    //checking that the model_fitted contains a fit from FMGWR
    wrap_predict_input<_FGWR_ALGO_>(model_fitted);
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);



    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _estimation_iter_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter};     //Exact or Cascade estimation
    std::string _bc_                = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_            = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _bnc_               = std::string{FDAGWR_B_NAMES::bnc};                                //bnc
    std::string _beta_nc_           = std::string{FDAGWR_BETAS_NAMES::beta_nc};                        //beta_nc
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _partial_residuals_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res};               //partial residuals 
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                          //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                       //response reconstruction weights
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};     //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};     //stationary betas
    std::string _cov_no_stat_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_NON_STATIONARY_>()}; //event-dependent training covariates
    std::string _beta_no_stat_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_NON_STATIONARY_>()}; //event-dependent betas
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations
    std::string _cascade_estimate_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cascade_estimate};    //if using in cascade-estimation


    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[_elem_for_pred_];
    //list with partial residuals
    Rcpp::List partial_residuals = fitted_model[_partial_residuals_];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[_input_info_];
    //list with elements of the response
    Rcpp::List response_input            = training_input[_response_];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input      = training_input[_response_rec_w_];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[_cov_stat_];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[_beta_stat_];
    //list with elements of events-dependent covariates
    Rcpp::List non_stationary_cov_input          = training_input[_cov_no_stat_];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_non_stationary_cov_input     = training_input[_beta_no_stat_];

    //ESTIMATION TECHNIQUE
    bool in_cascade_estimation = training_input[_cascade_estimate_];
    //DOMAIN INFORMATION
    std::size_t n_train = training_input[_n_];
    _FD_INPUT_TYPE_ a   = training_input[_a_];
    _FD_INPUT_TYPE_ b   = training_input[_b_];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);      //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[_abscissa_];              //abscissa point for which the training data are discretized
    //RESPONSE
    std::size_t number_basis_response_ = response_input[_n_basis_];
    std::string basis_type_response_   = response_input[_t_basis_];
    std::size_t degree_basis_response_ = response_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[_coeff_basis_]); 
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[_n_basis_];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[_t_basis_];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[_coeff_basis_]);  
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_       = stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(stationary_cov_input[_coeff_basis_]);
    //NON STATIONARY COV    
    std::size_t q_NC                                          = non_stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_non_stationary_cov_ = non_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_non_stationary_cov_  = non_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_non_stationary_cov_ = non_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_non_stationary_cov_       = non_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_non_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_non_stationary_cov_.data(),knots_non_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_non_stationary_cov_ = wrap_covariates_coefficients<_NON_STATIONARY_>(non_stationary_cov_input[_coeff_basis_]);
    std::vector<double> lambda_non_stationary_cov_ = non_stationary_cov_input[_penalties_];
    auto coordinates_non_stationary_               = reader_data<_DATA_TYPE_,_NAN_REM_>(non_stationary_cov_input[_coords_]);     
    double kernel_bandwith_non_stationary_cov_     = non_stationary_cov_input[_bdw_ker_];  
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[_bc_];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[_coeff_basis_]);  //sono tutte Lc_jx1
        Bc.push_back(Bc_i);}
    //NON-STATIONAY BETAS  
    std::vector<std::size_t> number_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_non_stationary_cov_  = beta_non_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_non_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_non_stationary_cov_.data(),knots_beta_non_stationary_cov_.size()); 
    //PARTIAL RESIDUALS
    auto c_tilde_hat = reader_data<_DATA_TYPE_,_NAN_REM_>(partial_residuals[std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat}]);

    //covariates names
    // stationary
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(model_fitted[_bc_]);
    //non stationary
    std::vector<std::string> names_non_stationary_cov_ = wrap_covariates_names<_NON_STATIONARY_>(model_fitted[_bnc_]);

    ////////////////////////////////////////
    /////   TRAINING OBJECT CREATION   /////
    ////////////////////////////////////////
    //BASIS SYSTEMS FOR THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    //non-stationary (Eta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_NC(knots_beta_non_stationary_cov_eigen_w_, 
                                                    degree_basis_beta_non_stationary_cov_, 
                                                    number_basis_beta_non_stationary_cov_, 
                                                    q_NC);


    //PENALIZATION MATRICES                                               
    //non-stationary
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_NC(std::move(bs_NC),lambda_non_stationary_cov_);
    std::size_t Lnc = R_NC.L();
    std::vector<std::size_t> Lnc_j = R_NC.Lj();
    //additional info stationary
    std::size_t Lc = std::reduce(number_basis_beta_stationary_cov_.cbegin(),number_basis_beta_stationary_cov_.cend(),static_cast<std::size_t>(0));
    std::vector<std::size_t> Lc_j = number_basis_beta_stationary_cov_;


    //MODEL FITTED COVARIATES
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y_train_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_y_train_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_train_(std::move(coefficients_response_),std::move(basis_y_train_));
    //sttaionary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_train_(coefficients_stationary_cov_,
                                                               q_C,
                                                               basis_types_stationary_cov_,
                                                               degree_basis_stationary_cov_,
                                                               number_basis_stationary_cov_,
                                                               knots_stationary_cov_eigen_w_,
                                                               basis_fac);
    //events covariates
    functional_data_covariates<_DOMAIN_,_NON_STATIONARY_> x_NC_fd_train_(coefficients_non_stationary_cov_,
                                                                        q_NC,
                                                                        basis_types_non_stationary_cov_,
                                                                        degree_basis_non_stationary_cov_,
                                                                        number_basis_non_stationary_cov_,
                                                                        knots_non_stationary_cov_eigen_w_,
                                                                        basis_fac);


    //wrapping all the functional elements in a functional_matrix
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> eta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_NC);
    //phi: a sparse functional matrix n_trainx(n_train*Ly), where L is the number of basis for the response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_response_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    using response_basis_tmp_t = extract_template_t< decltype(basis_response_)::element_type >; 
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> phi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(*basis_response_,n_train,number_basis_response_);
    //y_train: a column vector of dimension n_trainx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_train_,number_threads);
    //Xc_train: a functional matrix of dimension n_trainxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_train_,number_threads);
    //Xnc_train: a functional matrix of dimension n_trainxqnc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xnc_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_NON_STATIONARY_>(x_NC_fd_train_,number_threads);



    ////////////////////////////////////////
    /////////        CONSTRUCTING W   //////
    ////////////////////////////////////////
    //distances
    auto coordinates_non_stationary_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_non_stationary_to_pred);
    check_dim_input<_NON_STATIONARY_>(units_to_be_predicted,coordinates_non_stationary_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_NON_STATIONARY_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_non_stationary_to_pred_.cols(),"coordinates matrix columns");
    distance_matrix_pred<_DISTANCE_> distances_non_stationary_to_pred_(std::move(coordinates_non_stationary_),std::move(coordinates_non_stationary_to_pred_));
    distances_non_stationary_to_pred_.compute_distances();
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    //functional weight matrix
    //non-stationary
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_,_KERNEL_,_DISTANCE_> W_NC_pred(rec_weights_y_fd_,
                                                                                                                                                                                      std::move(distances_non_stationary_to_pred_),
                                                                                                                                                                                      kernel_bandwith_non_stationary_cov_,
                                                                                                                                                                                      number_threads,
                                                                                                                                                                                      true);
    W_NC_pred.compute_weights_pred();  
    //Wnc_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Wnc_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_>(W_NC_pred,number_threads);
    //map containing the W
    std::map<std::string,std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>>> W_new = {
        {std::string{covariate_type<_NON_STATIONARY_>()},Wnc_pred}};


    //fgwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(Bc),
                                                                                                 std::move(omega),
                                                                                                 q_C,
                                                                                                 Lc,
                                                                                                 Lc_j,
                                                                                                 std::move(eta),
                                                                                                 q_NC,
                                                                                                 Lnc,
                                                                                                 Lnc_j,
                                                                                                 std::move(phi),
                                                                                                 number_basis_response_,
                                                                                                 std::move(c_tilde_hat),
                                                                                                 std::move(y_train),
                                                                                                 std::move(Xc_train),
                                                                                                 std::move(Xnc_train),
                                                                                                 std::move(R_NC.PenalizationMatrix()),
                                                                                                 a,
                                                                                                 b,
                                                                                                 n_intervals,
                                                                                                 n_train,
                                                                                                 number_threads,
                                                                                                 in_cascade_estimation);

    Rcout << "Betas tuning" << std::endl;                                                                                             

    //retrieve partial residuals
    fwr_predictor->computePartialResiduals();
    //compute the new b for the non-stationary covariates
    fwr_predictor->computeBNew(W_new);
    //compute the beta for stationary covariates
    fwr_predictor->computeStationaryBetas();            
    //compute the beta for non-stationary covariates
    fwr_predictor->computeNonStationaryBetas();   
    //evaluating the betas   
    fwr_predictor->evalBetas(abscissa_points_ev_);

    Rcout << "Betas tuned" << std::endl; 

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fwr_predictor->bCoefficients(),
                                                 names_stationary_cov_,
                                                 basis_types_beta_stationary_cov_,
                                                 number_basis_beta_stationary_cov_,
                                                 knots_beta_stationary_cov_,
                                                 names_non_stationary_cov_,
                                                 basis_types_beta_non_stationary_cov_,
                                                 number_basis_beta_non_stationary_cov_,
                                                 knots_beta_non_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {});
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fwr_predictor->betas(),
                                           abscissa_points_ev_,
                                           names_stationary_cov_,
                                           names_non_stationary_cov_,
                                           {},
                                           {});


    //returning element                                       
    Rcpp::List l;
    //predictor
    l[_model_name_ + "_predictor"] = "predictor_" + std::string{algo_type<_FGWR_ALGO_>()};
    l[_estimation_iter_] = estimation_iter(in_cascade_estimation);
    //stationary covariate basis expansion coefficients for beta_c
    l[_bc_ + "_pred"]  = b_coefficients[_bc_];
    //beta_c
    l[_beta_c_ + "_pred"] = betas[_beta_c_];
    //event-dependent covariate basis expansion coefficients for beta_nc
    l[_bnc_ + "_pred"]  = b_coefficients[_bnc_];
    //beta_nc
    l[_beta_nc_ + "_pred"] = betas[_beta_nc_];

    return l;
}


//
// [[Rcpp::export]]
Rcpp::List tune_new_betas_FGWR(Rcpp::NumericMatrix coordinates_non_stationary_to_pred,   
                               int units_to_be_predicted,
                               Rcpp::NumericVector abscissa_ev,
                               Rcpp::List model_fitted,
                               int n_intervals_quadrature = 100,
                               Rcpp::Nullable<int> num_threads = R_NilValue)
{
    Rcout << "Functional Geographically Weighted Regression bew betas tuning" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT


    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FGWR_;                               //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _NON_STATIONARY_ = FDAGWR_COVARIATES_TYPES::NON_STATIONARY;      //enum for non-stationary covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    if(units_to_be_predicted <= 0){ Rcout << "Number of unit to be predicted has to be a positive number" << std::endl;}
    //checking that the model_fitted contains a fit from FMSGWR_ESC
    wrap_predict_input<_FGWR_ALGO_>(model_fitted);
    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF INTERVALS FOR INTEGRATING VIA MIDPOINT QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_quadrature(n_intervals_quadrature);


    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _bnc_               = std::string{FDAGWR_B_NAMES::bnc};                                //bc
    std::string _beta_nc_           = std::string{FDAGWR_BETAS_NAMES::beta_nc};                        //beta_c
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                            //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                         //response reconstruction weights
    std::string _cov_no_stat_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_NON_STATIONARY_>()};   //stationary training covariates
    std::string _beta_no_stat_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_NON_STATIONARY_>()};   //stationary betas  
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations

    //list with the fitted model
    Rcpp::List fitted_model   = model_fitted[_elem_for_pred_];
    //lists with the input of the training
    Rcpp::List training_input = fitted_model[_input_info_];
    //list with elements of the response
    Rcpp::List response_input       = training_input[_response_];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input = training_input[_response_rec_w_];
    //list with elements of events-dependent covariates
    Rcpp::List non_stationary_cov_input      = training_input[_cov_no_stat_];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_non_stationary_cov_input = training_input[_beta_no_stat_];


    //DOMAIN INFORMATION
    std::size_t n_train = training_input[_n_];
    _FD_INPUT_TYPE_ a   = training_input[_a_];
    _FD_INPUT_TYPE_ b   = training_input[_b_];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);     //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[_abscissa_];             //abscissa point for which the training data are discretized
    //RESPONSE
    std::size_t number_basis_response_ = response_input[_n_basis_];
    std::string basis_type_response_   = response_input[_t_basis_];
    std::size_t degree_basis_response_ = response_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[_coeff_basis_]);  
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[_n_basis_];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[_t_basis_];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[_coeff_basis_]);  
    //NON STATIONARY COV    
    std::size_t q_NC                                          = non_stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_non_stationary_cov_ = non_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_non_stationary_cov_  = non_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_non_stationary_cov_ = non_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_non_stationary_cov_       = non_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_non_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_non_stationary_cov_.data(),knots_non_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_non_stationary_cov_ = wrap_covariates_coefficients<_NON_STATIONARY_>(non_stationary_cov_input[_coeff_basis_]);
    std::vector<double> lambda_non_stationary_cov_ = non_stationary_cov_input[_penalties_];
    auto coordinates_non_stationary_               = reader_data<_DATA_TYPE_,_NAN_REM_>(non_stationary_cov_input[_coords_]);     
    double kernel_bandwith_non_stationary_cov_     = non_stationary_cov_input[_bdw_ker_];  
    //NON-STATIONAY BETAS  
    std::vector<std::size_t> number_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_non_stationary_cov_  = beta_non_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_non_stationary_cov_ = beta_non_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_non_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_non_stationary_cov_.data(),knots_beta_non_stationary_cov_.size()); 

    //covariates names
    //non stationary
    std::vector<std::string> names_non_stationary_cov_ = wrap_covariates_names<_NON_STATIONARY_>(model_fitted[_bnc_]);

    ////////////////////////////////////////
    /////   TRAINING OBJECT CREATION   /////
    ////////////////////////////////////////
    //BASIS SYSTEMS FOR THE BETAS
    //non-stationary (Eta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_NC(knots_beta_non_stationary_cov_eigen_w_, 
                                                    degree_basis_beta_non_stationary_cov_, 
                                                    number_basis_beta_non_stationary_cov_, 
                                                    q_NC);


    //PENALIZATION MATRICES                                               
    //non-stationary
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_NC(std::move(bs_NC),lambda_non_stationary_cov_);
    std::size_t Lnc = R_NC.L();
    std::vector<std::size_t> Lnc_j = R_NC.Lj();



    //MODEL FITTED RESPONSE and COVARIATES
    //response
    //response
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_y_train_ = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using response_basis_tmp_t = extract_template_t< decltype(basis_y_train_)::element_type >;   
    functional_data< _DOMAIN_, response_basis_tmp_t::template_type > y_fd_train_(std::move(coefficients_response_),std::move(basis_y_train_));
    //events covariates
    functional_data_covariates<_DOMAIN_,_NON_STATIONARY_> x_NC_fd_train_(coefficients_non_stationary_cov_,
                                                                        q_NC,
                                                                        basis_types_non_stationary_cov_,
                                                                        degree_basis_non_stationary_cov_,
                                                                        number_basis_non_stationary_cov_,
                                                                        knots_non_stationary_cov_eigen_w_,
                                                                        basis_fac);


    //wrapping all the functional elements in a functional_matrix
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> eta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_NC);
    //y_train: a column vector of dimension n_trainx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_train_,number_threads);
    //Xnc_train: a functional matrix of dimension n_trainxqnc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xnc_train = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_NON_STATIONARY_>(x_NC_fd_train_,number_threads);



    ////////////////////////////////////////
    /////////        CONSTRUCTING W   //////
    ////////////////////////////////////////
    //distances
    auto coordinates_non_stationary_to_pred_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_non_stationary_to_pred);
    check_dim_input<_NON_STATIONARY_>(units_to_be_predicted,coordinates_non_stationary_to_pred_.rows(),"coordinates matrix rows");
    check_dim_input<_NON_STATIONARY_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_non_stationary_to_pred_.cols(),"coordinates matrix columns");
    distance_matrix_pred<_DISTANCE_> distances_non_stationary_to_pred_(std::move(coordinates_non_stationary_),std::move(coordinates_non_stationary_to_pred_));
    distances_non_stationary_to_pred_.compute_distances();
    //response reconstruction weights
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_rec_weights_response_ = basis_fac.create(basis_type_rec_weights_response_,knots_response_eigen_w_,degree_basis_rec_weights_response_,number_basis_rec_weights_response_);
    //extracting the template param of the basis for fd (access it in the template params list with ::template_type)  
    using rec_weights_response_basis_tmp_t = extract_template_t< decltype(basis_rec_weights_response_)::element_type >;   
    functional_data< _DOMAIN_, rec_weights_response_basis_tmp_t::template_type > rec_weights_y_fd_(std::move(coefficients_rec_weights_response_),std::move(basis_rec_weights_response_));
    //functional weight matrix
    //non-stationary
    functional_weight_matrix_non_stationary<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_,_KERNEL_,_DISTANCE_> W_NC_pred(rec_weights_y_fd_,
                                                                                                                                                                                      std::move(distances_non_stationary_to_pred_),
                                                                                                                                                                                      kernel_bandwith_non_stationary_cov_,
                                                                                                                                                                                      number_threads,
                                                                                                                                                                                      true);
    W_NC_pred.compute_weights_pred();  
    //Wnc_pred: n_pred diagonal functional matrices of dimension n_trainxn_train
    std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> > Wnc_pred = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,rec_weights_response_basis_tmp_t::template_type,_NON_STATIONARY_>(W_NC_pred,number_threads);
    //map containing the W
    std::map<std::string,std::vector< functional_matrix_diagonal<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>>> W_new = {
        {std::string{covariate_type<_NON_STATIONARY_>()},Wnc_pred}};


    //fwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(eta),
                                                                                                 q_NC,
                                                                                                 Lnc,
                                                                                                 Lnc_j,
                                                                                                 std::move(y_train),
                                                                                                 std::move(Xnc_train),
                                                                                                 std::move(R_NC.PenalizationMatrix()),
                                                                                                 a,
                                                                                                 b,
                                                                                                 n_intervals,
                                                                                                 n_train,
                                                                                                 number_threads);

    Rcout << "Betas tuning" << std::endl;
                                                                                                 
    //compute the new b for the non-stationary covariates
    fwr_predictor->computeBNew(W_new);          
    //compute the beta for non-stationary covariates
    fwr_predictor->computeNonStationaryBetas();   
    //evaluating the betas   
    fwr_predictor->evalBetas(abscissa_points_ev_);

    Rcout << "Betas tuned" << std::endl;

    //retrieving the results, wrapping them in order to be returned into R
    //b                                                                        
    Rcpp::List b_coefficients = wrap_b_to_R_list(fwr_predictor->bCoefficients(),
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 names_non_stationary_cov_,
                                                 basis_types_beta_non_stationary_cov_,
                                                 number_basis_beta_non_stationary_cov_,
                                                 knots_beta_non_stationary_cov_,
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {},
                                                 {});
    //betas
    Rcpp::List betas = wrap_beta_to_R_list(fwr_predictor->betas(),
                                           abscissa_points_ev_,
                                           {},
                                           names_non_stationary_cov_,
                                           {},
                                           {});


    //returning element                                       
    Rcpp::List l;
    //predictor
    l[_model_name_ + "_predictor"] = "predictor_" + std::string{algo_type<_FGWR_ALGO_>()};
    //event-dependent covariate basis expansion coefficients for beta_e
    l[_bnc_ + "_pred"]  = b_coefficients[_bnc_];
    //beta_e
    l[_beta_nc_ + "_pred"] = betas[_beta_nc_];

    return l;
}



//
// [[Rcpp::export]]
Rcpp::List new_y_FMSGWR_ESC(Rcpp::List coeff_stationary_cov_to_pred,
                            Rcpp::List coeff_events_cov_to_pred,  
                            Rcpp::List coeff_stations_cov_to_pred,
                            int units_to_be_predicted,
                            Rcpp::NumericVector abscissa_ev,
                            Rcpp::List new_beta,
                            Rcpp::List model_fitted,
                            int n_knots_smoothing_pred = 100,
                            Rcpp::Nullable<int> num_threads = R_NilValue)
{
    Rcout << "Functional Multi-Source Geographically Weighted Regression ESC new y2" << std::endl;

    //EVERY COLUMN A UNIT, EVERY ROW A RAW EVALUATION/BASIS COEFFICIENT
    //ONLY FOR COORDINATES, EVERY ROW IS A UNIT


    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::_FMSGWR_ESC_;                         //fgwr type (estimating stationary -> station-dependent -> event-dependent)
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                  //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;            //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;              //enum for stationary covariates
    constexpr auto _EVENT_ = FDAGWR_COVARIATES_TYPES::EVENT;                        //enum for event covariates
    constexpr auto _STATION_ = FDAGWR_COVARIATES_TYPES::STATION;                    //enum for station covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;           //enum for the penalization order
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                        //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                         //how to remove nan (with mean of non-nans)
    
    if(units_to_be_predicted <= 0){ Rcout << "Number of unit to be predicted has to be a positive number" << std::endl;}
    //checking that the model_fitted contains a fit from FMSGWR_ESC
    wrap_predict_input<_FGWR_ALGO_>(model_fitted);
Rcout << "Creation basis factory" << std::endl;    
    //instance of the factory for the basis
    basis_factory::basisFactory& basis_fac(basis_factory::basisFactory::Instance());    

    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////
Rcout << "Wrap first parameters" << std::endl;
    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
    // NUMBER OF KNOTS TO PERFORM SMOOTHING ON THE RESPONSE WITHOUT THE NON-STATIONARY COMPONENTS
Rcout << "Wrap n_knots_smoothing_pred" << std::endl;
    int n_knots_smoothing_y_new = wrap_and_check_n_knots_smoothing(n_knots_smoothing_pred);


Rcout << "Strings" << std::endl;
    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    // NAME OF THE LIST ELEMENT COMING FROM THE FITTING MODEL FUNCTION
    //names main outputs
    std::string _model_name_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name};          //FWR model used
    std::string _estimation_iter_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter};     //Exact or Cascade estimation
    std::string _bc_                = std::string{FDAGWR_B_NAMES::bc};                                 //bc
    std::string _beta_c_            = std::string{FDAGWR_BETAS_NAMES::beta_c};                         //beta_c
    std::string _be_                = std::string{FDAGWR_B_NAMES::be};                                 //be
    std::string _beta_e_            = std::string{FDAGWR_BETAS_NAMES::beta_e};                         //beta_e
    std::string _bs_                = std::string{FDAGWR_B_NAMES::bs};                                 //bs
    std::string _beta_s_            = std::string{FDAGWR_BETAS_NAMES::beta_s};                         //beta_s
    std::string _elem_for_pred_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred};       //elements used to predict (reconstructing training data and partial residuals)
    std::string _partial_residuals_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::p_res};               //partial residuals 
    std::string _input_info_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::inputs_info};         //training data information needed for prediction
    //names secondary outputs, contained in the main ones
    //the different covariates
    std::string _response_       = std::string{covariate_type<_RESPONSE_>()};                                                        //response
    std::string _response_rec_w_ = std::string{covariate_type<_REC_WEIGHTS_>()};                                                     //response reconstruction weights
    std::string _cov_stat_       = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATIONARY_>()};   //stationary training covariates
    std::string _beta_stat_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATIONARY_>()};   //stationary betas
    std::string _cov_event_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_EVENT_>()};        //event-dependent training covariates
    std::string _beta_event_     = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_EVENT_>()};        //event-dependent betas
    std::string _cov_station_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cov}  + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates
    std::string _beta_station_   = std::string{FDAGWR_HELPERS_for_PRED_NAMES::beta} + std::string{covariate_type<_STATION_>()};      //station-dependent training covariates    
    //training data features
    std::string _q_              = std::string{FDAGWR_HELPERS_for_PRED_NAMES::q};                   //number of covariate
    std::string _n_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n_basis};             //number of basis
    std::string _t_basis_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_t};             //type of basis
    std::string _deg_basis_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_deg};           //degree of basis
    std::string _knots_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::basis_knots};         //knots of basis
    std::string _coeff_basis_    = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis};         //coefficients of basis expansion
    std::string _penalties_      = std::string{FDAGWR_HELPERS_for_PRED_NAMES::penalties};           //lambdas for penalization
    std::string _coords_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::coords};              //location UTM coordinates
    std::string _bdw_ker_        = std::string{FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker};             //kernel bandwith 
    //domain
    std::string _n_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::n};                   //number of training units
    std::string _a_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::a};                   //left domain extreme
    std::string _b_                = std::string{FDAGWR_HELPERS_for_PRED_NAMES::b};                   //right domain extreme
    std::string _abscissa_         = std::string{FDAGWR_HELPERS_for_PRED_NAMES::abscissa};            //abscissa of training data raw evaluations
    std::string _cascade_estimate_ = std::string{FDAGWR_HELPERS_for_PRED_NAMES::cascade_estimate};         //if using in cascade-estimation


    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[_elem_for_pred_];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[_input_info_];
    //list with elements of the response
    Rcpp::List response_input            = training_input[_response_];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[_cov_stat_];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[_beta_stat_];
    //list with elements of events-dependent covariates
    Rcpp::List events_cov_input          = training_input[_cov_event_];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_events_cov_input     = training_input[_beta_event_];
    //list with elements of stations-dependent covariates
    Rcpp::List stations_cov_input        = training_input[_cov_station_];
    //list with elements of the beta of stations-dependent covariates
    Rcpp::List beta_stations_cov_input   = training_input[_beta_station_];
Rcout << "Reading from the model_fitted" << std::endl;
    //ESTIMATION TECHNIQUE
    bool in_cascade_estimation = training_input[_cascade_estimate_];
    //DOMAIN INFORMATION
    std::size_t n_train = training_input[_n_];
    _FD_INPUT_TYPE_ a   = training_input[_a_];
    _FD_INPUT_TYPE_ b   = training_input[_b_];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);     //abscissa points for which the evaluation of the prediction is required
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[_n_basis_];
    std::string basis_type_response_   = response_input[_t_basis_];
    std::size_t degree_basis_response_ = response_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[_coeff_basis_]); 
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_); 
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[_q_];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_       = stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    //EVENTS COV    
    std::size_t q_E                                   = events_cov_input[_q_];
    std::vector<std::size_t> number_basis_events_cov_ = events_cov_input[_n_basis_];
    std::vector<std::string> basis_types_events_cov_  = events_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_events_cov_ = events_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_events_cov_       = events_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_events_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    //STATIONS COV  
    std::size_t q_S                                     = stations_cov_input[_q_];
    std::vector<std::size_t> number_basis_stations_cov_ = stations_cov_input[_n_basis_];
    std::vector<std::string> basis_types_stations_cov_  = stations_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_stations_cov_ = stations_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stations_cov_       = stations_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_stations_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());    
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[_bc_];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[_coeff_basis_]);  //Lc_j x 1
        Bc.push_back(Bc_i);}
    //EVENTS BETAS  
    std::vector<std::size_t> number_basis_beta_events_cov_ = beta_events_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_events_cov_  = beta_events_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = beta_events_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_events_cov_ = beta_events_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_events_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size()); 
    //STATIONS BETAS    
    std::vector<std::size_t> number_basis_beta_stations_cov_ = beta_stations_cov_input[_n_basis_];
    std::vector<std::string> basis_types_beta_stations_cov_  = beta_stations_cov_input[_t_basis_];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = beta_stations_cov_input[_deg_basis_];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stations_cov_ = beta_stations_cov_input[_knots_basis_];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stations_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());
    
    
Rcout << "Reading from new_beta" << std::endl;    
    //TUNED NON-STATIONARY BETAS
    //saving the betas basis expansion coefficients for tuned event-dependent covariates
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix>> Be_tuned; //vettore esterno: per ogni covariata S. Interno: per ogni unità di training
    Be_tuned.reserve(q_E);
    Rcpp::List Be_tuned_list = new_beta[_be_ + "_pred"];
    for(std::size_t i = 0; i < q_E; ++i){
        Rcpp::List Be_tuned_i_list = Be_tuned_list[i];
        auto Be_tuned_i = wrap_covariates_coefficients<_EVENT_>(Be_tuned_i_list[_coeff_basis_]);  //Le_j x 1
        Be_tuned.push_back(Be_tuned_i);}
    //saving the betas basis expansion coefficients for tuned station-dependent covariates
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix>> Bs_tuned; //vettore esterno: per ogni covariata S. Interno: per ogni unità di training
    Bs_tuned.reserve(q_S);
    Rcpp::List Bs_tuned_list = new_beta[_bs_ + "_pred"];
    for(std::size_t i = 0; i < q_S; ++i){
        Rcpp::List Bs_tuned_i_list = Bs_tuned_list[i];
        auto Bs_tuned_i = wrap_covariates_coefficients<_STATION_>(Bs_tuned_i_list[_coeff_basis_]);  //Ls_j x 1
        Bs_tuned.push_back(Bs_tuned_i);}



Rcout << "Be_fitted" << std::endl;
for(std::size_t i = 0; i < Be_tuned.size(); ++i){
    Rcout << "CovE " << i << std::endl;
    for(std::size_t j = 0; j < Be_tuned[i].size(); ++j){
        Rcout << Be_tuned[i][j] << std::endl;
    }
}


Rcout << "Bs_fitted" << std::endl;
for(std::size_t i = 0; i < Bs_tuned.size(); ++i){
    Rcout << "CovS " << i << std::endl;
    for(std::size_t j = 0; j < Bs_tuned[i].size(); ++j){
        Rcout << Bs_tuned[i][j] << std::endl;
    }
}


Rcout << "Creating the basis systems for the betas" << std::endl;
    //BASIS SYSTEMS OF THE BETAS
    //stationary (Omega)
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                   degree_basis_beta_stationary_cov_, 
                                                   number_basis_beta_stationary_cov_, 
                                                   q_C);
    //events (Theta)
    basis_systems< _DOMAIN_, bsplines_basis > bs_E(knots_beta_events_cov_eigen_w_, 
                                                   degree_basis_beta_events_cov_, 
                                                   number_basis_beta_events_cov_, 
                                                   q_E);
    //stations (Psi)
    basis_systems< _DOMAIN_, bsplines_basis > bs_S(knots_beta_stations_cov_eigen_w_,  
                                                   degree_basis_beta_stations_cov_, 
                                                   number_basis_beta_stations_cov_, 
                                                   q_S);

    //basis numbers beta
    //stationary
    std::size_t Lc = std::reduce(number_basis_beta_stationary_cov_.cbegin(),number_basis_beta_stationary_cov_.cend(),static_cast<std::size_t>(0));
    std::vector<std::size_t> Lc_j = number_basis_beta_stationary_cov_;
    //events
    std::size_t Le = std::reduce(number_basis_beta_events_cov_.cbegin(),number_basis_beta_events_cov_.cend(),static_cast<std::size_t>(0));
    std::vector<std::size_t> Le_j = number_basis_beta_events_cov_;
    //stations
    std::size_t Ls = std::reduce(number_basis_beta_stations_cov_.cbegin(),number_basis_beta_stations_cov_.cend(),static_cast<std::size_t>(0));
    std::vector<std::size_t> Ls_j = number_basis_beta_stations_cov_;

    //wrapping all the functional elements in a functional_matrix
    //omega: a sparse functional matrix of dimension qcxLc
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> omega = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_C);
    //theta: a sparse functional matrix of dimension qexLe
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> theta = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_E);
    //psi: a sparse functional matrix of dimension qsxLs
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> psi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,bsplines_basis>(bs_S);


Rcout << "Covariates to be pred" << std::endl;
    //////////////////////////////////////////////
    ///// WRAPPING COVARIATES TO BE PREDICTED ////
    //////////////////////////////////////////////
    // stationary
    //covariates names
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_to_be_pred_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov_to_pred); 
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(units_to_be_predicted,coefficients_stationary_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //events
    //covariates names
    std::vector<std::string> names_events_cov_ = wrap_covariates_names<_EVENT_>(coeff_events_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_to_be_pred_ = wrap_covariates_coefficients<_EVENT_>(coeff_events_cov_to_pred); 
    for(std::size_t i = 0; i < q_E; ++i){   
        check_dim_input<_EVENT_>(number_basis_events_cov_[i],coefficients_events_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_EVENT_>(units_to_be_predicted,coefficients_events_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //stations
    //covariates names
    std::vector<std::string> names_stations_cov_ = wrap_covariates_names<_STATION_>(coeff_stations_cov_to_pred);
    //covariates basis expansion coefficients
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_to_be_pred_ = wrap_covariates_coefficients<_STATION_>(coeff_stations_cov_to_pred);
    for(std::size_t i = 0; i < q_S; ++i){   
        check_dim_input<_STATION_>(number_basis_stations_cov_[i],coefficients_stations_cov_to_be_pred_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATION_>(units_to_be_predicted,coefficients_stations_cov_to_be_pred_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    
    //TO BE PREDICTED COVARIATES  
    //stationary covariates
    functional_data_covariates<_DOMAIN_,_STATIONARY_> x_C_fd_to_be_pred_(coefficients_stationary_cov_to_be_pred_,
                                                                         q_C,
                                                                         basis_types_stationary_cov_,
                                                                         degree_basis_stationary_cov_,
                                                                         number_basis_stationary_cov_,
                                                                         knots_stationary_cov_eigen_w_,
                                                                         basis_fac);
    //events covariates
    functional_data_covariates<_DOMAIN_,_EVENT_>   x_E_fd_to_be_pred_(coefficients_events_cov_to_be_pred_,
                                                                      q_E,
                                                                      basis_types_events_cov_,
                                                                      degree_basis_events_cov_,
                                                                      number_basis_events_cov_,
                                                                      knots_events_cov_eigen_w_,
                                                                      basis_fac);
    //stations covariates
    functional_data_covariates<_DOMAIN_,_STATION_> x_S_fd_to_be_pred_(coefficients_stations_cov_to_be_pred_,
                                                                      q_S,
                                                                      basis_types_stations_cov_,
                                                                      degree_basis_stations_cov_,
                                                                      number_basis_stations_cov_,
                                                                      knots_stations_cov_eigen_w_,
                                                                      basis_fac);
    //Xc_new: a functional matrix of dimension n_newxqc
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xc_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATIONARY_>(x_C_fd_to_be_pred_,number_threads);                                                               
    //Xe_new: a functional matrix of dimension n_newxqe
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xe_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_EVENT_>(x_E_fd_to_be_pred_,number_threads);
    //Xs_new: a functional matrix of dimension n_newxqs
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> Xs_new = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,_STATION_>(x_S_fd_to_be_pred_,number_threads);
    //map containing the X
    std::map<std::string,functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>> X_new = {
        {std::string{covariate_type<_STATIONARY_>()},Xc_new},
        {std::string{covariate_type<_EVENT_>()},     Xe_new},
        {std::string{covariate_type<_STATION_>()},   Xs_new}};


std::cout << "Quasi nel New constructor" << std::endl;    
    //MI SERVE
    //fwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(Bc),
                                                                                                 std::move(Be_tuned),
                                                                                                 std::move(Bs_tuned),
                                                                                                 std::move(omega),
                                                                                                 q_C,
                                                                                                 Lc,
                                                                                                 Lc_j,
                                                                                                 std::move(theta),
                                                                                                 q_E,
                                                                                                 Le,
                                                                                                 Le_j,
                                                                                                 std::move(psi),
                                                                                                 q_S,
                                                                                                 Ls,
                                                                                                 Ls_j,
                                                                                                 units_to_be_predicted,
                                                                                                 a,
                                                                                                 b,
                                                                                                 1,
                                                                                                 n_train,
                                                                                                 number_threads,
                                                                                                 in_cascade_estimation);


Rcout << "betaC" << std::endl;
    //compute the beta for stationary covariates
    fwr_predictor->computeStationaryBetas();  

Rcout << "betaNC" << std::endl;          
    //compute the beta for non-stationary covariates
    fwr_predictor->computeNonStationaryBetas(); 
 
Rcout << "Pred" << std::endl;  
    //perform prediction
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y_pred = fwr_predictor->predict(X_new);
Rcout << "Pred ev" << std::endl;
    //evaluating the prediction
    std::vector< std::vector< _FD_OUTPUT_TYPE_>> y_pred_ev = fwr_predictor->evalPred(y_pred,abscissa_points_ev_);
Rcout << "Pred smoothing" << std::endl;
    //smoothing of the prediction
    auto y_pred_smooth_coeff = fwr_predictor->smoothPred<_DOMAIN_>(y_pred,*basis_pred,knots_smoothing_pred);
Rcout << "Wrap out" << std::endl;
    //predictions evaluations
    Rcpp::List y_pred_ev_R = wrap_prediction_to_R_list<_FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_>(y_pred_ev,
                                                                                          abscissa_points_ev_,
                                                                                          y_pred_smooth_coeff,
                                                                                          basis_type_response_,
                                                                                          number_basis_response_,
                                                                                          degree_basis_response_,
                                                                                          knots_smoothing_pred);

    //returning element                                       
    Rcpp::List l;
    //predictor
    l[_model_name_ + "_predictor"] = "predictor_" + std::string{algo_type<_FGWR_ALGO_>()};
    l[_estimation_iter_]           = estimation_iter(in_cascade_estimation);
    //predictions
    l[std::string{FDAGWR_HELPERS_for_PRED_NAMES::pred}] = y_pred_ev_R;

    return l;

   /* 
    Rcpp::List l;
    //predictor
    l[_model_name_ + "_predictor"] = "predictor_" + std::string{algo_type<_FGWR_ALGO_>()};
    l[_estimation_iter_]           = estimation_iter(in_cascade_estimation);
    return l;
    */
}

