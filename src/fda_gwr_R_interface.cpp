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
* @brief Contains the R-interfaced functions of the package 'fdagwr', which implement Functional Geographical Weighted Regression
*        coefficients estimation, for different (multi-source (FMSGWR), mixed (FMGWR), geographically weighted (FGWR) and 
*        weighted (FWR)) functional regression models.
* @author Andrea Enrico Franzoni
*/




using namespace Rcpp;

//
// [[Rcpp::depends(RcppEigen)]]


/*!
* @brief Function to check fdagwr package installation
*/
//
// [[Rcpp::export]]
void installation_fdagwr(){   Rcout << "fdagwr4 has been installed"<< std::endl;}



//
// [[Rcpp::export]]
void fdagwr_testing()
{
  //creation of a collection of systems of bsplines: in this case, there are two basis systems.
  // Both of them are constructed over 13 knots equally spaced. In the first system, there are 15 bsplines of grade 3,
  // while in the second system there are 14 bsplines of grade 2
  
  //basis domain geometry (from fdaPDE)
  using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry; 
  //domain
  double a = -2.5;
  double b = 1;
  //number of basis systems
  std::size_t q = 2;
  //knots
  FDAGWR_TRAITS::Dense_Vector knots_basis = FDAGWR_TRAITS::Dense_Vector::LinSpaced(13, a, b);
  //basis degree
  std::vector<std::size_t> degree_basis{3,2};
  //basis number
  std::vector<std::size_t> number_basis{15,14};

  //creation of the collection of the basis system: the problem is in this constructor
  basis_systems< _DOMAIN_, bsplines_basis > bs(knots_basis, 
                                               degree_basis, 
                                               number_basis, 
                                               q);
  //evaluating the two basis systems in 0.3
  double loc = 0.3;   //abscissa of evaluation
  auto evaluation_basis_system_1 = bs.eval_base(loc,0);
  Rcout << "Evaluation of the first basis system in " << loc << std::endl;
  Rcout << evaluation_basis_system_1 << std::endl;
  auto evaluation_basis_system_2 = bs.eval_base(loc,1);
  Rcout << "Evaluation of the second basis system in " << loc << std::endl;
  Rcout << evaluation_basis_system_2 << std::endl;
}


/*!
* @brief Fitting a Functional Multi-Source Geographically Weighted Regression ESC model. The covariates are functional objects, divided into
*        three categories: stationary covariates (C), constant over space, event-dependent covariates (E), that vary depending on the spatial coordinates of the event, 
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
* @param coeff_rec_weights_y_points matrix of double containing the coefficients of the weights to reconstruct the (functional) response: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param degree_basis_rec_weights_y_points non-negative integer: the degree of the basis used for response reconstruction weights. Default explained below (can be NULL).
* @param n_basis_rec_weights_y_points positive integer: number of basis for the basis expansion of response reconstruction weights. It must match number of rows of coeff_rec_weights_y_points. Default explained below (can be NULL).
* @param coeff_stationary_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                             The name of the i-th element is the name of the i-th stationary covariate (default: "reg.Ci").
* @param basis_types_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariates basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate. Default explained below (can be NULL).
* @param n_basis_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate. It must match number of rows of the i-th element of coeff_stationary_cov. Default explained below (can be NULL).
* @param penalization_stationary_cov vector of non-negative double: element i-th is the penalization used for the i-th stationary covariate.
* @param knots_beta_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates (functional regression) coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param n_basis_beta_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param coeff_events_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th events-dependent covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                             The name of the i-th element is the name of the i-th events-dependent covariate (default: "reg.Ei").
* @param basis_types_events_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariates basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_events_cov vector of double with the abscissa points with respect which the basis expansions of the events-dependent covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_events_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th events-dependent covariate. Default explained below (can be NULL).
* @param n_basis_events_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th events-dependent covariate. It must match number of rows of the i-th element of coeff_events_cov. Default explained below (can be NULL).
* @param penalization_events_cov vector of non-negative double: element i-th is the penalization used for the i-th events-dependent covariate.
* @param coordinates_events matrix of double containing the UTM coordinates of the event of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
* @param kernel_bandwith_events positive double indicating the bandwith of the gaussian kernel used to smooth the distances within events.
* @param knots_beta_events_cov vector of double with the abscissa points with respect which the basis expansions of the events-dependent covariates (functional regression) coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_events_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th events-dependent covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param n_basis_beta_events_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th events-dependent covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param coeff_stations_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stations-dependent covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                             The name of the i-th element is the name of the i-th stations-dependent covariate (default: "reg.Si").
* @param basis_types_stations_cov vector of strings, element i-th containing the type of basis used for the i-th stations-dependent covariates basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_stations_cov vector of double with the abscissa points with respect which the basis expansions of the stations-dependent covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_stations_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stations-dependent covariate. Default explained below (can be NULL).
* @param n_basis_stations_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stations-dependent covariate. It must match number of rows of the i-th element of coeff_stations_cov. Default explained below (can be NULL).
* @param penalization_stations_cov vector of non-negative double: element i-th is the penalization used for the i-th stations-dependent covariate.
* @param coordinates_stations matrix of double containing the UTM coordinates of the station of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
* @param kernel_bandwith_stations positive double indicating the bandwith of the gaussian kernel used to smooth the distances within stations.
* @param knots_beta_stations_cov vector of double with the abscissa points with respect which the basis expansions of the stations-dependent covariates (functional regression) coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_stations_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stations-dependent covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param n_basis_beta_stations_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stations-dependent covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param n_knots_smoothing number of knots used to perform the smoothing on the response obtained leaving out all the non-stationary components
* @param n_intervals_trapezoidal_quadrature number of intervals used while performing integration via adaptive trapezoidal quadrature rule
* @param target_error_trapezoidal_quadrature target error while integrating via adaptive trapezoidal quadrature rule
* @param max_iterations_trapezoidal_quadrature max number of iterations for integrating via adaptive trapezoidal quadrature rule
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine running fmsgwr.
* @param basis_type_y_points string containing the type of basis used for the (functional) response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_type_rec_weights_y_points string containing the type of basis used for the weights to reconstruct the (functional) response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariates (functional regression) coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_events_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariates (functional regression) coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stations_cov vector of strings, element i-th containing the type of basis used for the i-th stations-dependent covariates (functional regression) coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @return an R list containing:
*         - "FGWR": string containing the type of fgwr used ("FGWR_FMS_ESC")
*         - "Bc": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Basis_coeff": a Lc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "Basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective element of basis_types_beta_stationary_cov)
*                 - "Basis_number": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stationary_cov)
*                 - "Basis_knots": the knots used to create the basis system for the beta (it is the input knots_beta_stationary_cov)
*         - "Beta_c": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Beta_eval": a vector of double containing the discrete evaluations of the stationary beta
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "Be": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "Basis_coeff": a list, containg, for each unit, a Le_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "Basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_events_cov)
*                 - "Basis_number": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_events_cov)
*                 - "Basis_knots": the knots used to create the basis system for the beta (it is the input knots_beta_events_cov)
*         - "Beta_e": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)  
*         - "Bs": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "Basis_coeff": a list, containg, for each unit, a Ls_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "Basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_stations_cov)
*                 - "Basis_number": the number of basis used for performing the beta basis expansion (eespective elements of n_basis_beta_stations_cov)
*                 - "Basis_knots": the knots used to create the basis system for the beta (it is the input knots_beta_stations_cov)
*         - "Beta_s": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "predictor_info": a list containing partial residuals and information of the fitted model to perform predictions for new statistical units:
*                             - "partial_res": a list containing information to compute the partial residuals:
*                                              - "c_tilde_hat": vector of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon
*                                              - "A__": vector of matrices with the operator A_e for each statistical unit
*                                              - "B__for_K": vector of matrices with the operator B_e used for the K_e_s(t) computation, for each statistical unit
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "Response reconstruction weights": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "cov_Event": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                                            - "penalizations":
*                                                            - "coordinates":
*                                                            - "kernel_bwd_Event":
*                                              - "beta_Event": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "cov_Station": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                                            - "penalizatins":
*                                                            - "coordinates":
*                                                            - "kernel_bwd_Station":
*                                              - "beta_Station": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "a" 
*                                              - "b"
*                                              - "abscissa"
* @details constant basis are used, for a covariate, if it resembles a scalar shape. It consists of a straight line with y-value equal to 1 all over the data domain.
*          Can be seen as a B-spline basis with degree 0, number of basis 1, using one knot, consequently having only one coefficient for the only basis for each statistical unit.
*          fmsgwr sets all the feats accordingly if reads constant basis.
*          However, recall that the response is a functional datum, as the regressors coefficients. Since the package's basis variety could be hopefully enlarged in the future 
*          (for example, introducing Fourier basis for handling data that present periodical behaviors), the input parameters regarding basis types for response, response reconstruction
*          weights and regressors coefficients are left at the end of the input list, and defaulted as NULL. Consequently they will use a B-spline basis system, and should NOT use a constant basis,
*          Recall to perform externally the basis expansion before using the package, and afterwards passing basis types, degree and number and basis expansion coefficients and knots coherently
* @note a little excursion about degree and number of basis passed as input. For each specific covariate, or the response, if using B-spline basis, remember that number of knots = number of basis - degree + 1. 
*       By default, if passing NULL, fmsgwr uses a cubic B-spline system of basis, the number of basis is computed coherently from the number of knots (that is the only mandatory input parameter).
*       Passing only the degree of the bsplines, the number of basis used will be set accordingly, and viceversa if passing only the number of basis. 
*       But, take care that the number of basis used has to match the number of rows of coefficients matrix (for EACH type of basis). If not, an exception is thrown. No problems arise if letting fmsgwr defaulting the number of basis.
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
                      bool bf_estimation = false,
                      int n_knots_smoothing = 100,
                      int n_intervals_trapezoidal_quadrature = 100,
                      double target_error_trapezoidal_quadrature = 1e-3,
                      int max_iterations_trapezoidal_quadrature = 100,
                      Rcpp::Nullable<int> num_threads = R_NilValue,
                      std::string basis_type_y_points = "bsplines",
                      std::string basis_type_rec_weights_y_points = "bsplines",
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_events_cov = R_NilValue,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stations_cov = R_NilValue)
{
    //funzione per il multi-source gwr
    //  !!!!!!!! NB: l'ordine delle basi su c++ corrisponde al degree su R !!!!!
    Rcout << "Functional Multi-Source Geographically Weighted Regression ESC" << std::endl;


    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'

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
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_trapezoidal_quadrature(n_intervals_trapezoidal_quadrature);
    // TARGET ERROR WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    double target_error = wrap_and_check_target_error_trapezoidal_quadrature(target_error_trapezoidal_quadrature);
    // MAXIMUM NUMBER OF ITERATIONS WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int max_iterations = wrap_and_check_max_iterations_trapezoidal_quadrature(max_iterations_trapezoidal_quadrature);


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
    std::size_t number_basis_response_ = number_and_degree_basis_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t degree_basis_response_ = number_and_degree_basis_response_[FDAGWR_FEATS::degree_basis_string];
    check_dim_input<_RESPONSE_>(number_basis_response_,coefficients_response_.rows(),"response coefficients matrix rows");
    check_dim_input<_RESPONSE_>(number_of_statistical_units_,coefficients_response_.cols(),"response coefficients matrix columns");     
    //response reconstruction weights
    auto number_and_degree_basis_rec_weights_response_ = wrap_and_check_basis_number_and_degree<_REC_WEIGHTS_>(n_basis_rec_weights_y_points,degree_basis_rec_weights_y_points,knots_response_.size(),basis_type_rec_weights_response_);
    std::size_t number_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t degree_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[FDAGWR_FEATS::degree_basis_string];
    check_dim_input<_REC_WEIGHTS_>(number_basis_rec_weights_response_,coefficients_rec_weights_response_.rows(),"response reconstruction weights coefficients matrix rows");
    check_dim_input<_REC_WEIGHTS_>(number_of_statistical_units_,coefficients_rec_weights_response_.cols(),"response reconstruction weights coefficients matrix columns");     
    //stationary cov
    auto number_and_degree_basis_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_stationary_cov,degrees_basis_stationary_cov,knots_stationary_cov_.size(),q_C,basis_types_stationary_cov_);
    std::vector<std::size_t> number_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[FDAGWR_FEATS::degree_basis_string];
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stationary cov
    auto number_and_degree_basis_beta_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_beta_stationary_cov,degrees_basis_beta_stationary_cov,knots_beta_stationary_cov_.size(),q_C,basis_types_beta_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[FDAGWR_FEATS::degree_basis_string];
    //events cov    
    auto number_and_degree_basis_events_cov_ = wrap_and_check_basis_number_and_degree<_EVENT_>(n_basis_events_cov,degrees_basis_events_cov,knots_events_cov_.size(),q_E,basis_types_events_cov_);
    std::vector<std::size_t> number_basis_events_cov_ = number_and_degree_basis_events_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_events_cov_ = number_and_degree_basis_events_cov_[FDAGWR_FEATS::degree_basis_string];
    for(std::size_t i = 0; i < q_E; ++i){   
        check_dim_input<_EVENT_>(number_basis_events_cov_[i],coefficients_events_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_EVENT_>(number_of_statistical_units_,coefficients_events_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta events cov
    auto number_and_degree_basis_beta_events_cov_ = wrap_and_check_basis_number_and_degree<_EVENT_>(n_basis_beta_events_cov,degrees_basis_beta_events_cov,knots_beta_events_cov_.size(),q_E,basis_types_beta_events_cov_);
    std::vector<std::size_t> number_basis_beta_events_cov_ = number_and_degree_basis_beta_events_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = number_and_degree_basis_beta_events_cov_[FDAGWR_FEATS::degree_basis_string];
    //stations cov
    auto number_and_degree_basis_stations_cov_ = wrap_and_check_basis_number_and_degree<_STATION_>(n_basis_stations_cov,degrees_basis_stations_cov,knots_stations_cov_.size(),q_S,basis_types_stations_cov_);
    std::vector<std::size_t> number_basis_stations_cov_ = number_and_degree_basis_stations_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_stations_cov_ = number_and_degree_basis_stations_cov_[FDAGWR_FEATS::degree_basis_string];
    for(std::size_t i = 0; i < q_S; ++i){   
        check_dim_input<_STATION_>(number_basis_stations_cov_[i],coefficients_stations_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATION_>(number_of_statistical_units_,coefficients_stations_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stations cov 
    auto number_and_degree_basis_beta_stations_cov_ = wrap_and_check_basis_number_and_degree<_STATION_>(n_basis_beta_stations_cov,degrees_basis_beta_stations_cov,knots_beta_stations_cov_.size(),q_S,basis_types_beta_stations_cov_);
    std::vector<std::size_t> number_basis_beta_stations_cov_ = number_and_degree_basis_beta_stations_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = number_and_degree_basis_beta_stations_cov_[FDAGWR_FEATS::degree_basis_string];


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


    //fgwr algorithm
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
                                                                                    target_error,
                                                                                    max_iterations,
                                                                                    abscissa_points_,
                                                                                    number_of_statistical_units_,
                                                                                    number_threads,
                                                                                    bf_estimation);

    //computing the b
    fgwr_algo->compute();
    //evaluating the betas   
    fgwr_algo->evalBetas();

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
    //regression model used and estimation technique
    l[FDAGWR_HELPERS_for_PRED_NAMES::model_name] = algo_type<_FGWR_ALGO_>();
    l[FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter] = estimation_iter(bf_estimation);
    //stationary covariate basis expansion coefficients for beta_c
    l[FDAGWR_B_NAMES::bc]  = b_coefficients[FDAGWR_B_NAMES::bc];
    //beta_c
    l[FDAGWR_BETAS_NAMES::beta_c] = betas[FDAGWR_BETAS_NAMES::beta_c];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[FDAGWR_B_NAMES::be]  = b_coefficients[FDAGWR_B_NAMES::be];
    //beta_e
    l[FDAGWR_BETAS_NAMES::beta_e] = betas[FDAGWR_BETAS_NAMES::beta_e];
    //station-dependent covariate basis expansion coefficients for beta_s
    l[FDAGWR_B_NAMES::bs]  = b_coefficients[FDAGWR_B_NAMES::bs];
    //beta_s
    l[FDAGWR_BETAS_NAMES::beta_s] = betas[FDAGWR_BETAS_NAMES::beta_s];

    //returning all the elements needed to perform prediction
    Rcpp::List elem_for_pred;
    Rcpp::List inputs_info;
    //p res
    elem_for_pred[FDAGWR_HELPERS_for_PRED_NAMES::p_res] = p_res;
    //input of y
    Rcpp::List response_input;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_type_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = Rcpp::wrap(coefficients_response_out_);
    inputs_info[covariate_type<FDAGWR_COVARIATES_TYPES::RESPONSE>()] = response_input;
    //input of w for y  
    Rcpp::List response_rec_w_input;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_type_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = Rcpp::wrap(coefficients_rec_weights_response_out_);
    inputs_info[covariate_type<FDAGWR_COVARIATES_TYPES::REC_WEIGHTS>()] = response_rec_w_input;
    //input of C
    Rcpp::List C_input;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::q] = q_C;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = toRList(coefficients_stationary_cov_,false);
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<FDAGWR_COVARIATES_TYPES::STATIONARY>()] = C_input;
    //input of Beta C   
    Rcpp::List beta_C_input;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_beta_stationary_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<FDAGWR_COVARIATES_TYPES::STATIONARY>()] = beta_C_input;
    //input of E
    Rcpp::List E_input;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::q] = q_E;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_events_cov_;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_events_cov_;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_events_cov_;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_events_cov_;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = toRList(coefficients_events_cov_,false);
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties] = lambda_events_cov_;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::coords] = Rcpp::wrap(coordinates_events_out_);
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker] = kernel_bandwith_events;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<FDAGWR_COVARIATES_TYPES::EVENT>()] = E_input;
    //input of Beta E   
    Rcpp::List beta_E_input;
    beta_E_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_beta_events_cov_;
    beta_E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_beta_events_cov_;
    beta_E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_beta_events_cov_;
    beta_E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_beta_events_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<FDAGWR_COVARIATES_TYPES::EVENT>()] = beta_E_input;
    //input of S    
    Rcpp::List S_input;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::q] = q_S;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_stations_cov_;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_stations_cov_;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_stations_cov_;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_stations_cov_;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = toRList(coefficients_stations_cov_,false);
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties] = lambda_stations_cov_;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::coords] = Rcpp::wrap(coordinates_stations_out_);
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker] = kernel_bandwith_stations;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<FDAGWR_COVARIATES_TYPES::STATION>()] = S_input;
    //input of Beta S
    Rcpp::List beta_S_input;
    beta_S_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_beta_stations_cov_;
    beta_S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_beta_stations_cov_;
    beta_S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_beta_stations_cov_;
    beta_S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_beta_stations_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<FDAGWR_COVARIATES_TYPES::STATION>()] = beta_S_input;
    //domain
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::n] = number_of_statistical_units_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::a] = a;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::b] = b;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::abscissa] = abscissa_points_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::bf_estimate] = bf_estimation;
    //adding all the elements to perform prediction
    elem_for_pred[FDAGWR_HELPERS_for_PRED_NAMES::inputs_info] = inputs_info;
    l[FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred] = elem_for_pred;

    return l;
}


/*!
* @brief
* @param coeff_stationary_cov_to_pred
* @param coeff_events_cov_to_pred
* @param coordinates_events_to_pred
* @param coeff_stations_cov_to_pred
* @param coordinates_stations_to_pred
* @param units_to_be_predicted number of units to be predicted
* @param abscissa_ev abscissa for which then the predicted reponse and betas are made available
* @param model_fitted: an R list containing:
*         - "FGWR": string containing the type of fgwr used ("FGWR_FMS_ESC")
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
*                 - "basis_num": the number of basis used for performing the beta basis expansion (eespective elements of n_basis_beta_stations_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stations_cov)
*         - "Beta_s": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "predictor_info": a list containing partial residuals and information of the fitted model to perform predictions for new statistical units (derives from the fitting):
*                             - "partial_res": a list containing information to compute the partial residuals:
*                                              - "c_tilde_hat": vector of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon
*                                              - "A__": vector of matrices with the operator A_e for each statistical unit
*                                              - "B__for_K": vector of matrices with the operator B_e used for the K_e_s(t) computation, for each statistical unit
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": 
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "Response reconstruction weights": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "beta_Stationary": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "cov_Event": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                                            - "penalizations":
*                                                            - "coordinates":
*                                                            - "kernel_bwd_":
*                                              - "beta_Event": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "cov_Station": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                                            - "penalizatins":
*                                                            - "coordinates":
*                                                            - "kernel_bwd_":
*                                              - "beta_Station": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "n": number of units used to train
*                                              - "a"
*                                              - "b"
*                                              - "abscissa": abscissa for which we have the training set raw evalautions of response and covariates
* @param n_intervals_trapezoidal_quadrature
* @param target_error_trapezoidal_quadrature
* @param max_iterations_trapezoidal_quadrature
* @param num_threads
* @return an R list containing the the response predicted
*         - "FGWR_predictor": string containing the model 
*         - "prediction": list containing:
*                        - "prediction_ev": the raw evaluation of each unit response
*                        - "abscissa": the abscissa points for which the prediction is available
*         - "Bc_pred": list containing, for each sttionary covariate
*                      - "basis_coeff":
*                      - "basis_num": 
*                      - "basis_type":
*                      - "knots":
*         - "Beta_c_pred": list containing, for each stationary cov
*                           - "Beta_eval"
*                           - "Abscissa"
*         - "Be_pred": list containing, for each sttionary covariate
*                      - "basis_coeff":
*                      - "basis_num": 
*                      - "basis_type":
*                      - "knots":
*         - "Beta_e_pred": list containing, for each stationary cov
*                           - "Beta_eval" (list)
*                           - "Abscissa"
*         - "Bs_pred": list containing, for each sttionary covariate
*                      - "basis_coeff":
*                      - "basis_num": 
*                      - "basis_type":
*                      - "knots":
*         - "Beta_s_pred": list containing, for each stationary cov
*                           - "Beta_eval" (list)
*                           - "Abscissa"
* @details NB: LE COVARIATE DEVONO ESSERE SAMPLATE IN CORRISPONDENZA DEI SAMPLE POINTS CHE SONO STATI USATI NEL FITTING
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
                              int n_intervals_trapezoidal_quadrature = 100,
                              double target_error_trapezoidal_quadrature = 1e-3,
                              int max_iterations_trapezoidal_quadrature = 100,
                              Rcpp::Nullable<int> num_threads = R_NilValue)
{
    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'


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
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_trapezoidal_quadrature(n_intervals_trapezoidal_quadrature);
    // TARGET ERROR WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    double target_error = wrap_and_check_target_error_trapezoidal_quadrature(target_error_trapezoidal_quadrature);
    // MAXIMUM NUMBER OF ITERATIONS WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int max_iterations = wrap_and_check_max_iterations_trapezoidal_quadrature(max_iterations_trapezoidal_quadrature);


    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred];
    //list with partial residuals
    Rcpp::List partial_residuals = fitted_model[FDAGWR_HELPERS_for_PRED_NAMES::p_res];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[FDAGWR_HELPERS_for_PRED_NAMES::inputs_info];
    //list with elements of the response
    Rcpp::List response_input            = training_input[covariate_type<_RESPONSE_>()];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input      = training_input[covariate_type<_REC_WEIGHTS_>()];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<_STATIONARY_>()];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<_STATIONARY_>()];
    //list with elements of events-dependent covariates
    Rcpp::List events_cov_input          = training_input[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<_EVENT_>()];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_events_cov_input     = training_input[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<_EVENT_>()];
    //list with elements of stations-dependent covariates
    Rcpp::List stations_cov_input        = training_input[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<_STATION_>()];
    //list with elements of the beta of stations-dependent covariates
    Rcpp::List beta_stations_cov_input   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<_STATION_>()];

    //ESTIMATION TECHNIQUE
    bool bf_estimation = training_input[FDAGWR_HELPERS_for_PRED_NAMES::bf_estimate];
    //DOMAIN INFORMATION
    std::size_t n_train = training_input[FDAGWR_HELPERS_for_PRED_NAMES::n];
    _FD_INPUT_TYPE_ a   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::a];
    _FD_INPUT_TYPE_ b   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::b];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);                         //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[FDAGWR_HELPERS_for_PRED_NAMES::abscissa];    //abscissa point for which the training data are discretized
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::string basis_type_response_   = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::size_t degree_basis_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]); 
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);  
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::q];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_       = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
    //EVENTS COV    
    std::size_t q_E                                   = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::q];
    std::vector<std::size_t> number_basis_events_cov_ = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_events_cov_  = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_events_cov_ = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_events_cov_       = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_events_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_ = wrap_covariates_coefficients<_EVENT_>(events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
    std::vector<double> lambda_events_cov_ = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties];
    auto coordinates_events_               = reader_data<_DATA_TYPE_,_NAN_REM_>(events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coords]);     
    double kernel_bandwith_events_cov_     = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker];
    //STATIONS COV  
    std::size_t q_S                                     = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::q];
    std::vector<std::size_t> number_basis_stations_cov_ = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_stations_cov_  = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_stations_cov_ = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stations_cov_       = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_stations_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_ = wrap_covariates_coefficients<_STATION_>(stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
    std::vector<double> lambda_stations_cov_ = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties];
    auto coordinates_stations_               = reader_data<_DATA_TYPE_,_NAN_REM_>(stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coords]);
    double kernel_bandwith_stations_cov_     = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker];    
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[FDAGWR_B_NAMES::bc];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);  //sono tutte Lc_jx1
        Bc.push_back(Bc_i);}
    //EVENTS BETAS  
    std::vector<std::size_t> number_basis_beta_events_cov_ = beta_events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_beta_events_cov_  = beta_events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = beta_events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_events_cov_ = beta_events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_beta_events_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size()); 
    //STATIONS BETAS    
    std::vector<std::size_t> number_basis_beta_stations_cov_ = beta_stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_beta_stations_cov_  = beta_stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = beta_stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stations_cov_ = beta_stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stations_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());
    //saving the betas basis expansion coefficients for station-dependent covariates
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix>> Bs; //vettore esterno: per ogni covariata S. Interno: per ogni unità di training
    Bs.reserve(q_S);
    Rcpp::List Bs_list = model_fitted[FDAGWR_B_NAMES::bs];
    for(std::size_t i = 0; i < q_S; ++i){
        Rcpp::List Bs_i_list = Bs_list[i];
        auto Bs_i = wrap_covariates_coefficients<_STATION_>(Bs_i_list[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
        Bs.push_back(Bs_i);}
    //PARTIAL RESIDUALS
    auto c_tilde_hat = reader_data<_DATA_TYPE_,_NAN_REM_>(partial_residuals[FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> A_E_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[FDAGWR_HELPERS_for_PRED_NAMES::p_res_A__]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> B_E_for_K_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[FDAGWR_HELPERS_for_PRED_NAMES::p_res_B__for_K]);


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
        {covariate_type<_STATIONARY_>(),Xc_new},
        {covariate_type<_EVENT_>(),Xe_new},
        {covariate_type<_STATION_>(),Xs_new}};

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
        {covariate_type<_EVENT_>(),We_pred},
        {covariate_type<_STATION_>(),Ws_pred}};


    //fgwr predictor
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
                                                                                                   target_error,
                                                                                                   max_iterations,
                                                                                                   n_train,
                                                                                                   number_threads,
                                                                                                   bf_estimation);

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
    l[FDAGWR_HELPERS_for_PRED_NAMES::model_name + "_predictor"] = "predictor_" + algo_type<_FGWR_ALGO_>();
    l[FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter] = estimation_iter(bf_estimation);
    //predictions
    l[FDAGWR_HELPERS_for_PRED_NAMES::pred] = y_pred_ev_R;
    //stationary covariate basis expansion coefficients for beta_c
    l[FDAGWR_B_NAMES::bc + "_pred"]  = b_coefficients[FDAGWR_B_NAMES::bc];
    //beta_c
    l[FDAGWR_BETAS_NAMES::beta_c + "_pred"] = betas[FDAGWR_BETAS_NAMES::beta_c];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[FDAGWR_B_NAMES::be + "_pred"]  = b_coefficients[FDAGWR_B_NAMES::be];
    //beta_e
    l[FDAGWR_BETAS_NAMES::beta_e + "_pred"] = betas[FDAGWR_BETAS_NAMES::beta_e];
    //station-dependent covariate basis expansion coefficients for beta_s
    l[FDAGWR_B_NAMES::bs + "_pred"]  = b_coefficients[FDAGWR_B_NAMES::bs];
    //beta_s
    l[FDAGWR_BETAS_NAMES::beta_s + "_pred"] = betas[FDAGWR_BETAS_NAMES::beta_s];

    return l;
}





/*!
* @brief Function to perform Functional Multi-Source Geographically Weighted Regression SEC.
* @param y_points matrix of double containing the raw response: each row represents a specific abscissa for which the response evaluation is available, each column a statistical unit. Response isa already reconstructed.
* @param t_points vector of double with the abscissa points with respect which the raw evaluations of y_points are available (length of t_points is equal to the number of rows of y_points).
* @param left_extreme_domain double indicating the left extreme of the functional data (not necessarily the smaller element in t_points).
* @param right_extreme_domain double indicating the right extreme of the functional data (not necessarily the biggest element in t_points).
* @param coeff_y_points matrix of double containing the coefficient of response's basis expansion: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
* @param knots_y_points vector of double with the abscissa points with respect which the basis expansions of the response and response reconstruction weights are performed (all elements contained in [a,b]). 
* @param degree_basis_y_points non-negative integer: the degree of the basis used for the basis expansion of the (functional) response. Default explained below (can be NULL).
* @param n_basis_y_points positive integer: number of basis for the basis expansion of the (functional) response. It must match number of rows of coeff_y_points. Default explained below (can be NULL).
* @param coeff_rec_weights_y_points matrix of double containing the coefficients of the weights to reconstruct the (functional) response: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                                   More about response reconstruction weights meaning can be found in the link below.
* @param degree_basis_rec_weights_y_points non-negative integer: the degree of the basis used for response reconstruction weights. Default explained below (can be NULL).
* @param n_basis_rec_weights_y_points positive integer: number of basis for the basis expansion of response reconstruction weights. It must match number of rows of coeff_rec_weights_y_points. Default explained below (can be NULL).
* @param coeff_stationary_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                             The name of the i-th element is the name of the i-th stationary covariate (default: "reg.Ci").
* @param basis_types_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariates basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate. Default explained below (can be NULL).
* @param n_basis_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate. It must match number of rows of the i-th element of coeff_stationary_cov. Default explained below (can be NULL).
* @param penalization_stationary_cov vector of non-negative double: element i-th is the penalization used for the i-th stationary covariate.
* @param knots_beta_stationary_cov vector of double with the abscissa points with respect which the basis expansions of the stationary covariates (functional regression) coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_stationary_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param n_basis_beta_stationary_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param coeff_events_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th events-dependent covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                             The name of the i-th element is the name of the i-th events-dependent covariate (default: "reg.Ei").
* @param basis_types_events_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariates basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_events_cov vector of double with the abscissa points with respect which the basis expansions of the events-dependent covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_events_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th events-dependent covariate. Default explained below (can be NULL).
* @param n_basis_events_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th events-dependent covariate. It must match number of rows of the i-th element of coeff_events_cov. Default explained below (can be NULL).
* @param penalization_events_cov vector of non-negative double: element i-th is the penalization used for the i-th events-dependent covariate.
* @param coordinates_events matrix of double containing the UTM coordinates of the event of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
* @param kernel_bandwith_events positive double indicating the bandwith of the gaussian kernel used to smooth the distances within events.
* @param knots_beta_events_cov vector of double with the abscissa points with respect which the basis expansions of the events-dependent covariates (functional regression) coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_events_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th events-dependent covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param n_basis_beta_events_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th events-dependent covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param coeff_stations_cov list of matrices of doubles: element i-th containing the coefficients for the basis expansion of the i-th stations-dependent covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
*                             The name of the i-th element is the name of the i-th stations-dependent covariate (default: "reg.Si").
* @param basis_types_stations_cov vector of strings, element i-th containing the type of basis used for the i-th stations-dependent covariates basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param knots_stations_cov vector of double with the abscissa points with respect which the basis expansions of the stations-dependent covariates are performed (all elements contained in [a,b]). 
* @param degrees_basis_stations_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stations-dependent covariate. Default explained below (can be NULL).
* @param n_basis_stations_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stations-dependent covariate. It must match number of rows of the i-th element of coeff_stations_cov. Default explained below (can be NULL).
* @param penalization_stations_cov vector of non-negative double: element i-th is the penalization used for the i-th stations-dependent covariate.
* @param coordinates_stations matrix of double containing the UTM coordinates of the station of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
* @param kernel_bandwith_stations positive double indicating the bandwith of the gaussian kernel used to smooth the distances within stations.
* @param knots_beta_stations_cov vector of double with the abscissa points with respect which the basis expansions of the stations-dependent covariates (functional regression) coefficients are performed (all elements contained in [a,b]). 
* @param degrees_basis_beta_stations_cov vector of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stations-dependent covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param n_basis_beta_stations_cov vector of positive integers: element i-th is the number of basis for the basis expansion of the i-th stations-dependent covariate (functional regression) coefficient. Default explained below (can be NULL).
* @param n_knots_smoothing number of knots used to perform the smoothing on the response obtained leaving out all the non-stationary components
* @param n_intervals_trapezoidal_quadrature number of intervals used while performing integration via adaptive trapezoidal quadrature rule
* @param target_error_trapezoidal_quadrature target error while integrating via adaptive trapezoidal quadrature rule
* @param max_iterations_trapezoidal_quadrature max number of iterations for integrating via adaptive trapezoidal quadrature rule
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine running fmsgwr.
* @param basis_type_y_points string containing the type of basis used for the (functional) response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_type_rec_weights_y_points string containing the type of basis used for the weights to reconstruct the (functional) response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariates (functional regression) coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_events_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariates (functional regression) coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stations_cov vector of strings, element i-th containing the type of basis used for the i-th stations-dependent covariates (functional regression) coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @return an R list containing:
*         - "FGWR": string containing the type of fgwr used ("FGWR_FMS_SEC")
*         - "Bc": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Basis_coeff": a Lc_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "Basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective element of basis_types_beta_stationary_cov)
*                 - "Basis_number": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_stationary_cov)
*                 - "Basis_knots": the knots used to create the basis system for the beta (it is the input knots_beta_stationary_cov)
*         - "Beta_c": a list containing, for each stationary covariate regression coefficent (each element is named with the element names in the list coeff_stationary_cov (default, if not given: "CovC*")) a list with:
*                 - "Beta_eval": a vector of double containing the discrete evaluations of the stationary beta
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "Be": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "Basis_coeff": a list, containg, for each unit, a Le_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "Basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_events_cov)
*                 - "Basis_number": the number of basis used for performing the beta basis expansion (respective elements of n_basis_beta_events_cov)
*                 - "Basis_knots": the knots used to create the basis system for the beta (it is the input knots_beta_events_cov)
*         - "Beta_e": a list containing, for each event-dependent covariate regression coefficent (each element is named with the element names in the list coeff_events_cov (default, if not given: "CovE*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)  
*         - "Bs": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "Basis_coeff": a list, containg, for each unit, a Ls_jx1 vector of double, containing the coefficients of the basis expansion of the beta
*                 - "Basis_type": a string containing the basis type over which the beta basis expansion is performed. Possible values: "bsplines", "constant". (Respective elements of basis_types_beta_stations_cov)
*                 - "Basis_number": the number of basis used for performing the beta basis expansion (eespective elements of n_basis_beta_stations_cov)
*                 - "Basis_knots": the knots used to create the basis system for the beta (it is the input knots_beta_stations_cov)
*         - "Beta_s": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "predictor_info": a list containing partial residuals and information of the fitted model to perform predictions for new statistical units:
*                             - "partial_res": a list containing information to compute the partial residuals:
*                                              - "c_tilde_hat": vector of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon
*                                              - "A__": vector of matrices with the operator A_e for each statistical unit
*                                              - "B__for_K": vector of matrices with the operator B_e used for the K_e_s(t) computation, for each statistical unit
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "Response reconstruction weights": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "cov_Event": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                                            - "penalizations":
*                                                            - "coordinates":
*                                                            - "kernel_bwd_Event":
*                                              - "beta_Event": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "cov_Station": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                                            - "penalizatins":
*                                                            - "coordinates":
*                                                            - "kernel_bwd_Station":
*                                              - "beta_Station": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "a" 
*                                              - "b"
*                                              - "abscissa"
* @details constant basis are used, for a covariate, if it resembles a scalar shape. It consists of a straight line with y-value equal to 1 all over the data domain.
*          Can be seen as a B-spline basis with degree 0, number of basis 1, using one knot, consequently having only one coefficient for the only basis for each statistical unit.
*          fmsgwr sets all the feats accordingly if reads constant basis.
*          However, recall that the response is a functional datum, as the regressors coefficients. Since the package's basis variety could be hopefully enlarged in the future 
*          (for example, introducing Fourier basis for handling data that present periodical behaviors), the input parameters regarding basis types for response, response reconstruction
*          weights and regressors coefficients are left at the end of the input list, and defaulted as NULL. Consequently they will use a B-spline basis system, and should NOT use a constant basis,
*          Recall to perform externally the basis expansion before using the package, and afterwards passing basis types, degree and number and basis expansion coefficients and knots coherently
* @note a little excursion about degree and number of basis passed as input. For each specific covariate, or the response, if using B-spline basis, remember that number of knots = number of basis - degree + 1. 
*       By default, if passing NULL, fmsgwr uses a cubic B-spline system of basis, the number of basis is computed coherently from the number of knots (that is the only mandatory input parameter).
*       Passing only the degree of the bsplines, the number of basis used will be set accordingly, and viceversa if passing only the number of basis. 
*       But, take care that the number of basis used has to match the number of rows of coefficients matrix (for EACH type of basis). If not, an exception is thrown. No problems arise if letting fmsgwr defaulting the number of basis.
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
                      bool bf_estimation = false,                      
                      int n_knots_smoothing = 100,
                      int n_intervals_trapezoidal_quadrature = 100,
                      double target_error_trapezoidal_quadrature = 1e-3,
                      int max_iterations_trapezoidal_quadrature = 100,
                      Rcpp::Nullable<int> num_threads = R_NilValue,
                      std::string basis_type_y_points = "bsplines",
                      std::string basis_type_rec_weights_y_points = "bsplines",
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_events_cov = R_NilValue,
                      Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stations_cov = R_NilValue)
{
    //funzione per il multi-source gwr
    //  !!!!!!!! NB: l'ordine delle basi su c++ corrisponde al degree su R !!!!!
    Rcout << "Functional Multi-Source Geographically Weighted Regression SEC" << std::endl;


    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'

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
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_trapezoidal_quadrature(n_intervals_trapezoidal_quadrature);
    // TARGET ERROR WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    double target_error = wrap_and_check_target_error_trapezoidal_quadrature(target_error_trapezoidal_quadrature);
    // MAXIMUM NUMBER OF ITERATIONS WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int max_iterations = wrap_and_check_max_iterations_trapezoidal_quadrature(max_iterations_trapezoidal_quadrature);


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
    std::size_t number_basis_response_ = number_and_degree_basis_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t degree_basis_response_ = number_and_degree_basis_response_[FDAGWR_FEATS::degree_basis_string];
    check_dim_input<_RESPONSE_>(number_basis_response_,coefficients_response_.rows(),"response coefficients matrix rows");
    check_dim_input<_RESPONSE_>(number_of_statistical_units_,coefficients_response_.cols(),"response coefficients matrix columns");     
    //response reconstruction weights
    auto number_and_degree_basis_rec_weights_response_ = wrap_and_check_basis_number_and_degree<_REC_WEIGHTS_>(n_basis_rec_weights_y_points,degree_basis_rec_weights_y_points,knots_response_.size(),basis_type_rec_weights_response_);
    std::size_t number_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t degree_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[FDAGWR_FEATS::degree_basis_string];
    check_dim_input<_REC_WEIGHTS_>(number_basis_rec_weights_response_,coefficients_rec_weights_response_.rows(),"response reconstruction weights coefficients matrix rows");
    check_dim_input<_REC_WEIGHTS_>(number_of_statistical_units_,coefficients_rec_weights_response_.cols(),"response reconstruction weights coefficients matrix columns");     
    //stationary cov
    auto number_and_degree_basis_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_stationary_cov,degrees_basis_stationary_cov,knots_stationary_cov_.size(),q_C,basis_types_stationary_cov_);
    std::vector<std::size_t> number_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[FDAGWR_FEATS::degree_basis_string];
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stationary cov
    auto number_and_degree_basis_beta_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_beta_stationary_cov,degrees_basis_beta_stationary_cov,knots_beta_stationary_cov_.size(),q_C,basis_types_beta_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[FDAGWR_FEATS::degree_basis_string];
    //events cov    
    auto number_and_degree_basis_events_cov_ = wrap_and_check_basis_number_and_degree<_EVENT_>(n_basis_events_cov,degrees_basis_events_cov,knots_events_cov_.size(),q_E,basis_types_events_cov_);
    std::vector<std::size_t> number_basis_events_cov_ = number_and_degree_basis_events_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_events_cov_ = number_and_degree_basis_events_cov_[FDAGWR_FEATS::degree_basis_string];
    for(std::size_t i = 0; i < q_E; ++i){   
        check_dim_input<_EVENT_>(number_basis_events_cov_[i],coefficients_events_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_EVENT_>(number_of_statistical_units_,coefficients_events_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta events cov
    auto number_and_degree_basis_beta_events_cov_ = wrap_and_check_basis_number_and_degree<_EVENT_>(n_basis_beta_events_cov,degrees_basis_beta_events_cov,knots_beta_events_cov_.size(),q_E,basis_types_beta_events_cov_);
    std::vector<std::size_t> number_basis_beta_events_cov_ = number_and_degree_basis_beta_events_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = number_and_degree_basis_beta_events_cov_[FDAGWR_FEATS::degree_basis_string];
    //stations cov
    auto number_and_degree_basis_stations_cov_ = wrap_and_check_basis_number_and_degree<_STATION_>(n_basis_stations_cov,degrees_basis_stations_cov,knots_stations_cov_.size(),q_S,basis_types_stations_cov_);
    std::vector<std::size_t> number_basis_stations_cov_ = number_and_degree_basis_stations_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_stations_cov_ = number_and_degree_basis_stations_cov_[FDAGWR_FEATS::degree_basis_string];
    for(std::size_t i = 0; i < q_S; ++i){   
        check_dim_input<_STATION_>(number_basis_stations_cov_[i],coefficients_stations_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATION_>(number_of_statistical_units_,coefficients_stations_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stations cov 
    auto number_and_degree_basis_beta_stations_cov_ = wrap_and_check_basis_number_and_degree<_STATION_>(n_basis_beta_stations_cov,degrees_basis_beta_stations_cov,knots_beta_stations_cov_.size(),q_S,basis_types_beta_stations_cov_);
    std::vector<std::size_t> number_basis_beta_stations_cov_ = number_and_degree_basis_beta_stations_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = number_and_degree_basis_beta_stations_cov_[FDAGWR_FEATS::degree_basis_string];


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


    //fgwr algorithm
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
                                                                                    target_error,
                                                                                    max_iterations,
                                                                                    abscissa_points_,
                                                                                    number_of_statistical_units_,
                                                                                    number_threads,
                                                                                    bf_estimation);

    //computing the b
    fgwr_algo->compute();
    //evaluating the betas   
    fgwr_algo->evalBetas();

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
    //regression model used 
    l[FDAGWR_HELPERS_for_PRED_NAMES::model_name] = algo_type<_FGWR_ALGO_>();
    //stationary covariate basis expansion coefficients for beta_c
    l[FDAGWR_B_NAMES::bc]  = b_coefficients[FDAGWR_B_NAMES::bc];
    //beta_c
    l[FDAGWR_BETAS_NAMES::beta_c] = betas[FDAGWR_BETAS_NAMES::beta_c];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[FDAGWR_B_NAMES::be]  = b_coefficients[FDAGWR_B_NAMES::be];
    //beta_e
    l[FDAGWR_BETAS_NAMES::beta_e] = betas[FDAGWR_BETAS_NAMES::beta_e];
    //station-dependent covariate basis expansion coefficients for beta_s
    l[FDAGWR_B_NAMES::bs]  = b_coefficients[FDAGWR_B_NAMES::bs];
    //beta_s
    l[FDAGWR_BETAS_NAMES::beta_s] = betas[FDAGWR_BETAS_NAMES::beta_s];

    //returning all the elements needed to perform prediction
    Rcpp::List elem_for_pred;
    Rcpp::List inputs_info;
    //p res
    elem_for_pred[FDAGWR_HELPERS_for_PRED_NAMES::p_res] = p_res;
    l[FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter] = estimation_iter(bf_estimation);
    //input of y
    Rcpp::List response_input;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_type_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = Rcpp::wrap(coefficients_response_out_);
    inputs_info[covariate_type<FDAGWR_COVARIATES_TYPES::RESPONSE>()] = response_input;
    //input of w for y  
    Rcpp::List response_rec_w_input;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_type_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = Rcpp::wrap(coefficients_rec_weights_response_out_);
    inputs_info[covariate_type<FDAGWR_COVARIATES_TYPES::REC_WEIGHTS>()] = response_rec_w_input;
    //input of C
    Rcpp::List C_input;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::q] = q_C;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = toRList(coefficients_stationary_cov_,false);
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<FDAGWR_COVARIATES_TYPES::STATIONARY>()] = C_input;
    //input of Beta C   
    Rcpp::List beta_C_input;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_beta_stationary_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<FDAGWR_COVARIATES_TYPES::STATIONARY>()] = beta_C_input;
    //input of E
    Rcpp::List E_input;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::q] = q_E;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_events_cov_;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_events_cov_;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_events_cov_;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_events_cov_;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = toRList(coefficients_events_cov_,false);
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties] = lambda_events_cov_;
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::coords] = Rcpp::wrap(coordinates_events_out_);
    E_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker] = kernel_bandwith_events;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<FDAGWR_COVARIATES_TYPES::EVENT>()] = E_input;
    //input of Beta E   
    Rcpp::List beta_E_input;
    beta_E_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_beta_events_cov_;
    beta_E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_beta_events_cov_;
    beta_E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_beta_events_cov_;
    beta_E_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_beta_events_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<FDAGWR_COVARIATES_TYPES::EVENT>()] = beta_E_input;
    //input of S    
    Rcpp::List S_input;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::q] = q_S;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_stations_cov_;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_stations_cov_;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_stations_cov_;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_stations_cov_;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = toRList(coefficients_stations_cov_,false);
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties] = lambda_stations_cov_;
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::coords] = Rcpp::wrap(coordinates_stations_out_);
    S_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker] = kernel_bandwith_stations;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<FDAGWR_COVARIATES_TYPES::STATION>()] = S_input;
    //input of Beta S
    Rcpp::List beta_S_input;
    beta_S_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_beta_stations_cov_;
    beta_S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_beta_stations_cov_;
    beta_S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_beta_stations_cov_;
    beta_S_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_beta_stations_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<FDAGWR_COVARIATES_TYPES::STATION>()] = beta_S_input;
    //domain
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::n] = number_of_statistical_units_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::a] = a;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::b] = b;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::abscissa] = abscissa_points_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::bf_estimate] = bf_estimation;
    //adding all the elements to perform prediction
    elem_for_pred[FDAGWR_HELPERS_for_PRED_NAMES::inputs_info] = inputs_info;
    l[FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred] = elem_for_pred;

    return l;
}


/*!
* @brief
* @param coeff_stationary_cov_to_pred
* @param coeff_events_cov_to_pred
* @param coordinates_events_to_pred
* @param coeff_stations_cov_to_pred
* @param coordinates_stations_to_pred
* @param units_to_be_predicted number of units to be predicted
* @param abscissa_ev abscissa for which then the predicted reponse and betas are made available
* @param model_fitted: an R list containing:
*         - "FGWR": string containing the type of fgwr used ("FGWR_FMS_SEC")
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
*                 - "basis_num": the number of basis used for performing the beta basis expansion (eespective elements of n_basis_beta_stations_cov)
*                 - "knots": the knots used to create the basis system for the beta (it is the input knots_beta_stations_cov)
*         - "Beta_s": a list containing, for each station-dependent covariate regression coefficent (each element is named with the element names in the list coeff_stations_cov (default, if not given: "CovS*")) a list with:
*                 - "Beta_eval": a list containing vectors of double with the discrete evaluation of the non-stationary beta, one for each statistical unit
*                 - "Abscissa": the domain points for which the evaluation of the beta is available (it is the input t_points)
*         - "predictor_info": a list containing partial residuals and information of the fitted model to perform predictions for new statistical units (derives from the fitting):
*                             - "partial_res": a list containing information to compute the partial residuals:
*                                              - "c_tilde_hat": vector of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon
*                                              - "A__": vector of matrices with the operator A_e for each statistical unit
*                                              - "B__for_K": vector of matrices with the operator B_e used for the K_e_s(t) computation, for each statistical unit
*                             - "inputs_info": a list containing information about the data used to fit the model:
*                                              - "Response": list:
*                                                            - "basis_num": 
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "Response reconstruction weights": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                              - "cov_Stationary": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "beta_Stationary": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "cov_Event": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                                            - "penalizations":
*                                                            - "coordinates":
*                                                            - "kernel_bwd_":
*                                              - "beta_Event": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "cov_Station": list:
*                                                            - "number_covariates"
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                                            - "basis_coeff":
*                                                            - "penalizatins":
*                                                            - "coordinates":
*                                                            - "kernel_bwd_":
*                                              - "beta_Station": list:
*                                                            - "basis_num":
*                                                            - "basis_type":
*                                                            - "basis_deg":
*                                                            - "knots":
*                                              - "n": number of units used to train
*                                              - "a"
*                                              - "b"
*                                              - "abscissa": abscissa for which we have the training set raw evalautions of response and covariates
* @param n_intervals_trapezoidal_quadrature
* @param target_error_trapezoidal_quadrature
* @param max_iterations_trapezoidal_quadrature
* @param num_threads
* @return an R list containing the the response predicted
*         - "FGWR_predictor": string containing the model 
*         - "prediction": list containing:
*                        - "prediction_ev": the raw evaluation of each unit response
*                        - "abscissa": the abscissa points for which the prediction is available
*         - "Bc_pred": list containing, for each sttionary covariate
*                      - "basis_coeff":
*                      - "basis_num": 
*                      - "basis_type":
*                      - "knots":
*         - "Beta_c_pred": list containing, for each stationary cov
*                           - "Beta_eval"
*                           - "Abscissa"
*         - "Be_pred": list containing, for each sttionary covariate
*                      - "basis_coeff":
*                      - "basis_num": 
*                      - "basis_type":
*                      - "knots":
*         - "Beta_e_pred": list containing, for each stationary cov
*                           - "Beta_eval" (list)
*                           - "Abscissa"
*         - "Bs_pred": list containing, for each sttionary covariate
*                      - "basis_coeff":
*                      - "basis_num": 
*                      - "basis_type":
*                      - "knots":
*         - "Beta_s_pred": list containing, for each stationary cov
*                           - "Beta_eval" (list)
*                           - "Abscissa"
* @details NB: LE COVARIATE DEVONO ESSERE SAMPLATE IN CORRISPONDENZA DEI SAMPLE POINTS CHE SONO STATI USATI NEL FITTING
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
                              int n_intervals_trapezoidal_quadrature = 100,
                              double target_error_trapezoidal_quadrature = 1e-3,
                              int max_iterations_trapezoidal_quadrature = 100,
                              Rcpp::Nullable<int> num_threads = R_NilValue)
{
    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'


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
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_trapezoidal_quadrature(n_intervals_trapezoidal_quadrature);
    // TARGET ERROR WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    double target_error = wrap_and_check_target_error_trapezoidal_quadrature(target_error_trapezoidal_quadrature);
    // MAXIMUM NUMBER OF ITERATIONS WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int max_iterations = wrap_and_check_max_iterations_trapezoidal_quadrature(max_iterations_trapezoidal_quadrature);


    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred];
    //list with partial residuals
    Rcpp::List partial_residuals = fitted_model[FDAGWR_HELPERS_for_PRED_NAMES::p_res];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[FDAGWR_HELPERS_for_PRED_NAMES::inputs_info];
    //list with elements of the response
    Rcpp::List response_input            = training_input[covariate_type<_RESPONSE_>()];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input      = training_input[covariate_type<_REC_WEIGHTS_>()];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<_STATIONARY_>()];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<_STATIONARY_>()];
    //list with elements of events-dependent covariates
    Rcpp::List events_cov_input          = training_input[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<_EVENT_>()];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_events_cov_input     = training_input[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<_EVENT_>()];
    //list with elements of stations-dependent covariates
    Rcpp::List stations_cov_input        = training_input[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<_STATION_>()];
    //list with elements of the beta of stations-dependent covariates
    Rcpp::List beta_stations_cov_input   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<_STATION_>()];

    //ESTIMATION TECHNIQUE
    bool bf_estimation = training_input[FDAGWR_HELPERS_for_PRED_NAMES::bf_estimate];
    //DOMAIN INFORMATION
    std::size_t n_train = training_input[FDAGWR_HELPERS_for_PRED_NAMES::n];
    _FD_INPUT_TYPE_ a   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::a];
    _FD_INPUT_TYPE_ b   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::b];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);                         //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[FDAGWR_HELPERS_for_PRED_NAMES::abscissa];    //abscissa point for which the training data are discretized
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::string basis_type_response_   = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::size_t degree_basis_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]); 
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);  
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::q];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_       = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
    //EVENTS COV    
    std::size_t q_E                                   = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::q];
    std::vector<std::size_t> number_basis_events_cov_ = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_events_cov_  = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_events_cov_ = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_events_cov_       = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_events_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_events_cov_ = wrap_covariates_coefficients<_EVENT_>(events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
    std::vector<double> lambda_events_cov_ = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties];
    auto coordinates_events_               = reader_data<_DATA_TYPE_,_NAN_REM_>(events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coords]);     
    double kernel_bandwith_events_cov_     = events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker];
    //STATIONS COV  
    std::size_t q_S                                     = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::q];
    std::vector<std::size_t> number_basis_stations_cov_ = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_stations_cov_  = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_stations_cov_ = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stations_cov_       = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_stations_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stations_cov_ = wrap_covariates_coefficients<_STATION_>(stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
    std::vector<double> lambda_stations_cov_ = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties];
    auto coordinates_stations_               = reader_data<_DATA_TYPE_,_NAN_REM_>(stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coords]);
    double kernel_bandwith_stations_cov_     = stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker];    
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[FDAGWR_B_NAMES::bc];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);  //sono tutte Lc_jx1
        Bc.push_back(Bc_i);}
    //EVENTS BETAS  
    std::vector<std::size_t> number_basis_beta_events_cov_ = beta_events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_beta_events_cov_  = beta_events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_beta_events_cov_ = beta_events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_events_cov_ = beta_events_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_beta_events_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size()); 
    //saving the betas basis expansion coefficients for events-dependent covariates
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix>> Be; //vettore esterno: per ogni covariata E. Interno: per ogni unità di training
    Be.reserve(q_E);
    Rcpp::List Be_list = model_fitted[FDAGWR_B_NAMES::be];
    for(std::size_t i = 0; i < q_E; ++i){
        Rcpp::List Be_i_list = Be_list[i];
        auto Be_i = wrap_covariates_coefficients<_EVENT_>(Be_i_list[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
        Be.push_back(Be_i);}
    //STATIONS BETAS    
    std::vector<std::size_t> number_basis_beta_stations_cov_ = beta_stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_beta_stations_cov_  = beta_stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_beta_stations_cov_ = beta_stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stations_cov_ = beta_stations_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stations_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());
    //PARTIAL RESIDUALS
    auto c_tilde_hat = reader_data<_DATA_TYPE_,_NAN_REM_>(partial_residuals[FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> A_S_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[FDAGWR_HELPERS_for_PRED_NAMES::p_res_A__]);
    std::vector<FDAGWR_TRAITS::Dense_Matrix> B_S_for_K_i = wrap_covariates_coefficients<_RESPONSE_>(partial_residuals[FDAGWR_HELPERS_for_PRED_NAMES::p_res_B__for_K]);


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
        {covariate_type<_STATIONARY_>(),Xc_new},
        {covariate_type<_EVENT_>(),Xe_new},
        {covariate_type<_STATION_>(),Xs_new}};

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
        {covariate_type<_EVENT_>(),We_pred},
        {covariate_type<_STATION_>(),Ws_pred}};


    //fgwr predictor
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
                                                                                                   target_error,
                                                                                                   max_iterations,
                                                                                                   n_train,
                                                                                                   number_threads,
                                                                                                   bf_estimation);

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
    l[FDAGWR_HELPERS_for_PRED_NAMES::model_name] = algo_type<_FGWR_ALGO_>();
    l[FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter] = estimation_iter(bf_estimation);
    //predictions
    l[FDAGWR_HELPERS_for_PRED_NAMES::pred] = y_pred_ev_R;
    //stationary covariate basis expansion coefficients for beta_c
    l[FDAGWR_B_NAMES::bc + "_pred"]  = b_coefficients[FDAGWR_B_NAMES::bc];
    //beta_c
    l[FDAGWR_BETAS_NAMES::beta_c + "_pred"] = betas[FDAGWR_BETAS_NAMES::beta_c];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[FDAGWR_B_NAMES::be + "_pred"]  = b_coefficients[FDAGWR_B_NAMES::be];
    //beta_e
    l[FDAGWR_BETAS_NAMES::beta_e + "_pred"] = betas[FDAGWR_BETAS_NAMES::beta_e];
    //station-dependent covariate basis expansion coefficients for beta_s
    l[FDAGWR_B_NAMES::bs + "_pred"]  = b_coefficients[FDAGWR_B_NAMES::bs];
    //beta_s
    l[FDAGWR_BETAS_NAMES::beta_s + "_pred"] = betas[FDAGWR_BETAS_NAMES::beta_s];

    return l;
}





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
                 bool bf_estimation = false,                 
                 int n_knots_smoothing = 100,
                 int n_intervals_trapezoidal_quadrature = 100,
                 double target_error_trapezoidal_quadrature = 1e-3,
                 int max_iterations_trapezoidal_quadrature = 100,
                 Rcpp::Nullable<int> num_threads = R_NilValue,
                 std::string basis_type_y_points = "bsplines",
                 std::string basis_type_rec_weights_y_points = "bsplines",
                 Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue,
                 Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_non_stationary_cov = R_NilValue)
{
    //funzione per il multi-source gwr
    //  !!!!!!!! NB: l'ordine delle basi su c++ corrisponde al degree su R !!!!!
    Rcout << "Functional Mixed Geographically Weighted Regression" << std::endl;


    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'

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
    int n_intervals = wrap_and_check_n_intervals_trapezoidal_quadrature(n_intervals_trapezoidal_quadrature);
    // TARGET ERROR WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    double target_error = wrap_and_check_target_error_trapezoidal_quadrature(target_error_trapezoidal_quadrature);
    // MAXIMUM NUMBER OF ITERATIONS WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int max_iterations = wrap_and_check_max_iterations_trapezoidal_quadrature(max_iterations_trapezoidal_quadrature);


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
    std::size_t number_basis_response_ = number_and_degree_basis_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t degree_basis_response_ = number_and_degree_basis_response_[FDAGWR_FEATS::degree_basis_string];
    check_dim_input<_RESPONSE_>(number_basis_response_,coefficients_response_.rows(),"response coefficients matrix rows");
    check_dim_input<_RESPONSE_>(number_of_statistical_units_,coefficients_response_.cols(),"response coefficients matrix columns");     
    //response reconstruction weights
    auto number_and_degree_basis_rec_weights_response_ = wrap_and_check_basis_number_and_degree<_REC_WEIGHTS_>(n_basis_rec_weights_y_points,degree_basis_rec_weights_y_points,knots_response_.size(),basis_type_rec_weights_response_);
    std::size_t number_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t degree_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[FDAGWR_FEATS::degree_basis_string];
    check_dim_input<_REC_WEIGHTS_>(number_basis_rec_weights_response_,coefficients_rec_weights_response_.rows(),"response reconstruction weights coefficients matrix rows");
    check_dim_input<_REC_WEIGHTS_>(number_of_statistical_units_,coefficients_rec_weights_response_.cols(),"response reconstruction weights coefficients matrix columns");     
    //stationary cov
    auto number_and_degree_basis_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_stationary_cov,degrees_basis_stationary_cov,knots_stationary_cov_.size(),q_C,basis_types_stationary_cov_);
    std::vector<std::size_t> number_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[FDAGWR_FEATS::degree_basis_string];
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stationary cov
    auto number_and_degree_basis_beta_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_beta_stationary_cov,degrees_basis_beta_stationary_cov,knots_beta_stationary_cov_.size(),q_C,basis_types_beta_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[FDAGWR_FEATS::degree_basis_string];
    //non stationary cov    
    auto number_and_degree_basis_non_stationary_cov_ = wrap_and_check_basis_number_and_degree<_NON_STATIONARY_>(n_basis_non_stationary_cov,degrees_basis_non_stationary_cov,knots_non_stationary_cov_.size(),q_NC,basis_types_non_stationary_cov_);
    std::vector<std::size_t> number_basis_non_stationary_cov_ = number_and_degree_basis_non_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_non_stationary_cov_ = number_and_degree_basis_non_stationary_cov_[FDAGWR_FEATS::degree_basis_string];
    for(std::size_t i = 0; i < q_NC; ++i){   
        check_dim_input<_NON_STATIONARY_>(number_basis_non_stationary_cov_[i],coefficients_non_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_NON_STATIONARY_>(number_of_statistical_units_,coefficients_non_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta non stationary cov
    auto number_and_degree_basis_beta_non_stationary_cov_ = wrap_and_check_basis_number_and_degree<_NON_STATIONARY_>(n_basis_beta_non_stationary_cov,degrees_basis_beta_non_stationary_cov,knots_beta_non_stationary_cov_.size(),q_NC,basis_types_beta_non_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_non_stationary_cov_ = number_and_degree_basis_beta_non_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_beta_non_stationary_cov_ = number_and_degree_basis_beta_non_stationary_cov_[FDAGWR_FEATS::degree_basis_string];


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


    //fgwr algorithm
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
                                                                                   target_error,
                                                                                   max_iterations,
                                                                                   abscissa_points_,
                                                                                   number_of_statistical_units_,
                                                                                   number_threads,
                                                                                   bf_estimation);

    //computing the b
    fgwr_algo->compute();
    //evaluating the betas   
    fgwr_algo->evalBetas();

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
    //regression model used and estimation technique
    l[FDAGWR_HELPERS_for_PRED_NAMES::model_name] = algo_type<_FGWR_ALGO_>();
    l[FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter] = estimation_iter(bf_estimation);
    //stationary covariate basis expansion coefficients for beta_c
    l[FDAGWR_B_NAMES::bc]  = b_coefficients[FDAGWR_B_NAMES::bc];
    //beta_c
    l[FDAGWR_BETAS_NAMES::beta_c] = betas[FDAGWR_BETAS_NAMES::beta_c];
    //non stationary covariate basis expansion coefficients for beta_nc
    l[FDAGWR_B_NAMES::bnc] = b_coefficients[FDAGWR_B_NAMES::bnc];
    //beta_nc
    l[FDAGWR_BETAS_NAMES::beta_nc] = betas[FDAGWR_BETAS_NAMES::beta_nc];

    //returning all the elements needed to perform prediction
    Rcpp::List elem_for_pred;
    Rcpp::List inputs_info;
    //p res
    elem_for_pred[FDAGWR_HELPERS_for_PRED_NAMES::p_res] = p_res;
    //input of y
    Rcpp::List response_input;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_type_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = Rcpp::wrap(coefficients_response_out_);
    inputs_info[covariate_type<FDAGWR_COVARIATES_TYPES::RESPONSE>()] = response_input;
    //input of w for y  
    Rcpp::List response_rec_w_input;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_type_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = Rcpp::wrap(coefficients_rec_weights_response_out_);
    inputs_info[covariate_type<FDAGWR_COVARIATES_TYPES::REC_WEIGHTS>()] = response_rec_w_input;
    //input of C
    Rcpp::List C_input;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::q] = q_C;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = toRList(coefficients_stationary_cov_,false);
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<FDAGWR_COVARIATES_TYPES::STATIONARY>()] = C_input;
    //input of Beta C   
    Rcpp::List beta_C_input;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_beta_stationary_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<FDAGWR_COVARIATES_TYPES::STATIONARY>()] = beta_C_input;
    //input of NC
    Rcpp::List NC_input;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::q] = q_NC;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_non_stationary_cov_;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_non_stationary_cov_;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_non_stationary_cov_;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_non_stationary_cov_;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = toRList(coefficients_non_stationary_cov_,false);
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties] = lambda_non_stationary_cov_;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::coords] = Rcpp::wrap(coordinates_non_stationary_out_);
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker] = kernel_bandwith_non_stationary;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<FDAGWR_COVARIATES_TYPES::NON_STATIONARY>()] = NC_input;
    //input of Beta NC   
    Rcpp::List beta_NC_input;
    beta_NC_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_beta_non_stationary_cov_;
    beta_NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_beta_non_stationary_cov_;
    beta_NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_beta_non_stationary_cov_;
    beta_NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_beta_non_stationary_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<FDAGWR_COVARIATES_TYPES::NON_STATIONARY>()] = beta_NC_input;
    //domain
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::n] = number_of_statistical_units_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::a] = a;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::b] = b;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::abscissa] = abscissa_points_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::bf_estimate] = bf_estimation;
    //adding all the elements to perform prediction
    elem_for_pred[FDAGWR_HELPERS_for_PRED_NAMES::inputs_info] = inputs_info;
    l[FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred] = elem_for_pred;

    return l;
}


//
// [[Rcpp::export]]
Rcpp::List predict_FMGWR(Rcpp::List coeff_stationary_cov_to_pred,
                         Rcpp::List coeff_non_stationary_cov_to_pred,
                         Rcpp::NumericMatrix coordinates_non_stationary_to_pred,   
                         int units_to_be_predicted,
                         Rcpp::NumericVector abscissa_ev,
                         Rcpp::List model_fitted,
                         int n_knots_smoothing_pred = 100,
                         int n_intervals_trapezoidal_quadrature = 100,
                         double target_error_trapezoidal_quadrature = 1e-3,
                         int max_iterations_trapezoidal_quadrature = 100,
                         Rcpp::Nullable<int> num_threads = R_NilValue)
{
    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'


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
    int n_intervals = wrap_and_check_n_intervals_trapezoidal_quadrature(n_intervals_trapezoidal_quadrature);
    // TARGET ERROR WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    double target_error = wrap_and_check_target_error_trapezoidal_quadrature(target_error_trapezoidal_quadrature);
    // MAXIMUM NUMBER OF ITERATIONS WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int max_iterations = wrap_and_check_max_iterations_trapezoidal_quadrature(max_iterations_trapezoidal_quadrature);



    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred];
    //list with partial residuals
    Rcpp::List partial_residuals = fitted_model[FDAGWR_HELPERS_for_PRED_NAMES::p_res];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[FDAGWR_HELPERS_for_PRED_NAMES::inputs_info];
    //list with elements of the response
    Rcpp::List response_input            = training_input[covariate_type<_RESPONSE_>()];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input      = training_input[covariate_type<_REC_WEIGHTS_>()];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<_STATIONARY_>()];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<_STATIONARY_>()];
    //list with elements of events-dependent covariates
    Rcpp::List non_stationary_cov_input          = training_input[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<_NON_STATIONARY_>()];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_non_stationary_cov_input     = training_input[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<_NON_STATIONARY_>()];

    //ESTIMATION TECHNIQUE
    bool bf_estimation = training_input[FDAGWR_HELPERS_for_PRED_NAMES::bf_estimate];
    //DOMAIN INFORMATION
    std::size_t n_train = training_input[FDAGWR_HELPERS_for_PRED_NAMES::n];
    _FD_INPUT_TYPE_ a   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::a];
    _FD_INPUT_TYPE_ b   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::b];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);                         //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[FDAGWR_HELPERS_for_PRED_NAMES::abscissa];    //abscissa point for which the training data are discretized
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::string basis_type_response_   = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::size_t degree_basis_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]); 
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);  
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::q];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_       = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
    //NON STATIONARY COV    
    std::size_t q_NC                                          = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::q];
    std::vector<std::size_t> number_basis_non_stationary_cov_ = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_non_stationary_cov_  = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_non_stationary_cov_ = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_non_stationary_cov_       = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_non_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_non_stationary_cov_.data(),knots_non_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_non_stationary_cov_ = wrap_covariates_coefficients<_NON_STATIONARY_>(non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
    std::vector<double> lambda_non_stationary_cov_ = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties];
    auto coordinates_non_stationary_               = reader_data<_DATA_TYPE_,_NAN_REM_>(non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coords]);     
    double kernel_bandwith_non_stationary_cov_     = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker];  
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[FDAGWR_B_NAMES::bc];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);  //sono tutte Lc_jx1
        Bc.push_back(Bc_i);}
    //NON-STATIONAY BETAS  
    std::vector<std::size_t> number_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_beta_non_stationary_cov_  = beta_non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_non_stationary_cov_ = beta_non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_beta_non_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_non_stationary_cov_.data(),knots_beta_non_stationary_cov_.size()); 
    //PARTIAL RESIDUALS
    auto c_tilde_hat = reader_data<_DATA_TYPE_,_NAN_REM_>(partial_residuals[FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat]);

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
        {covariate_type<_STATIONARY_>(),Xc_new},
        {covariate_type<_NON_STATIONARY_>(),Xnc_new}};

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
        {covariate_type<_NON_STATIONARY_>(),Wnc_pred}};


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
                                                                                                 target_error,
                                                                                                 max_iterations,
                                                                                                 n_train,
                                                                                                 number_threads,
                                                                                                 bf_estimation);

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
    l[FDAGWR_HELPERS_for_PRED_NAMES::model_name + "_predictor"] = "predictor_" + algo_type<_FGWR_ALGO_>();
    l[FDAGWR_HELPERS_for_PRED_NAMES::estimation_iter] = estimation_iter(bf_estimation);
    //predictions
    l[FDAGWR_HELPERS_for_PRED_NAMES::pred] = y_pred_ev_R;
    //stationary covariate basis expansion coefficients for beta_c
    l[FDAGWR_B_NAMES::bc + "_pred"]  = b_coefficients[FDAGWR_B_NAMES::bc];
    //beta_c
    l[FDAGWR_BETAS_NAMES::beta_c + "_pred"] = betas[FDAGWR_BETAS_NAMES::beta_c];
    //event-dependent covariate basis expansion coefficients for beta_e
    l[FDAGWR_B_NAMES::bnc + "_pred"]  = b_coefficients[FDAGWR_B_NAMES::bnc];
    //beta_e
    l[FDAGWR_BETAS_NAMES::beta_nc + "_pred"] = betas[FDAGWR_BETAS_NAMES::beta_nc];

    return l;
}





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
                int n_intervals_trapezoidal_quadrature = 100,
                double target_error_trapezoidal_quadrature = 1e-3,
                int max_iterations_trapezoidal_quadrature = 100,
                Rcpp::Nullable<int> num_threads = R_NilValue,
                std::string basis_type_y_points = "bsplines",
                std::string basis_type_rec_weights_y_points = "bsplines",
                Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue,
                Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_non_stationary_cov = R_NilValue)
{
    //funzione per il multi-source gwr
    //  !!!!!!!! NB: l'ordine delle basi su c++ corrisponde al degree su R !!!!!
    Rcout << "Functional Geographically Weighted Regression" << std::endl;


    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'

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
    int n_intervals = wrap_and_check_n_intervals_trapezoidal_quadrature(n_intervals_trapezoidal_quadrature);
    // TARGET ERROR WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    double target_error = wrap_and_check_target_error_trapezoidal_quadrature(target_error_trapezoidal_quadrature);
    // MAXIMUM NUMBER OF ITERATIONS WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int max_iterations = wrap_and_check_max_iterations_trapezoidal_quadrature(max_iterations_trapezoidal_quadrature);


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
    std::size_t number_basis_response_ = number_and_degree_basis_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t degree_basis_response_ = number_and_degree_basis_response_[FDAGWR_FEATS::degree_basis_string];
    check_dim_input<_RESPONSE_>(number_basis_response_,coefficients_response_.rows(),"response coefficients matrix rows");
    check_dim_input<_RESPONSE_>(number_of_statistical_units_,coefficients_response_.cols(),"response coefficients matrix columns");     
    //response reconstruction weights
    auto number_and_degree_basis_rec_weights_response_ = wrap_and_check_basis_number_and_degree<_REC_WEIGHTS_>(n_basis_rec_weights_y_points,degree_basis_rec_weights_y_points,knots_response_.size(),basis_type_rec_weights_response_);
    std::size_t number_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t degree_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[FDAGWR_FEATS::degree_basis_string];
    check_dim_input<_REC_WEIGHTS_>(number_basis_rec_weights_response_,coefficients_rec_weights_response_.rows(),"response reconstruction weights coefficients matrix rows");
    check_dim_input<_REC_WEIGHTS_>(number_of_statistical_units_,coefficients_rec_weights_response_.cols(),"response reconstruction weights coefficients matrix columns");     
    //non stationary cov    
    auto number_and_degree_basis_non_stationary_cov_ = wrap_and_check_basis_number_and_degree<_NON_STATIONARY_>(n_basis_non_stationary_cov,degrees_basis_non_stationary_cov,knots_non_stationary_cov_.size(),q_NC,basis_types_non_stationary_cov_);
    std::vector<std::size_t> number_basis_non_stationary_cov_ = number_and_degree_basis_non_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_non_stationary_cov_ = number_and_degree_basis_non_stationary_cov_[FDAGWR_FEATS::degree_basis_string];
    for(std::size_t i = 0; i < q_NC; ++i){   
        check_dim_input<_NON_STATIONARY_>(number_basis_non_stationary_cov_[i],coefficients_non_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_NON_STATIONARY_>(number_of_statistical_units_,coefficients_non_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta non stationary cov
    auto number_and_degree_basis_beta_non_stationary_cov_ = wrap_and_check_basis_number_and_degree<_NON_STATIONARY_>(n_basis_beta_non_stationary_cov,degrees_basis_beta_non_stationary_cov,knots_beta_non_stationary_cov_.size(),q_NC,basis_types_beta_non_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_non_stationary_cov_ = number_and_degree_basis_beta_non_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_beta_non_stationary_cov_ = number_and_degree_basis_beta_non_stationary_cov_[FDAGWR_FEATS::degree_basis_string];


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
                                                                                   target_error,
                                                                                   max_iterations,
                                                                                   abscissa_points_,
                                                                                   number_of_statistical_units_,
                                                                                   number_threads);

    //computing the b
    fgwr_algo->compute();
    //evaluating the betas   
    fgwr_algo->evalBetas();

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
    //regression model used 
    l[FDAGWR_HELPERS_for_PRED_NAMES::model_name] = algo_type<_FGWR_ALGO_>();
    //non stationary covariate basis expansion coefficients for beta_nc
    l[FDAGWR_B_NAMES::bnc] = b_coefficients[FDAGWR_B_NAMES::bnc];
    //beta_nc
    l[FDAGWR_BETAS_NAMES::beta_nc] = betas[FDAGWR_BETAS_NAMES::beta_nc];

    //returning all the elements needed to perform prediction
    Rcpp::List elem_for_pred;
    Rcpp::List inputs_info; 
    //input of y
    Rcpp::List response_input;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_type_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = Rcpp::wrap(coefficients_response_out_);
    inputs_info[covariate_type<FDAGWR_COVARIATES_TYPES::RESPONSE>()] = response_input;
    //input of w for y  
    Rcpp::List response_rec_w_input;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_type_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_rec_weights_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_response_;
    response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = Rcpp::wrap(coefficients_rec_weights_response_out_);
    inputs_info[covariate_type<FDAGWR_COVARIATES_TYPES::REC_WEIGHTS>()] = response_rec_w_input;
    //input of NC
    Rcpp::List NC_input;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::q] = q_NC;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_non_stationary_cov_;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_non_stationary_cov_;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_non_stationary_cov_;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_non_stationary_cov_;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = toRList(coefficients_non_stationary_cov_,false);
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties] = lambda_non_stationary_cov_;
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::coords] = Rcpp::wrap(coordinates_non_stationary_out_);
    NC_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker] = kernel_bandwith_non_stationary;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<FDAGWR_COVARIATES_TYPES::NON_STATIONARY>()] = NC_input;
    //input of Beta NC   
    Rcpp::List beta_NC_input;
    beta_NC_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_beta_non_stationary_cov_;
    beta_NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_beta_non_stationary_cov_;
    beta_NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_beta_non_stationary_cov_;
    beta_NC_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_beta_non_stationary_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<FDAGWR_COVARIATES_TYPES::NON_STATIONARY>()] = beta_NC_input;
    //domain
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::n] = number_of_statistical_units_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::a] = a;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::b] = b;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::abscissa] = abscissa_points_;
    //adding all the elements to perform prediction
    elem_for_pred[FDAGWR_HELPERS_for_PRED_NAMES::inputs_info] = inputs_info;
    l[FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred] = elem_for_pred;

    return l;
}


//
// [[Rcpp::export]]
Rcpp::List predict_FGWR(Rcpp::List coeff_non_stationary_cov_to_pred,
                        Rcpp::NumericMatrix coordinates_non_stationary_to_pred,   
                        int units_to_be_predicted,
                        Rcpp::NumericVector abscissa_ev,
                        Rcpp::List model_fitted,
                        int n_knots_smoothing_pred = 100,
                        int n_intervals_trapezoidal_quadrature = 100,
                        double target_error_trapezoidal_quadrature = 1e-3,
                        int max_iterations_trapezoidal_quadrature = 100,
                        Rcpp::Nullable<int> num_threads = R_NilValue)
{
    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'


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
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_trapezoidal_quadrature(n_intervals_trapezoidal_quadrature);
    // TARGET ERROR WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    double target_error = wrap_and_check_target_error_trapezoidal_quadrature(target_error_trapezoidal_quadrature);
    // MAXIMUM NUMBER OF ITERATIONS WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int max_iterations = wrap_and_check_max_iterations_trapezoidal_quadrature(max_iterations_trapezoidal_quadrature);


    ////////////////////////////////////////////////////////////
    /////// RETRIEVING INFORMATION FROM THE MODEL FITTED ///////
    ////////////////////////////////////////////////////////////
    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[FDAGWR_HELPERS_for_PRED_NAMES::inputs_info];
    //list with elements of the response
    Rcpp::List response_input            = training_input[covariate_type<_RESPONSE_>()];
    //list with elements of response reconstruction weights
    Rcpp::List response_rec_w_input      = training_input[covariate_type<_REC_WEIGHTS_>()];
    //list with elements of events-dependent covariates
    Rcpp::List non_stationary_cov_input          = training_input[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<_NON_STATIONARY_>()];
    //list with elements of the beta of events-dependent covariates
    Rcpp::List beta_non_stationary_cov_input     = training_input[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<_NON_STATIONARY_>()];


    //DOMAIN INFORMATION
    std::size_t n_train = training_input[FDAGWR_HELPERS_for_PRED_NAMES::n];
    _FD_INPUT_TYPE_ a   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::a];
    _FD_INPUT_TYPE_ b   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::b];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);                         //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[FDAGWR_HELPERS_for_PRED_NAMES::abscissa];    //abscissa point for which the training data are discretized
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::string basis_type_response_   = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::size_t degree_basis_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    auto coefficients_response_                               = reader_data<_DATA_TYPE_,_NAN_REM_>(response_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);  
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //RESPONDE RECONSTRUCTION WEIGHTS   
    std::size_t number_basis_rec_weights_response_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::string basis_type_rec_weights_response_   = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::size_t degree_basis_rec_weights_response_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_rec_w_ = response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_response_rec_w_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_rec_w_.data(),knots_response_rec_w_.size());
    auto coefficients_rec_weights_response_                         = reader_data<_DATA_TYPE_,_NAN_REM_>(response_rec_w_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);  
    //NON STATIONARY COV    
    std::size_t q_NC                                          = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::q];
    std::vector<std::size_t> number_basis_non_stationary_cov_ = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_non_stationary_cov_  = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_non_stationary_cov_ = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_non_stationary_cov_       = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_non_stationary_cov_eigen_w_             = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_non_stationary_cov_.data(),knots_non_stationary_cov_.size());
    std::vector<FDAGWR_TRAITS::Dense_Matrix> coefficients_non_stationary_cov_ = wrap_covariates_coefficients<_NON_STATIONARY_>(non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);
    std::vector<double> lambda_non_stationary_cov_ = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::penalties];
    auto coordinates_non_stationary_               = reader_data<_DATA_TYPE_,_NAN_REM_>(non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::coords]);     
    double kernel_bandwith_non_stationary_cov_     = non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::bdw_ker];  
    //NON-STATIONAY BETAS  
    std::vector<std::size_t> number_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_beta_non_stationary_cov_  = beta_non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_beta_non_stationary_cov_ = beta_non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_non_stationary_cov_ = beta_non_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
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
        {covariate_type<_NON_STATIONARY_>(),Xnc_new}};

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
        {covariate_type<_NON_STATIONARY_>(),Wnc_pred}};


    //fgwr predictor
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
                                                                                                 target_error,
                                                                                                 max_iterations,
                                                                                                 n_train,
                                                                                                 number_threads);

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
    l[FDAGWR_HELPERS_for_PRED_NAMES::model_name + "_predictor"] = "predictor_" + algo_type<_FGWR_ALGO_>();
    //predictions
    l[FDAGWR_HELPERS_for_PRED_NAMES::pred] = y_pred_ev_R;
    //event-dependent covariate basis expansion coefficients for beta_e
    l[FDAGWR_B_NAMES::bnc + "_pred"]  = b_coefficients[FDAGWR_B_NAMES::bnc];
    //beta_e
    l[FDAGWR_BETAS_NAMES::beta_nc + "_pred"] = betas[FDAGWR_BETAS_NAMES::beta_nc];

    return l;
}





/*!
* @brief
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
                int n_intervals_trapezoidal_quadrature = 100,
                double target_error_trapezoidal_quadrature = 1e-3,
                int max_iterations_trapezoidal_quadrature = 100,
                Rcpp::Nullable<int> num_threads = R_NilValue,
                std::string basis_type_y_points = "bsplines",
                std::string basis_type_rec_weights_y_points = "bsplines",
                Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue)
{
    //funzione per il multi-source gwr
    //  !!!!!!!! NB: l'ordine delle basi su c++ corrisponde al degree su R !!!!!
    Rcout << "Functional Geographically Weighted Regression" << std::endl;


    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'

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
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_trapezoidal_quadrature(n_intervals_trapezoidal_quadrature);
    // TARGET ERROR WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    double target_error = wrap_and_check_target_error_trapezoidal_quadrature(target_error_trapezoidal_quadrature);
    // MAXIMUM NUMBER OF ITERATIONS WHILE INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int max_iterations = wrap_and_check_max_iterations_trapezoidal_quadrature(max_iterations_trapezoidal_quadrature);


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
    std::size_t number_basis_response_ = number_and_degree_basis_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t degree_basis_response_ = number_and_degree_basis_response_[FDAGWR_FEATS::degree_basis_string];
    check_dim_input<_RESPONSE_>(number_basis_response_,coefficients_response_.rows(),"response coefficients matrix rows");
    check_dim_input<_RESPONSE_>(number_of_statistical_units_,coefficients_response_.cols(),"response coefficients matrix columns");     
    //response reconstruction weights
    auto number_and_degree_basis_rec_weights_response_ = wrap_and_check_basis_number_and_degree<_REC_WEIGHTS_>(n_basis_rec_weights_y_points,degree_basis_rec_weights_y_points,knots_response_.size(),basis_type_rec_weights_response_);
    std::size_t number_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t degree_basis_rec_weights_response_ = number_and_degree_basis_rec_weights_response_[FDAGWR_FEATS::degree_basis_string];
    check_dim_input<_REC_WEIGHTS_>(number_basis_rec_weights_response_,coefficients_rec_weights_response_.rows(),"response reconstruction weights coefficients matrix rows");
    check_dim_input<_REC_WEIGHTS_>(number_of_statistical_units_,coefficients_rec_weights_response_.cols(),"response reconstruction weights coefficients matrix columns");     
    //stationary cov
    auto number_and_degree_basis_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_stationary_cov,degrees_basis_stationary_cov,knots_stationary_cov_.size(),q_C,basis_types_stationary_cov_);
    std::vector<std::size_t> number_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_stationary_cov_ = number_and_degree_basis_stationary_cov_[FDAGWR_FEATS::degree_basis_string];
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_[i].rows(),"covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_stationary_cov_[i].cols(),"covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stationary cov
    auto number_and_degree_basis_beta_stationary_cov_ = wrap_and_check_basis_number_and_degree<_STATIONARY_>(n_basis_beta_stationary_cov,degrees_basis_beta_stationary_cov,knots_beta_stationary_cov_.size(),q_C,basis_types_beta_stationary_cov_);
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = number_and_degree_basis_beta_stationary_cov_[FDAGWR_FEATS::degree_basis_string];


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
                                                                                    target_error,
                                                                                    max_iterations,
                                                                                    abscissa_points_,
                                                                                    number_of_statistical_units_,
                                                                                    number_threads);

    //computing the b
    fgwr_algo->compute();
    //evaluating the betas   
    fgwr_algo->evalBetas();

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
    //regression model used 
    l[FDAGWR_HELPERS_for_PRED_NAMES::model_name] = algo_type<_FGWR_ALGO_>();
    //stationary covariate basis expansion coefficients for beta_c
    l[FDAGWR_B_NAMES::bc]  = b_coefficients[FDAGWR_B_NAMES::bc];
    //beta_c
    l[FDAGWR_BETAS_NAMES::beta_c] = betas[FDAGWR_BETAS_NAMES::beta_c];

    //returning all the elements needed to perform prediction
    Rcpp::List elem_for_pred;
    Rcpp::List inputs_info;
    //input of y
    Rcpp::List response_input;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_type_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_response_;
    response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_response_;
    inputs_info[covariate_type<FDAGWR_COVARIATES_TYPES::RESPONSE>()] = response_input;
    //input of C
    Rcpp::List C_input;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::q] = q_C;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_stationary_cov_;
    C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_stationary_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<FDAGWR_COVARIATES_TYPES::STATIONARY>()] = C_input;
    //input of Beta C   
    Rcpp::List beta_C_input;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis]  = number_basis_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t] = basis_types_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]  = degree_basis_beta_stationary_cov_;
    beta_C_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = knots_beta_stationary_cov_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<FDAGWR_COVARIATES_TYPES::STATIONARY>()] = beta_C_input;
    //domain
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::n] = number_of_statistical_units_;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::a] = a;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::b] = b;
    inputs_info[FDAGWR_HELPERS_for_PRED_NAMES::abscissa] = abscissa_points_;
    //adding all the elements to perform prediction
    elem_for_pred[FDAGWR_HELPERS_for_PRED_NAMES::inputs_info] = inputs_info;
    l[FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred] = elem_for_pred;

    return l;
}


//
// [[Rcpp::export]]
Rcpp::List predict_FWR(Rcpp::List coeff_stationary_cov_to_pred,
                        int units_to_be_predicted,
                        Rcpp::NumericVector abscissa_ev,
                        Rcpp::List model_fitted,
                        int n_knots_smoothing_pred = 100,    
                        Rcpp::Nullable<int> num_threads = R_NilValue)
{
    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'


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
    //list with the fitted model
    Rcpp::List fitted_model      = model_fitted[FDAGWR_HELPERS_for_PRED_NAMES::elem_for_pred];
    //lists with the input of the training
    Rcpp::List training_input    = fitted_model[FDAGWR_HELPERS_for_PRED_NAMES::inputs_info];
    //list with elements of the response
    Rcpp::List response_input            = training_input[covariate_type<_RESPONSE_>()];
    //list with elements of stationary covariates
    Rcpp::List stationary_cov_input      = training_input[FDAGWR_HELPERS_for_PRED_NAMES::cov + covariate_type<_STATIONARY_>()];
    //list with elements of the beta of stationary covariates
    Rcpp::List beta_stationary_cov_input = training_input[FDAGWR_HELPERS_for_PRED_NAMES::beta + covariate_type<_STATIONARY_>()];

    //DOMAIN INFORMATION
    std::size_t n_train = training_input[FDAGWR_HELPERS_for_PRED_NAMES::n];
    _FD_INPUT_TYPE_ a   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::a];
    _FD_INPUT_TYPE_ b   = training_input[FDAGWR_HELPERS_for_PRED_NAMES::b];
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ev_ = wrap_abscissas(abscissa_ev,a,b);                         //abscissa points for which the evaluation of the prediction is required
    std::vector<_FD_INPUT_TYPE_> abscissa_points_ = training_input[FDAGWR_HELPERS_for_PRED_NAMES::abscissa];    //abscissa point for which the training data are discretized
    //knots for performing smoothing of the prediction(n_knots_smoothing_y_new knots equally spaced in (a,b))
    FDAGWR_TRAITS::Dense_Matrix knots_smoothing_pred = FDAGWR_TRAITS::Dense_Vector::LinSpaced(n_knots_smoothing_y_new, a, b);
    //RESPONSE
    std::size_t number_basis_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::string basis_type_response_   = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::size_t degree_basis_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_response_ = response_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    //basis used for doing prediction basis expansion are the same used to smooth the response of the training data
    std::unique_ptr<basis_base_class<_DOMAIN_>> basis_pred = basis_fac.create(basis_type_response_,knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    //STATIONARY COV        
    std::size_t q_C                                       = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::q];
    std::vector<std::size_t> number_basis_stationary_cov_ = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_stationary_cov_  = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_stationary_cov_ = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_stationary_cov_ = stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size()); 
    //STATIONARY BETAS
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::n_basis];
    std::vector<std::string> basis_types_beta_stationary_cov_  = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_t];
    std::vector<std::size_t> degree_basis_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_deg];
    std::vector<FDAGWR_TRAITS::fd_obj_x_type> knots_beta_stationary_cov_ = beta_stationary_cov_input[FDAGWR_HELPERS_for_PRED_NAMES::basis_knots];
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_       = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //saving the betas basis expansion coefficients for stationary covariates
    std::vector<FDAGWR_TRAITS::Dense_Matrix> Bc;
    Bc.reserve(q_C);
    Rcpp::List Bc_list = model_fitted[FDAGWR_B_NAMES::bc];
    for(std::size_t i = 0; i < q_C; ++i){
        Rcpp::List Bc_i_list = Bc_list[i];
        auto Bc_i = reader_data<_DATA_TYPE_,_NAN_REM_>(Bc_i_list[FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis]);  //sono tutte Lc_jx1
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
    std::map<std::string,functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_>> X_new = {{covariate_type<_STATIONARY_>(),Xc_new}};



    //fgwr predictor
    auto fwr_predictor = fwr_predictor_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(Bc),
                                                                                                   std::move(omega),
                                                                                                   q_C,
                                                                                                   Lc,
                                                                                                   Lc_j,
                                                                                                   a,
                                                                                                   b,
                                                                                                   1,
                                                                                                   1.0,
                                                                                                   1,
                                                                                                   n_train,
                                                                                                   number_threads);

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
    l[FDAGWR_HELPERS_for_PRED_NAMES::model_name + "_predictor"] = "predictor_" + algo_type<_FGWR_ALGO_>();
    //predictions
    l[FDAGWR_HELPERS_for_PRED_NAMES::pred] = y_pred_ev_R;
    //stationary covariate basis expansion coefficients for beta_c
    l[FDAGWR_B_NAMES::bc + "_pred"]  = b_coefficients[FDAGWR_B_NAMES::bc];
    //beta_c
    l[FDAGWR_BETAS_NAMES::beta_c + "_pred"] = betas[FDAGWR_BETAS_NAMES::beta_c];

    return l;
}
