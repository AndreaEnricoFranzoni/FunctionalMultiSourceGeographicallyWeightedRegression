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
// OUT OF OR IN CONNECTION WITH PPCKO OR THE USE OR OTHER DEALINGS IN
// fdagwr.


#include <RcppEigen.h>


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include "concepts_fdagwr.hpp"
#include "utility_fdagwr.hpp"


#include "data_reader.hpp"
#include "parameters_wrapper_fdagwr.hpp"


#include "basis_include.hpp"
#include "basis_bspline_systems.hpp"
#include "basis_factory_proxy.hpp"

#include "functional_data.hpp"
#include "functional_data_covariates.hpp"
#include "functional_weight_matrix_stat.hpp"
#include "functional_weight_matrix_no_stat.hpp"
#include "distance_matrix.hpp"
#include "penalization_matrix.hpp"

#include "functional_data_integration.hpp"
#include "fgwr_factory.hpp"



#include "functional_matrix.hpp"
#include "functional_matrix_sparse.hpp"
#include "functional_matrix_diagonal.hpp"
#include "functional_matrix_operators.hpp"
#include "functional_matrix_product.hpp"
#include "functional_matrix_into_wrapper.hpp"




using namespace Rcpp;

//
// [[Rcpp::depends(RcppEigen)]]



/*!
* @brief Function to perform Functional Multi-Source Geographically Weighted Regression.
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
* @param n_intervals_trapezoidal_quadrature number of intervals used while performing integration via trapezoidal quadrature rule
* @param num_threads number of threads to be used in OMP parallel directives. Default: maximum number of cores available in the machine running fmsgwr.
* @param basis_type_y_points string containing the type of basis used for the (functional) response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_type_rec_weights_y_points string containing the type of basis used for the weights to reconstruct the (functional) response basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stationary_cov vector of strings, element i-th containing the type of basis used for the i-th stationary covariates (functional regression) coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_events_cov vector of strings, element i-th containing the type of basis used for the i-th events-dependent covariates (functional regression) coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @param basis_types_beta_stations_cov vector of strings, element i-th containing the type of basis used for the i-th stations-dependent covariates (functional regression) coefficients basis expansion. Possible values: "bsplines", "constant". Defalut: "bsplines".
* @return an R list containing:
* -
* 
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
Rcpp::List FMSGWR(Rcpp::NumericMatrix y_points,
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
                  int n_intervals_trapezoidal_quadrature = 100,
                  Rcpp::Nullable<int> num_threads = R_NilValue,
                  std::string basis_type_y_points = "bsplines",
                  std::string basis_type_rec_weights_y_points = "bsplines",
                  Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stationary_cov = R_NilValue,
                  Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_events_cov = R_NilValue,
                  Rcpp::Nullable<Rcpp::CharacterVector> basis_types_beta_stations_cov = R_NilValue)
{
    //funzione per il multi-source gwr
    //  !!!!!!!! NB: l'ordine delle basi su c++ corrisponde al degree su R !!!!!


    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)

    //SOLO PER LE COORDINATE OGNI RIGA E' UN'UNITA'


    Rcout << "fdagwr.9: " << std::endl;

    using _DATA_TYPE_ = double;                                                     //data type
    using _FD_INPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_x_type;                           //data type for the abscissa of fdata (double)
    using _FD_OUTPUT_TYPE_ = FDAGWR_TRAITS::fd_obj_y_type;                          //data type for the image of fdata (double)
    using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry;                                 //domain geometry
    constexpr auto _FGWR_ALGO_ = FDAGWR_ALGO::GWR_FMS_ESC;                          //fgwr type (estimating stationary -> station-dependent -> event-dependent)
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
    // NUMBER OF INTERVALS FOR INTEGRATING VIA TRAPEZOIDAL QUADRATURE RULE
    int n_intervals = wrap_and_check_n_intervals_trapezoidal_quadrature(n_intervals_trapezoidal_quadrature);


    //  RESPONSE
    //raw data
    auto response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(y_points);       //Eigen dense matrix type (auto is necessary )
    //number of statistical units
    std::size_t number_of_statistical_units_ = response_.cols();
    //coefficients matrix
    auto coefficients_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_y_points);
    //reconstruction weights coefficients matrix
    auto coefficients_rec_weights_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_rec_weights_y_points);


    //  ABSCISSA POINTS of response
    std::vector<double> abscissa_points_ = wrap_abscissas(t_points,left_extreme_domain,right_extreme_domain);
    check_dim_input<_RESPONSE_>(response_.rows(), abscissa_points_.size(), "points for evaluation of raw data vector");   //check that size of abscissa points and number of evaluations of fd raw data coincide
    FDAGWR_TRAITS::Dense_Vector abscissa_points_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(abscissa_points_.data(),abscissa_points_.size());
    double a = left_extreme_domain;
    double b = right_extreme_domain;


    //  KNOTS
    //response
    std::vector<double> knots_response_ = wrap_abscissas(knots_y_points,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_response_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_response_.data(),knots_response_.size());
    //stationary cov
    std::vector<double> knots_stationary_cov_ = wrap_abscissas(knots_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    //beta stationary cov
    std::vector<double> knots_beta_stationary_cov_ = wrap_abscissas(knots_beta_stationary_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_stationary_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //events cov
    std::vector<double> knots_events_cov_ = wrap_abscissas(knots_events_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_events_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    //beta events cov
    std::vector<double> knots_beta_events_cov_ = wrap_abscissas(knots_beta_events_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_events_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size());
    //stations cov
    std::vector<double> knots_stations_cov_ = wrap_abscissas(knots_stations_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_stations_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    //stations beta cov
    std::vector<double> knots_beta_stations_cov_ = wrap_abscissas(knots_beta_stations_cov,a,b);
    FDAGWR_TRAITS::Dense_Vector knots_beta_stations_cov_eigen_w_ = Eigen::Map<FDAGWR_TRAITS::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());


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
    check_dim_input<_EVENT_>(number_of_statistical_units_,coordinates_events_.rows(),"coordinates matrix rows");
    check_dim_input<_EVENT_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_events_.cols(),"coordinates matrix columns");
    distance_matrix<_DISTANCE_> distances_events_cov_(std::move(coordinates_events_),number_threads);
    //stations  DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_stations_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_stations);
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
    //events
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_E(std::move(bs_E),lambda_events_cov_);
    //stations
    penalization_matrix<_DERVIATIVE_PENALIZED_> R_S(std::move(bs_S),lambda_stations_cov_);


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


   
    /*
   //TESTING ETs WITHIN FUNCTIONS
   double el = 1.0;

    std::function<double(double const &)> f1 = [](const double &x){return std::pow(x,2);};
    std::function<double(double const &)> f2 = [](const double &x){return std::pow(x,3);};
    std::vector<std::function<double(double const &)> > test_f1_f2{f1,f2};
    functional_matrix test_fm_1(test_f1_f2,1,2);

    Rcout << "FM: f1(2): " << test_fm_1(0,0)(el) << ", f2(2): " << test_fm_1(0,1)(el) << std::endl;


    std::function<double(double const &)> f3 = [](const double &x){return 1.0 + 2.0*x;};
    std::function<double(double const &)> f4 = [](const double &x){return 3.0 - x;};
    std::vector<std::function<double(double const &)> > test_f3_f4{f3,f4};
    functional_matrix test_fm_2(test_f3_f4,1,2);

    Rcout << "FM: f3(2): " << test_fm_2(0,0)(el) << ", f4(2): " << test_fm_2(0,1)(el) << std::endl;


    functional_matrix test_op = test_fm_1+test_fm_2;
    functional_matrix test_op2 = log(test_op);
    test_op = 5.0*(test_op+test_op2)*2.0;
    Rcout << "FM op: primo: " << test_op(0,0)(el) << ", secondo: " << test_op(0,1)(el) << std::endl;
    //END TESTING ETs WITHIN FUNCTIONS
    */
    



    ///////////////////////////////
    /////    FGWR ALGORITHM   /////
    ///////////////////////////////
    //wrapping all the functional elements in a functional_matrix

    //y: a column vector of dimension nx1
    functional_matrix<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> y = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_,number_threads);
    //phi: a sparse functional matrix nx(n*L), where L is the number of basis for the response
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> phi = wrap_into_fm<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_,_DOMAIN_,response_basis_tmp_t::template_type>(y_fd_.fdata_basis(),number_of_statistical_units_,number_basis_response_);
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
    auto fgwr_algo = fgwr_factory< _FGWR_ALGO_, _FD_INPUT_TYPE_, _FD_OUTPUT_TYPE_ >(std::move(y),
                                                                                    std::move(phi),
                                                                                    std::move(coefficients_response_),
                                                                                    std::move(Xc),
                                                                                    std::move(Wc),
                                                                                    std::move(R_C.PenalizationMatrix()),
                                                                                    std::move(omega),
                                                                                    std::move(Xe),
                                                                                    std::move(We),
                                                                                    std::move(R_E.PenalizationMatrix()),
                                                                                    std::move(theta),
                                                                                    std::move(Xs),
                                                                                    std::move(Ws),
                                                                                    std::move(R_S.PenalizationMatrix()),
                                                                                    std::move(psi),
                                                                                    a,
                                                                                    b,
                                                                                    n_intervals,
                                                                                    number_threads);
    fgwr_algo->compute();

    double loc = 0.3;
    std::size_t n_rows_test = 3;
    std::size_t n_cols_test = 2;
    std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)> f1 = [](const double & x){return x;};
    std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)> f2 = [](const double & x){return x + 4;};
    std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)> f3 = [](const double & x){return x*x;};
    std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)> f4 = [](const double & x){return x-1;};
    std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)> f5 = [](const double & x){return 5;};

    std::vector<std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)>> test{f1,f2,f3,f4,f5,f3};
    functional_matrix test_fdm_dense(test,3,2);
    std::vector<std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)>> test2{f5,f3,f1,f5,f2,f4};
    functional_matrix test_fdm_dense2(test2,2,3);
    std::vector<std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)>> test3{f5,f3,f1,f5,f2,f4,f1,f5,f2,f1,f5,f4};
    functional_matrix test_fdm_dense3(test3,4,3);
    std::vector<std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)>> test4{f5,f3,f1,f5,f2,f4,f1,f5,f2};
    functional_matrix test_fdm_dense4(test4,3,3);
    std::vector<std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)>> test5{f2,f4,f1,f5,f3,f5,f1,f3,f5};
    functional_matrix test_fdm_dense5(test5,3,3);
    std::vector<std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)>> testd{f5,f3,f1,f2};
    functional_matrix_diagonal test_fdm_d(testd,4);
    Eigen::MatrixXd M2 = Eigen::MatrixXd::Random(3,2);  // valori in [-1, 1]
    auto test_fdm_dense2_t = test_fdm_dense2.transpose();


    std::vector<std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)>> test_sm_v{f1,f2,f3,f4};
    std::vector<std::size_t> row_idx{0,0,0,2};
    std::vector<std::size_t> col_idx{0,1,2,2,4};
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> test_sm(test_sm_v,3,4,row_idx,col_idx);
    //Rcout << "Trasposto di una completa" << std::endl;
    auto test_sm_t = test_sm.transpose();



    //vettore colonna di matrice sparsa
    std::vector<std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)>> test_sm_v_cv{f1,f2,f3};
    std::vector<std::size_t> row_idx_cv{1,5,8};
    std::vector<std::size_t> col_idx_cv{0,3};
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> test_sm_cv(test_sm_v_cv,10,1,row_idx_cv,col_idx_cv);
    //Rcout << "Trasposto di un vettore colonna" << std::endl;
    auto test_sm_cv_t = test_sm_cv.transpose();


    Rcout << "Vettore colonna" << std::endl;
    for(std::size_t i = 0; i < test_sm_cv.rows(); ++i){
        for(std::size_t j = 0; j < test_sm_cv.cols(); ++j){
            Rcout << "Elem of CV (sparse) (" << i << "," << j << ") evaluated in " << loc << ": " << test_sm_cv(i,j)(loc) << std::endl;}}
    Rcout << "CV original: rows: " << test_sm_cv.rows() << ", cols: " << test_sm_cv.cols() << std::endl;
    Rcout << "CV rows idx " << std::endl;
    for(std::size_t i = 0; i < test_sm_cv.rows_idx().size(); ++i){Rcout << test_sm_cv.rows_idx()[i] << std::endl;}
    Rcout << "CV cols idx " << std::endl;
    for(std::size_t i = 0; i < test_sm_cv.cols_idx().size(); ++i){Rcout << test_sm_cv.cols_idx()[i] << std::endl;}

    Rcout << "Trasposto di un vettore colonna" << std::endl;
    for(std::size_t i = 0; i < test_sm_cv_t.rows(); ++i){
        for(std::size_t j = 0; j < test_sm_cv_t.cols(); ++j){
            Rcout << "Elem of CV T (" << i << "," << j << ") evaluated in " << loc << ": " << test_sm_cv_t(i,j)(loc) << std::endl;}}
    Rcout << "CV T: rows: " << test_sm_cv_t.rows() << ", cols: " << test_sm_cv_t.cols() << std::endl;
    Rcout << "CV T rows idx" << std::endl;
    for(std::size_t i = 0; i < test_sm_cv_t.rows_idx().size(); ++i){Rcout << test_sm_cv_t.rows_idx()[i] << std::endl;}
    Rcout << "CV T cols idx" << std::endl;
    for(std::size_t i = 0; i < test_sm_cv_t.size(); ++i){Rcout << test_sm_cv_t.cols_idx()[i] << std::endl;}






    //vettore riga di matrice sparsa
    std::vector<std::function<_FD_OUTPUT_TYPE_(const _FD_INPUT_TYPE_ &)>> test_sm_v_rv{f1,f2,f3};
    std::vector<std::size_t> row_idx_rv{0,0,0};
    std::vector<std::size_t> col_idx_rv{0,0,0,1,1,2,2,3};
    functional_matrix_sparse<_FD_INPUT_TYPE_,_FD_OUTPUT_TYPE_> test_sm_rv(test_sm_v_rv,1,7,row_idx_rv,col_idx_rv);
    auto test_sm_rv_t = test_sm_rv.transpose();

/*
    Rcout << "Vettore riga" << std::endl;
    for(std::size_t i = 0; i < test_sm_rv.rows(); ++i){
        for(std::size_t j = 0; j < test_sm_rv.cols(); ++j){
            Rcout << "Elem of RV (sparse) (" << i << "," << j << ") evaluated in " << loc << ": " << test_sm_rv(i,j)(loc) << std::endl;}}
    Rcout << "RV original: rows: " << test_sm_rv.rows() << ", cols: " << test_sm_rv.cols() << std::endl;
    Rcout << "RV rows idx " << std::endl;
    for(std::size_t i = 0; i < test_sm_rv.rows_idx().size(); ++i){Rcout << test_sm_rv.rows_idx()[i] << std::endl;}
    Rcout << "RV cols idx " << std::endl;
    for(std::size_t i = 0; i < test_sm_rv.cols_idx().size(); ++i){Rcout << test_sm_rv.cols_idx()[i] << std::endl;}

    Rcout << "Trasposto di un vettore riga" << std::endl;
    for(std::size_t i = 0; i < test_sm_rv_t.rows(); ++i){
        for(std::size_t j = 0; j < test_sm_rv_t.cols(); ++j){
            Rcout << "Elem of RV T (" << i << "," << j << ") evaluated in " << loc << ": " << test_sm_rv_t(i,j)(loc) << std::endl;}}
    Rcout << "RV T: rows: " << test_sm_rv_t.rows() << ", cols: " << test_sm_rv_t.cols() << std::endl;
    Rcout << "RV T rows idx" << std::endl;
    for(std::size_t i = 0; i < test_sm_rv_t.rows_idx().size(); ++i){Rcout << test_sm_rv_t.rows_idx()[i] << std::endl;}
    Rcout << "RV T cols idx" << std::endl;
    for(std::size_t i = 0; i < test_sm_rv_t.size(); ++i){Rcout << test_sm_rv_t.cols_idx()[i] << std::endl;}
*/







    //returning element
    Rcpp::List l;

    l["Type of gwr"] = "fmsgwr";

    return l;
}




//
// [[Rcpp::export]]
Rcpp::List FSGWR(double input_el = 1,
                 Rcpp::Nullable<int> num_threads = R_NilValue){
    //funzione per il source gwr

    //checking and wrapping input parameters
    int number_threads = wrap_num_thread(num_threads);

    //returning element
    Rcpp::List l;

    l["Type of gwr"] = "fsgwr";
    return l;
}


//
// [[Rcpp::export]]
Rcpp::List FGWR(double input_el=1,
                Rcpp::Nullable<int> num_threads = R_NilValue)
{
    //funzione per il gwr

    //checking and wrapping input parameters
    int number_threads = wrap_num_thread(num_threads);

    Rcout << "NT: " << number_threads << std::endl;

    //CHECK PARAMETER WRAPPING: BEBUGGING PURPOSES
    /*
    Rcout << "Number of statistical units: " << number_of_statistical_units_ << std::endl;
    Rcout << "Response " << ", type of basis: " << basis_type_response_ << ", basis degree: " << degree_basis_response_ << ", basis number: " << number_basis_response_ << std::endl;
    Rcout << "Response rec w" << ", type of basis: " << basis_type_rec_weights_response_ << ", basis degree: " << degree_basis_rec_weights_response_ << ", basis number: " << number_basis_rec_weights_response_ << std::endl;
    Rcout << "********" << std::endl;
    Rcout << "Stationary covs: " << q_C << std::endl;
    for(std::size_t i = 0; i < q_C; ++i)
    {
        Rcout << "Covariate " << i+1 << ", " << names_stationary_cov_[i] << ", type of basis: " << basis_types_stationary_cov_[i] << ", basis degree: " << degree_basis_stationary_cov_[i] << ", basis number: " << number_basis_stationary_cov_[i] << std::endl;
        Rcout << "Covariate regressor " << i+1 << ", type of basis: " << basis_types_beta_stationary_cov_[i] << ", basis degree: " << degree_basis_beta_stationary_cov_[i] << ", basis number: " << number_basis_beta_stationary_cov_[i] << std::endl;
    }
    Rcout << "********" << std::endl;
    Rcout << "Events covs: " << q_E << std::endl;
    for(std::size_t i = 0; i < q_E; ++i)
    {
        Rcout << "Covariate " << i+1 << ", " << names_events_cov_[i] << ", type of basis: " << basis_types_events_cov_[i] << ", basis degree: " << degree_basis_events_cov_[i] << ", basis number: " << number_basis_events_cov_[i] << std::endl;
        Rcout << "Covariate regressor " << i+1 << ", type of basis: " << basis_types_beta_events_cov_[i] << ", basis degree: " << degree_basis_beta_events_cov_[i] << ", basis number: " << number_basis_beta_events_cov_[i] << std::endl;
    }
    Rcout << "********" << std::endl;
    Rcout << "Stations covs: " << q_S << std::endl;
    for(std::size_t i = 0; i < q_S; ++i)
    {
        Rcout << "Covariate " << i+1 << ", " << names_stations_cov_[i] << ", type of basis: " << basis_types_stations_cov_[i] << ", basis degree: " << degree_basis_stations_cov_[i] << ", basis number: " << number_basis_stations_cov_[i] << std::endl;
        Rcout << "Covariate regressor " << i+1 << ", type of basis: " << basis_types_beta_stations_cov_[i] << ", basis degree: " << degree_basis_beta_stations_cov_[i] << ", basis number: " << number_basis_beta_stations_cov_[i] << std::endl;
    }
    */




    //returning element
    Rcpp::List l;

    l["Type of gwr"] = "fgwr";

    return l;
}