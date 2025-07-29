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


#include "traits_fdagwr.hpp"
#include "data_reader.hpp"
#include "parameters_wrapper_fdagwr.hpp"

#include "basis_include.hpp"
#include "basis_bspline_systems.hpp"


#include "functional_data.hpp"

#include "functional_weight_matrix_stat.hpp"
#include "functional_weight_matrix_no_stat.hpp"

#include "distance_matrix.hpp"
#include "penalization_matrix.hpp"

#include "test_basis_eval.hpp"





using namespace Rcpp;

//
// [[Rcpp::depends(RcppEigen)]]




//
// [[Rcpp::export]]
void fdagwr_test_function(std::string input_string) {

    Rcout << "First draft of fdagwr.9: " << input_string << std::endl;
    int test;

    test = test_fda_PDE(5.9);
}


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
Rcpp::List fmsgwr(Rcpp::NumericMatrix y_points,
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


    Rcout << "fdagwr.8: " << std::endl;

    using _DATA_TYPE_ = double;                                                      //data type
    using _DOMAIN_ = fdagwr_traits::Domain;                                          //domain geometry
    //using _BASIS_BETAS_ = bsplines_basis;                                          //basis for the betas
    constexpr auto _RESPONSE_ = FDAGWR_COVARIATES_TYPES::RESPONSE;                   //enum for the response
    constexpr auto _REC_WEIGHTS_ = FDAGWR_COVARIATES_TYPES::REC_WEIGHTS;             //enum for the response reconstruction weights
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;               //enum for stationary covariates
    constexpr auto _EVENT_ = FDAGWR_COVARIATES_TYPES::EVENT;                         //enum for event covariates
    constexpr auto _STATION_ = FDAGWR_COVARIATES_TYPES::STATION;                     //enum for station covariates
    constexpr auto _DERVIATIVE_PENALIZED_ = PENALIZED_DERIVATIVE::SECOND;            //enum for the penalization
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;                         //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                                 //kernel function to smooth the distances within statistcal units locations
    constexpr auto _NAN_REM_ = REM_NAN::MR;                                          //how to remove nan (with mean of non-nans)


    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);


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
    fdagwr_traits::Dense_Vector abscissa_points_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(abscissa_points_.data(),abscissa_points_.size());
    double a = left_extreme_domain;
    double b = right_extreme_domain;


    //  KNOTS
    //response
    std::vector<double> knots_response_ = wrap_abscissas(knots_y_points,a,b);
    fdagwr_traits::Dense_Vector knots_response_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_response_.data(),knots_response_.size());
    //stationary cov
    std::vector<double> knots_stationary_cov_ = wrap_abscissas(knots_stationary_cov,a,b);
    fdagwr_traits::Dense_Vector knots_stationary_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    //beta stationary cov
    std::vector<double> knots_beta_stationary_cov_ = wrap_abscissas(knots_beta_stationary_cov,a,b);
    fdagwr_traits::Dense_Vector knots_beta_stationary_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //events cov
    std::vector<double> knots_events_cov_ = wrap_abscissas(knots_events_cov,a,b);
    fdagwr_traits::Dense_Vector knots_events_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    //beta events cov
    std::vector<double> knots_beta_events_cov_ = wrap_abscissas(knots_beta_events_cov,a,b);
    fdagwr_traits::Dense_Vector knots_beta_events_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size());
    //stations cov
    std::vector<double> knots_stations_cov_ = wrap_abscissas(knots_stations_cov,a,b);
    fdagwr_traits::Dense_Vector knots_stations_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    //stations beta cov
    std::vector<double> knots_beta_stations_cov_ = wrap_abscissas(knots_beta_stations_cov,a,b);
    fdagwr_traits::Dense_Vector knots_beta_stations_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());


    //  COVARIATES names, coefficients and how many (q_), for every type
    //stationary 
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov);
    std::size_t q_C = names_stationary_cov_.size();    //number of stationary covariates
    std::vector<fdagwr_traits::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov);    
    //events
    std::vector<std::string> names_events_cov_ = wrap_covariates_names<_EVENT_>(coeff_events_cov);
    std::size_t q_E = names_events_cov_.size();        //number of events related covariates
    std::vector<fdagwr_traits::Dense_Matrix> coefficients_events_cov_ = wrap_covariates_coefficients<_EVENT_>(coeff_events_cov);
    //stations
    std::vector<std::string> names_stations_cov_ = wrap_covariates_names<_STATION_>(coeff_stations_cov);
    std::size_t q_S = names_stations_cov_.size();      //number of stations related covariates
    std::vector<fdagwr_traits::Dense_Matrix> coefficients_stations_cov_ = wrap_covariates_coefficients<_STATION_>(coeff_stations_cov);


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



    //COMPUTING DISTANCES
    //events
    distances_events_cov_.compute_distances();
    //stations
    distances_stations_cov_.compute_distances();


    //COMPUTING FUNCTIONAL WEIGHT MATRIX
    //stationary
    functional_weight_matrix_stationary<_STATIONARY_> W_C(coefficients_rec_weights_response_,
                                                          number_threads);
    W_C.compute_weights();                                                      
    //events
    functional_weight_matrix_non_stationary<_EVENT_,_KERNEL_,_DISTANCE_> W_E(coefficients_rec_weights_response_,
                                                                             std::move(distances_events_cov_),
                                                                             kernel_bandwith_events_cov_,
                                                                             number_threads);
    W_E.compute_weights();                                                                         
    //stations
    functional_weight_matrix_non_stationary<_STATION_,_KERNEL_,_DISTANCE_> W_S(coefficients_rec_weights_response_,
                                                                               std::move(distances_stations_cov_),
                                                                               kernel_bandwith_stations_cov_,
                                                                               number_threads);
    W_S.compute_weights();


    //COMPUTING THE BASIS SYSTEMS FOR THE BETAS
    //stationary
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                  degree_basis_beta_stationary_cov_, 
                                                  number_basis_beta_stationary_cov_, 
                                                  q_C);
    //events
    basis_systems< _DOMAIN_, bsplines_basis > bs_E(knots_beta_events_cov_eigen_w_, 
                                                  degree_basis_beta_events_cov_, 
                                                  number_basis_beta_events_cov_, 
                                                  q_E);
    //stations
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


    //FD OBJECTS
    //response
    Rcout << "Response: coeff rows: " << coefficients_response_.rows() << ", cols: " << coefficients_response_.cols() << ", nb: " << number_basis_response_ << std::endl;
    bsplines_basis<_DOMAIN_> basis_response_(knots_response_eigen_w_,degree_basis_response_,number_basis_response_);
    functional_data<_DOMAIN_,bsplines_basis > fd_response_(std::move(coefficients_response_),basis_response_);
    //decltype(basis_response_)
    //constant_basis<_DOMAIN_> basis_response_(knots_response_eigen_w_);
    //functional_data<_DOMAIN_,constant_basis > fd_response_(std::move(coefficients_response_),basis_response_);

    double el = 0.0;
    Rcout << "Eval basis pre in " << el << ": " << basis_response_.eval_base(el) << std::endl;
    
    for(std::size_t i = 0; i < fd_response_.n(); ++i){
        Rcout << "Eval unit " << i+1 << " in loc " << el << ": " << fd_response_.eval(el,i) << std::endl;
        Rcout << "Eval unit " << i+1 << " basis in loc " << el << ": " << fd_response_.fdata()[i].fdatum_basis().eval_base(el) << std::endl;
    }

    double el1 = -1.0;
    Rcout << "Eval basis pre in" << el1 << ": " << basis_response_.eval_base(el1) << std::endl;
    for(std::size_t i = 0; i < fd_response_.n(); ++i){
        Rcout << "Eval unit " << i+1 << " in loc " << el1 << ": " << fd_response_.eval(el1,i) << std::endl;
        Rcout << "Eval unit " << i+1 << " basis in loc " << el1 << ": " << fd_response_.fdata()[i].fdatum_basis().eval_base(el1) << std::endl;
    }
    


    //returning element
    Rcpp::List l;

    l["Type of gwr"] = "fmsgwr";

    return l;
}



//
// [[Rcpp::export]]
Rcpp::List test_distance_matrix(Rcpp::NumericMatrix coordinates,
                                Rcpp::Nullable<int> num_threads = R_NilValue)
{
    using T = double;

    auto coordinates_ = reader_data<T,REM_NAN::MR>(coordinates);
    std::size_t n_stat_units = coordinates_.rows();
    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);
     

    Rcpp::List l;
    l["Distanze"] = "";
    return l;
}


//
// [[Rcpp::export]]
Rcpp::List fsgwr(double input_el = 1,
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
Rcpp::List fgwr(double input_el=1,
                Rcpp::Nullable<int> num_threads = R_NilValue){
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