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
* @brief Function to perform fms-gwr
* @param y_points Rcpp::NumericMatrix (matrix of double) containing the raw (functional) response: each row represents a specific abscissa, each column a statistical units
* @param t_points Rcpp::NumericVector with the abscissa points with respect which the raw evaluations of y_points are available (length(t_points)==number of row (y_points))
* @param left_extreme_domain double indicating the left extreme of the domain of the response curve (not necessarily the smaller element in t_points)
* @param right_extreme_domain double indicating the right extreme of the domain of the response curve (not necessarily the biggest element in t_points)
* @param coeff_y_points Rcpp::NumericMatrix (matrix of double) containing the coefficient of the response's basis expansion: each row represents a specific basis, each column a statistical units
* @param knots_y_points Rcpp::NumericVector with the abscissa points with respect which the basis expansion of the y_points is performed
* @param n_order_basis_y_points integer: the degree of the bsplines used for the (functional) response
* @param n_basis_y_points integer: number of basis for the basis expansion of the (functional) response. It has to be the number of rows of coeff_y_points
* @param coeff_rec_weights_y_points Rcpp::NumericMatrix (matrix of double) containing the coefficients of the weights to reconstruct the (functional) response: each row represents a specific basis, each column a statistical units.
*                                   The above-mentioned basis expansion is performed with respect to knots_y_points 
* @param n_order_basis_rec_weights_y_points integer: the degree of the bsplines used for the reconstruction of the (functional) response
* @param n_basis_rec_weights_y_points integer: number of bsplines basis used for the reconstruction of the (functional) response
* @param coeff_stationary_cov list: each element is a Rcpp::NumericMatrix containing the coefficient for the basis expansion of the i-th stationary covariate: each row represents a specific basis, each column a statistical units.
*                             The name of the i-th element is the name of the i-th stationary covariate (default: "reg.Ci")
* @param 
* @return an R list containing:
* - 
* @note all the basis expansion have to be made with respect to bsplines o constant basis. 
*       For the bsplines basis, number of knots = number of basis - order(degree of pols) + 1
*/
//
// [[Rcpp::export]]
Rcpp::List fmsgwr(Rcpp::NumericMatrix y_points,
                  Rcpp::NumericVector t_points,
                  double left_extreme_domain,
                  double right_extreme_domain,
                  Rcpp::NumericMatrix coeff_y_points,
                  Rcpp::NumericVector knots_y_points,
                  Rcpp::Nullable<int> n_order_basis_y_points,
                  Rcpp::Nullable<int> n_basis_y_points,
                  Rcpp::NumericMatrix coeff_rec_weights_y_points,
                  Rcpp::Nullable<int> n_order_basis_rec_weights_y_points,
                  Rcpp::Nullable<int> n_basis_rec_weights_y_points,
                  Rcpp::List coeff_stationary_cov,
                  Rcpp::CharacterVector basis_types_stationary_cov,
                  Rcpp::NumericVector knots_stationary_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_order_basis_stationary_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_basis_stationary_cov,
                  Rcpp::NumericVector penalization_stationary_cov,
                  Rcpp::NumericVector knots_beta_stationary_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_order_basis_beta_stationary_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_stationary_cov,
                  Rcpp::List coeff_events_cov,
                  Rcpp::CharacterVector basis_types_events_cov,
                  Rcpp::NumericVector knots_events_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_order_basis_events_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_basis_events_cov,
                  Rcpp::NumericVector penalization_events_cov,
                  Rcpp::NumericMatrix coordinates_events,
                  double bandwith_events,
                  Rcpp::NumericVector knots_beta_events_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_order_basis_beta_events_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_events_cov,
                  Rcpp::List coeff_stations_cov,
                  Rcpp::CharacterVector basis_types_stations_cov,
                  Rcpp::NumericVector knots_stations_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_order_basis_stations_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_basis_stations_cov,
                  Rcpp::NumericVector penalization_stations_cov,
                  Rcpp::NumericMatrix coordinates_stations,
                  double bandwith_stations,
                  Rcpp::NumericVector knots_beta_stations_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_order_basis_beta_stations_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_stations_cov,
                  Rcpp::Nullable<int> num_threads = R_NilValue)
{
    //funzione per il multi-source gwr
    //  !!!!!!!! NB: l'ordine delle basi su c++ corrisponde al degree su R !!!!!


    //COME VENGONO PASSATE LE COSE: OGNI COLONNA E' UN'UNITA', OGNI RIGA UNA VALUTAZIONE FUNZIONALE/COEFFICIENTE DI BASE 
    //  (ANCHE PER LE COVARIATE DELLO STESSO TIPO, PUO' ESSERCI UN NUMERO DI BASI DIFFERENTE)


    Rcout << "fdagwr.1: " << std::endl;

    using _DATA_TYPE_ = double;                                                      //data type
    using _DOMAIN_ = fdagwr_traits::Domain;                                          //domain geometry
    //using _BASIS_BETAS_ = bsplines_basis;                                            //basis for the betas
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
    std::size_t number_of_statistical_units_ = response_.cols();
    //coefficients
    auto coefficients_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_y_points);
    //reconstruction weights
    auto coefficients_response_reconstruction_weights_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_rec_weights_y_points);
    

    //  ABSCISSA POINTS of response
    std::vector<double> abscissa_points_ = wrap_abscissas(t_points,left_extreme_domain,right_extreme_domain);
    fdagwr_traits::Dense_Vector abscissa_points_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(abscissa_points_.data(),abscissa_points_.size());
    double a = left_extreme_domain;
    double b = right_extreme_domain;


    //  KNOTS
    //response
    std::vector<double> knots_response_ = wrap_abscissas(knots_y_points,left_extreme_domain,right_extreme_domain);
    fdagwr_traits::Dense_Vector knots_response_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_response_.data(),knots_response_.size());
    //stationary cov
    std::vector<double> knots_stationary_cov_ = wrap_abscissas(knots_stationary_cov,left_extreme_domain,right_extreme_domain);
    fdagwr_traits::Dense_Vector knots_stationary_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_stationary_cov_.data(),knots_stationary_cov_.size());
    //beta stationary cov
    std::vector<double> knots_beta_stationary_cov_ = wrap_abscissas(knots_beta_stationary_cov,left_extreme_domain,right_extreme_domain);
    fdagwr_traits::Dense_Vector knots_beta_stationary_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_beta_stationary_cov_.data(),knots_beta_stationary_cov_.size());
    //events cov
    std::vector<double> knots_events_cov_ = wrap_abscissas(knots_events_cov,left_extreme_domain,right_extreme_domain);
    fdagwr_traits::Dense_Vector knots_events_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_events_cov_.data(),knots_events_cov_.size());
    //beta events cov
    std::vector<double> knots_beta_events_cov_ = wrap_abscissas(knots_beta_events_cov,left_extreme_domain,right_extreme_domain);
    fdagwr_traits::Dense_Vector knots_beta_events_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_beta_events_cov_.data(),knots_beta_events_cov_.size());
    //stations cov
    std::vector<double> knots_stations_cov_ = wrap_abscissas(knots_stations_cov,left_extreme_domain,right_extreme_domain);
    fdagwr_traits::Dense_Vector knots_stations_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_stations_cov_.data(),knots_stations_cov_.size());
    //stations beta cov
    std::vector<double> knots_beta_stations_cov_ = wrap_abscissas(knots_beta_stations_cov,left_extreme_domain,right_extreme_domain);
    fdagwr_traits::Dense_Vector knots_beta_stations_cov_eigen_w_ = Eigen::Map<fdagwr_traits::Dense_Vector>(knots_beta_stations_cov_.data(),knots_beta_stations_cov_.size());


    //  COVARIATES names, coefficients and how many
    //stationary 
    std::vector<std::string> names_stationary_cov_ = wrap_covariates_names<_STATIONARY_>(coeff_stationary_cov);
    std::vector<fdagwr_traits::Dense_Matrix> coefficients_stationary_cov_ = wrap_covariates_coefficients<_STATIONARY_>(coeff_stationary_cov);    
    std::size_t q_C = names_stationary_cov_.size();    //number of stationary covariates
    //events
    std::vector<std::string> names_events_cov_ = wrap_covariates_names<_EVENT_>(coeff_events_cov);
    std::vector<fdagwr_traits::Dense_Matrix> coefficients_events_cov_ = wrap_covariates_coefficients<_EVENT_>(coeff_events_cov);
    std::size_t q_E = names_events_cov_.size();        //number of events related covariates
    //stations
    std::vector<std::string> names_stations_cov_ = wrap_covariates_names<_STATION_>(coeff_stations_cov);
    std::vector<fdagwr_traits::Dense_Matrix> coefficients_stations_cov_ = wrap_covariates_coefficients<_STATION_>(coeff_stations_cov);
    std::size_t q_S = names_stations_cov_.size();      //number of stations related covariates


    //  BASIS TYPES
    //stationary
    std::vector<std::string> basis_types_stationary_cov_ = wrap_basis_type<_STATIONARY_>(basis_types_stationary_cov,q_C);
    //events
    std::vector<std::string> basis_types_events_cov_ = wrap_basis_type<_EVENT_>(basis_types_events_cov,q_E);
    //stations
    std::vector<std::string> basis_types_stations_cov_ = wrap_basis_type<_STATION_>(basis_types_stations_cov,q_S);


    //  NUMBER AND ORDER OF BASIS: checking matrix coefficients dimensions
    //response
    auto number_and_order_basis_response_ = wrap_basis_number_and_order(n_basis_y_points,n_order_basis_y_points,knots_response_.size());
    std::size_t number_basis_response_ = number_and_order_basis_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t order_basis_response_ = number_and_order_basis_response_[FDAGWR_FEATS::order_basis_string];
    check_dim_input<_STATIONARY_>(number_basis_response_,coefficients_response_.rows()," response coefficients matrix rows");
    check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_response_.cols()," response coefficients matrix columns");
    //response reconstruction weights
    auto number_and_order_basis_weights_response_ = wrap_basis_number_and_order(n_basis_rec_weights_y_points,n_order_basis_rec_weights_y_points,knots_response_.size());
    std::size_t number_basis_weights_response_ = number_and_order_basis_weights_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t order_basis_weights_response_ = number_and_order_basis_weights_response_[FDAGWR_FEATS::order_basis_string];
    check_dim_input<_STATIONARY_>(number_basis_weights_response_,coefficients_response_reconstruction_weights_.rows()," response reconstruction coefficients matrix rows");
    check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_response_.cols()," response reconstruction coefficients matrix columns");
    //stationary cov
    auto number_and_order_basis_stationary_cov_ = wrap_basis_numbers_and_orders<_STATIONARY_>(n_basis_stationary_cov,n_order_basis_stationary_cov,knots_stationary_cov_.size(),q_C);
    std::vector<std::size_t> number_basis_stationary_cov_ = number_and_order_basis_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_stationary_cov_ = number_and_order_basis_stationary_cov_[FDAGWR_FEATS::order_basis_string];
    for(std::size_t i = 0; i < q_C; ++i){   
        check_dim_input<_STATIONARY_>(number_basis_stationary_cov_[i],coefficients_stationary_cov_[i].rows()," covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATIONARY_>(number_of_statistical_units_,coefficients_stationary_cov_[i].cols()," covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stationary cov
    auto number_and_order_basis_beta_stationary_cov_ = wrap_basis_numbers_and_orders<_STATIONARY_>(n_basis_beta_stationary_cov,n_order_basis_beta_stationary_cov,knots_beta_stationary_cov_.size(),q_C);
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = number_and_order_basis_beta_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_beta_stationary_cov_ = number_and_order_basis_beta_stationary_cov_[FDAGWR_FEATS::order_basis_string];
    //events cov
    auto number_and_order_basis_events_cov_ = wrap_basis_numbers_and_orders<_EVENT_>(n_basis_events_cov,n_order_basis_events_cov,knots_events_cov_.size(),q_E);
    std::vector<std::size_t> number_basis_events_cov_ = number_and_order_basis_events_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_events_cov_ = number_and_order_basis_events_cov_[FDAGWR_FEATS::order_basis_string];
    for(std::size_t i = 0; i < q_E; ++i){   
        check_dim_input<_EVENT_>(number_basis_events_cov_[i],coefficients_events_cov_[i].rows()," covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_EVENT_>(number_of_statistical_units_,coefficients_events_cov_[i].cols()," covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta events cov
    auto number_and_order_basis_beta_events_cov_ = wrap_basis_numbers_and_orders<_EVENT_>(n_basis_beta_events_cov,n_order_basis_beta_events_cov,knots_beta_events_cov_.size(),q_E);
    std::vector<std::size_t> number_basis_beta_events_cov_ = number_and_order_basis_beta_events_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_beta_events_cov_ = number_and_order_basis_beta_events_cov_[FDAGWR_FEATS::order_basis_string];
    //stations cov
    auto number_and_order_basis_stations_cov_ = wrap_basis_numbers_and_orders<_STATION_>(n_basis_stations_cov,n_order_basis_stations_cov,knots_stations_cov_.size(),q_S);
    std::vector<std::size_t> number_basis_stations_cov_ = number_and_order_basis_stations_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_stations_cov_ = number_and_order_basis_stations_cov_[FDAGWR_FEATS::order_basis_string];
    for(std::size_t i = 0; i < q_E; ++i){   
        check_dim_input<_STATION_>(number_basis_stations_cov_[i],coefficients_stations_cov_[i].rows()," covariate " + std::to_string(i+1) + " coefficients matrix rows");
        check_dim_input<_STATION_>(number_of_statistical_units_,coefficients_stations_cov_[i].cols()," covariate " + std::to_string(i+1) + " coefficients matrix columns");}
    //beta stations cov 
    auto number_and_order_basis_beta_stations_cov_ = wrap_basis_numbers_and_orders<_STATION_>(n_basis_beta_stations_cov,n_order_basis_beta_stations_cov,knots_beta_stations_cov_.size(),q_S);
    std::vector<std::size_t> number_basis_beta_stations_cov_ = number_and_order_basis_beta_stations_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_beta_stations_cov_ = number_and_order_basis_beta_stations_cov_[FDAGWR_FEATS::order_basis_string];


    //  DISTANCES
    //events    DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_events_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_events);
    check_dim_input<_EVENT_>(number_of_statistical_units_,coordinates_events_.rows()," coordinates matrix rows");
    check_dim_input<_EVENT_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_events_.cols()," coordinates matrix columns");
    distance_matrix<_DISTANCE_> distances_events_cov_(std::move(coordinates_events_),number_threads);
    //stations  DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_stations_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_stations);
    check_dim_input<_STATION_>(number_of_statistical_units_,coordinates_stations_.rows()," coordinates matrix rows");
    check_dim_input<_STATION_>(FDAGWR_FEATS::number_of_geographical_coordinates,coordinates_stations_.cols()," coordinates matrix columns");
    distance_matrix<_DISTANCE_> distances_stations_cov_(std::move(coordinates_stations_),number_threads);


    //  PENALIZATION TERMS
    //stationary
    std::vector<double> lambda_stationary_cov_ = wrap_penalizations<_STATIONARY_>(penalization_stationary_cov,q_C);
    //events
    std::vector<double> lambda_events_cov_ = wrap_penalizations<_EVENT_>(penalization_events_cov,q_E);
    //stations
    std::vector<double> lambda_stations_cov_ = wrap_penalizations<_STATION_>(penalization_stations_cov,q_S);


    //  KERNEL BANDWITH
    //events
    double bandwith_events_cov_ = wrap_bandwith<_EVENT_>(bandwith_events);
    //stations
    double bandwith_stations_cov_ = wrap_bandwith<_STATION_>(bandwith_stations);

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
    functional_weight_matrix_stationary<_STATIONARY_> W_C(coefficients_response_reconstruction_weights_,
                                                          number_threads);
    W_C.compute_weights();                                                      
    //events
    functional_weight_matrix_non_stationary<_EVENT_,_KERNEL_,_DISTANCE_> W_E(coefficients_response_reconstruction_weights_,
                                                                             std::move(distances_events_cov_),
                                                                             bandwith_events_cov_,
                                                                             number_threads);
    W_E.compute_weights();                                                                         
    //stations
    functional_weight_matrix_non_stationary<_STATION_,_KERNEL_,_DISTANCE_> W_S(coefficients_response_reconstruction_weights_,
                                                                               std::move(distances_stations_cov_),
                                                                               bandwith_stations_cov_,
                                                                               number_threads);
    W_S.compute_weights();



    //COMPUTING THE BASIS SYSTEMS FOR THE BETAS
    //stationary
    basis_systems< _DOMAIN_, bsplines_basis > bs_C(knots_beta_stationary_cov_eigen_w_, 
                                                  order_basis_beta_stationary_cov_, 
                                                  number_basis_beta_stationary_cov_, 
                                                  q_C);
    //events
    basis_systems< _DOMAIN_, bsplines_basis > bs_E(knots_beta_events_cov_eigen_w_, 
                                                  order_basis_beta_events_cov_, 
                                                  number_basis_beta_events_cov_, 
                                                  q_E);
    //stations
    basis_systems< _DOMAIN_, bsplines_basis > bs_S(knots_beta_stations_cov_eigen_w_,  
                                                  order_basis_beta_stations_cov_, 
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
    bsplines_basis<_DOMAIN_> basis_response_(knots_response_eigen_w_,number_basis_response_,order_basis_response_);
    functional_data<_DOMAIN_,bsplines_basis > fd_response_(std::move(coefficients_response_),basis_response_);
    //decltype(basis_response_)
    //constant_basis<_DOMAIN_> basis_response_(knots_response_eigen_w_);
    //functional_data<_DOMAIN_,constant_basis > fd_response_(std::move(coefficients_response_),basis_response_);

    //double el = 0.0;
    //Rcout << "Eval basis pre in" << el << ": " << basis_response_.eval_base(el) << std::endl;
    /*
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
    */


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

    //returning element
    Rcpp::List l;

    l["Type of gwr"] = "fgwr";

    return l;
}