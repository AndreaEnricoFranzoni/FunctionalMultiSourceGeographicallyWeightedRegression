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

#include <string>

#include "traits_fdagwr.hpp"
#include "data_reader.hpp"

#include "basis_systems.hpp"
#include "parameters_wrapper_fdagwr.hpp"


#include "functional_weight_matrix_stat.hpp"
#include "functional_weight_matrix_no_stat.hpp"


#include "distance_matrix.hpp"

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
                  double penalization_y_points,
                  Rcpp::NumericMatrix coeff_rec_weights_y_points,
                  Rcpp::Nullable<int> n_order_basis_rec_weights_y_points,
                  Rcpp::Nullable<int> n_basis_rec_weights_y_points,
                  Rcpp::List coeff_stationary_cov,
                  Rcpp::NumericVector knots_stationary_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_order_basis_stationary_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_basis_stationary_cov,
                  Rcpp::NumericVector penalization_stationary_cov,
                  Rcpp::NumericVector knots_beta_stationary_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_order_basis_beta_stationary_cov,
                  Rcpp::Nullable<Rcpp::IntegerVector> n_basis_beta_stationary_cov,
                  Rcpp::List coeff_events_cov,
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


    Rcout << "fdagwr.38: " << std::endl;

    using _DATA_TYPE_ = double;                                         //data type
    constexpr auto _NAN_REM_ = REM_NAN::MR;                             //how to remove nan (with mean of non-nans)
    constexpr auto _STATIONARY_ = FDAGWR_COVARIATES_TYPES::STATIONARY;  //enum for stationary covariates
    constexpr auto _EVENT_ = FDAGWR_COVARIATES_TYPES::EVENT;            //enum for event covariates
    constexpr auto _STATION_ = FDAGWR_COVARIATES_TYPES::STATION;        //enum for station covariates
    constexpr auto _DISTANCE_ = DISTANCE_MEASURE::EUCLIDEAN;            //enum for euclidean distance within statistical units locations
    constexpr auto _KERNEL_ = KERNEL_FUNC::GAUSSIAN;                    //kernel function to smooth the distances within statistcal units locations



    ///////////////////////////////////////////////////////
    /////   CHECKING and WRAPPING INPUT PARAMETERS  ///////
    ///////////////////////////////////////////////////////

    //  NUMBER OF THREADS
    int number_threads = wrap_num_thread(num_threads);


    //  RESPONSE
    //raw data
    auto response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(y_points);       //Eigen dense matrix type (auto is necessary )
    //coefficients
    auto coefficients_response_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_y_points);
    //reconstruction weights
    auto coefficiente_response_reconstruction_weights_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coeff_rec_weights_y_points);


    //  ABSCISSA POINTS
    std::vector<double> abscissa_points_ = wrap_abscissas(t_points,left_extreme_domain,right_extreme_domain);


    //  KNOTS
    //response
    std::vector<double> knots_response_ = wrap_abscissas(knots_y_points,left_extreme_domain,right_extreme_domain);
    //stationary cov
    std::vector<double> knots_stationary_cov_ = wrap_abscissas(knots_stationary_cov,left_extreme_domain,right_extreme_domain);
    //beta stationary cov
    std::vector<double> knots_beta_stationary_cov_ = wrap_abscissas(knots_beta_stationary_cov,left_extreme_domain,right_extreme_domain);
    //events cov
    std::vector<double> knots_events_cov_ = wrap_abscissas(knots_events_cov,left_extreme_domain,right_extreme_domain);
    //beta events cov
    std::vector<double> knots_beta_events_cov_ = wrap_abscissas(knots_beta_events_cov,left_extreme_domain,right_extreme_domain);
    //stations cov
    std::vector<double> knots_stations_cov_ = wrap_abscissas(knots_stations_cov,left_extreme_domain,right_extreme_domain);
    //stations beta cov
    std::vector<double> knots_beta_stations_cov_ = wrap_abscissas(knots_beta_stations_cov,left_extreme_domain,right_extreme_domain);


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


    //  NUMBER AND ORDER OF BASIS
    //response
    auto number_and_order_basis_response_ = wrap_basis_number_and_order(n_basis_y_points,n_order_basis_y_points,knots_response_.size());
    std::size_t number_basis_response_ = number_and_order_basis_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t order_basis_response_ = number_and_order_basis_response_[FDAGWR_FEATS::order_basis_string];
    //response reconstruction weights
    auto number_and_order_basis_weights_response_ = wrap_basis_number_and_order(n_basis_rec_weights_y_points,n_order_basis_rec_weights_y_points,knots_response_.size());
    std::size_t number_basis_weights_response_ = number_and_order_basis_weights_response_[FDAGWR_FEATS::n_basis_string];
    std::size_t order_basis_weights_response_ = number_and_order_basis_weights_response_[FDAGWR_FEATS::order_basis_string];
    //stationary cov
    auto number_and_order_basis_stationary_cov_ = wrap_basis_numbers_and_orders<_STATIONARY_>(n_basis_stationary_cov,n_order_basis_stationary_cov,knots_stationary_cov_.size(),q_C);
    std::vector<std::size_t> number_basis_stationary_cov_ = number_and_order_basis_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_stationary_cov_ = number_and_order_basis_stationary_cov_[FDAGWR_FEATS::order_basis_string];
    //beta stationary cov
    auto number_and_order_basis_beta_stationary_cov_ = wrap_basis_numbers_and_orders<_STATIONARY_>(n_basis_beta_stationary_cov,n_order_basis_beta_stationary_cov,knots_beta_stationary_cov_.size(),q_C);
    std::vector<std::size_t> number_basis_beta_stationary_cov_ = number_and_order_basis_beta_stationary_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_beta_stationary_cov_ = number_and_order_basis_beta_stationary_cov_[FDAGWR_FEATS::order_basis_string];
    //events cov
    auto number_and_order_basis_events_cov_ = wrap_basis_numbers_and_orders<_EVENT_>(n_basis_events_cov,n_order_basis_events_cov,knots_events_cov_.size(),q_E);
    std::vector<std::size_t> number_basis_events_cov_ = number_and_order_basis_events_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_events_cov_ = number_and_order_basis_events_cov_[FDAGWR_FEATS::order_basis_string];
    //beta events cov
    auto number_and_order_basis_beta_events_cov_ = wrap_basis_numbers_and_orders<_EVENT_>(n_basis_beta_events_cov,n_order_basis_beta_events_cov,knots_beta_events_cov_.size(),q_E);
    std::vector<std::size_t> number_basis_beta_events_cov_ = number_and_order_basis_beta_events_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_beta_events_cov_ = number_and_order_basis_beta_events_cov_[FDAGWR_FEATS::order_basis_string];
    //stations cov
    auto number_and_order_basis_stations_cov_ = wrap_basis_numbers_and_orders<_STATION_>(n_basis_stations_cov,n_order_basis_stations_cov,knots_stations_cov_.size(),q_S);
    std::vector<std::size_t> number_basis_stations_cov_ = number_and_order_basis_stations_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_stations_cov_ = number_and_order_basis_stations_cov_[FDAGWR_FEATS::order_basis_string];
    //beta stations cov
    auto number_and_order_basis_beta_stations_cov_ = wrap_basis_numbers_and_orders<_STATION_>(n_basis_beta_stations_cov,n_order_basis_beta_stations_cov,knots_beta_stations_cov_.size(),q_S);
    std::vector<std::size_t> number_basis_beta_stations_cov_ = number_and_order_basis_beta_stations_cov_[FDAGWR_FEATS::n_basis_string];
    std::vector<std::size_t> order_basis_beta_stations_cov_ = number_and_order_basis_beta_stations_cov_[FDAGWR_FEATS::order_basis_string];


    //  DISTANCES
    //events    DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_events_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_events);
    distance_matrix<_DISTANCE_> distances_events_cov_(std::move(coordinates_events_),number_threads);
    //stations  DISTANCES HAVE TO BE COMPUTED WITH THE .compute_distances() method
    auto coordinates_stations_ = reader_data<_DATA_TYPE_,_NAN_REM_>(coordinates_stations);
    distance_matrix<_DISTANCE_> distances_stations_cov_(std::move(coordinates_stations_),number_threads);


    //  PENALIZATION TERMS
    //response
    double lambda_response_ = wrap_penalization(penalization_y_points);
    //stationary
    std::vector<double> lambda_stationary_cov_ = wrap_penalizations<_STATIONARY_>(penalization_stationary_cov);
    //events
    std::vector<double> lambda_events_cov_ = wrap_penalizations<_EVENT_>(penalization_events_cov);
    //stations
    std::vector<double> lambda_stations_cov_ = wrap_penalizations<_STATION_>(penalization_stations_cov);


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
    functional_weight_matrix_stationary<_STATIONARY_> W_C(coefficiente_response_reconstruction_weights_,
                                                          number_threads);
    W_C.compute_weights();                                                      
    //events
    functional_weight_matrix_non_stationary<_EVENT_,_KERNEL_,_DISTANCE_> W_E(coefficiente_response_reconstruction_weights_,
                                                                             std::move(distances_events_cov_),
                                                                             bandwith_events_cov_,
                                                                             number_threads);
    W_E.compute_weights();                                                                         
    //stations
    functional_weight_matrix_non_stationary<_STATION_,_KERNEL_,_DISTANCE_> W_S(coefficiente_response_reconstruction_weights_,
                                                                               std::move(distances_stations_cov_),
                                                                               bandwith_stations_cov_,
                                                                               number_threads);
    W_S.compute_weights();


    //COMPUTING THE BASIS
    basis_systems< fdapde::Triangulation<1, 1>, BASIS_TYPE::BSPLINES > bs(knots_stationary_cov_,
                                                                          order_basis_stationary_cov_,
                                                                          q_C);

    


    /*
    std::vector<BsSpace<Triangulation<1, 1>>> basis_;
    basis_.reserve(q_C);

    for(std::size_t i = 0; i < q_C; ++i)
    {
        Triangulation<1, 1> interval = Triangulation<1, 1>::Interval(knots_stationary_cov_.front(), knots_stationary_cov_.back(), knots_stationary_cov_.size());
        BsSpace<Triangulation<1, 1>> Vh(interval, order_basis_stationary_cov_[i]);
        basis_.push_back(Vh);

        // integration
        TrialFunction u(basis_[i]);
        TestFunction  v(basis_[i]);

        // mass matrix
        auto mass = integral(interval)(u * v);
        Eigen::SparseMatrix<double> M = mass.assemble();

        std::cout << "\n\nmass matrix:  [M]_{ij} = int_I (psi_i * psi_j) of cov " << i+1 << std::endl;
        std::cout << Eigen::Matrix<double, Dynamic, Dynamic>(M) << std::endl;
    }
    */




    
    
    //Eigen::Map<fdagwr_traits::Dense_Matrix> locs(ev_points.data(), ev_points.size(), 1);


    //testing_function(order_basis_stationary_cov_,knots_stationary_cov_);

    


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