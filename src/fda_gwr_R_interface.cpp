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
#include "parameters_wrapper_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include "basis_systems.hpp"



#include "weight_matrix_stat.hpp"
#include "weight_matrix_no_stat.hpp"
#include "test_basis_eval.hpp"


#include "fdaPDE-core/fdaPDE/splines.h"


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
Rcpp::List fmsgwr(Rcpp::NumericVector fd_points,
                  double left_extreme_domain,
                  double right_extreme_domain,
                  Rcpp::IntegerVector n_basis_stationary_cov,
                  Rcpp::IntegerVector n_order_basis_stationary_cov,
                  Rcpp::NumericVector knots_stationary_cov,
                  Rcpp::NumericVector penalization_stationary_cov,
                  Rcpp::IntegerVector n_basis_events_cov,
                  Rcpp::IntegerVector n_order_basis_events_cov,
                  Rcpp::NumericVector knots_events_cov,
                  Rcpp::NumericVector penalization_events_cov,
                  Rcpp::IntegerVector n_basis_stations_cov,
                  Rcpp::IntegerVector n_order_basis_stations_cov,
                  Rcpp::NumericVector knots_stations_cov,
                  Rcpp::NumericVector penalization_stations_cov,
                  Rcpp::Nullable<int> num_threads = R_NilValue){
    //funzione per il multi-source gwr
    //  !!!!!!!! NB: l'ordine delle basi su c++ corrisponde al degree su R !!!!!

    //Rcpp::NumericMatrix distances_events,
    //Rcpp::NumericMatrix distances_stations,

    //checking and wrapping input parameters
    int number_threads = wrap_num_thread(num_threads);


    std::vector<int> order_basis_test = Rcpp::as<std::vector<int>>(n_order_basis_stationary_cov);
    std::vector<double> knots_test = Rcpp::as<std::vector<double>>(knots_stationary_cov);


    basis_systems<BASIS_TYPE::BSPLINES> bs(3,order_basis_test,knots_test);

    std::vector<double> ev_points = Rcpp::as<std::vector<double>>(fd_points);
    Eigen::Map<fdagwr_traits::Dense_Matrix> locs(ev_points.data(), ev_points.size(), 1);

    for(std::size_t i = 0; i < bs.q(); ++i){
            Eigen::SparseMatrix<double> Psi = spline_basis_eval(bs.basis_systems()[i], locs);

            std::cout << i+1 << "basis evaluation at location" << std::endl;
            std::cout << Eigen::Matrix<double, Dynamic, Dynamic>(Psi) << std::endl; 
    }



    /*
    std::vector<double> trial{12.0,8.0,7.9};

    weight_matrix_stationary<KERNEL_FUNC::GAUSSIAN> trial_sm(trial,
                                                              trial.size(),
                                                              number_threads);

    auto mat = trial_sm.weights();
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            Rcout << mat.coeff(i, j) << " ";
        }
        Rcout << "" << std::endl;
    }

    std::vector<double> non_stat{7.0,1.4,7.9};
    weight_matrix_non_stationary<KERNEL_FUNC::GAUSSIAN> trial_wmns(trial,
                                                                    non_stat,
                                                                    trial.size(),
                                                                    100,
                                                                    number_threads);

    auto mat1 = trial_wmns.weights();
    for (int i = 0; i < mat1.rows(); ++i) {
        for (int j = 0; j < mat1.cols(); ++j) {
            Rcout << mat1.coeff(i, j) << " ";
        }
        Rcout << "" << std::endl;
    }
    */
    


    //returning element
    Rcpp::List l;

    l["Type of gwr"] = "fmsgwr";

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

    //returning element
    Rcpp::List l;

    l["Type of gwr"] = "fgwr";

    return l;
}