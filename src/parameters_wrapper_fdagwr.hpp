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



#ifndef FDAGWR_WRAP_PARAMS_HPP
#define FDAGWR_WRAP_PARAMS_HPP

#include <RcppEigen.h>


#include "traits_fdagwr.hpp"
#include "data_reader.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif


/*!
* @file parameters_wrapper.hpp
* @brief Contains methods to check and wrap R-inputs into fdagwr-coherent ones.
* @author Andrea Enrico Franzoni
*/



/*!
* @brief Wrapping the list of covariates
* 
* @note check input consistency
*/
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
std::vector<std::string>
wrap_covariates_names(Rcpp::List cov_coeff_list)
{
  // name of the covariates in the input list list
  Rcpp::Nullable<Rcpp::CharacterVector> cov_names_input_list = cov_coeff_list.names();
  // number of covariates 
  std::size_t number_cov = cov_coeff_list.size();
  // type of the covariates
  std::string covariates_type = covariate_type<fdagwr_cov_t>();


  //if all the covariates do not own a name: put default names for all of them
  if (cov_names_input_list.isNull())
  {
      //output container
      std::vector<std::string> covariates_names;
      covariates_names.reserve(number_cov);
      //put a default value
      for(std::size_t i = 0; i < number_cov; ++i){  covariates_names.emplace_back("Covariate" + covariates_type + std::to_string(i+1));}

      return covariates_names;
  }
  
  //copy the actual names, defaulting only the missing ones
  std::vector<std::string> covariates_names = as<std::vector<std::string>>(cov_names_input_list);

  for(std::size_t i = 0; i < number_cov; ++i){  
        if(covariates_names[i] == "" ){         
            covariates_names[i] = "Covariate" + covariates_type + std::to_string(i+1);}}

  return covariates_names;
}


//
//  [[Rcpp::depends(RcppEigen)]]
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
std::vector<fdagwr_traits::Dense_Matrix>
wrap_covariates_coefficients(Rcpp::List cov_coeff_list)
{
  // number of covariates 
  std::size_t number_cov = cov_coeff_list.size();

  //where to store the covariates coefficients
  std::vector<fdagwr_traits::Dense_Matrix> covariates_coefficients;
  covariates_coefficients.reserve(number_cov);

  //check and read the coefficients for all the covariates
  for(std::size_t i = 0; i < number_cov; ++i){
    
    //read the data
    covariates_coefficients.push_back(reader_data<double,REM_NAN::MR>(cov_coeff_list[i]));

    //checking that all the coefficients refer to the same amount of statistical units 
    //(checking that all the list elements have the same number of columns)
    if(i>0   &&   covariates_coefficients[i-1].cols()!=covariates_coefficients[i].cols())
    {   std::string error_message2 = "All covariates coefficients have to refer to the same number of statistical units";
        throw std::invalid_argument(error_message2);}
  }

  return covariates_coefficients;
}


/*!
* @brief Wrapping the points over which the discrete evaluations of the functional object are available/knots for basis system.
* @param abscissas Rcpp::NumericVector  containing the domain points
* @param a left domain extreme
* @param b right domain extreme
* @return an std::vector<double> containing the points.
* @note Check consistency of domain extremes and passed points, eventualy throwing an error.
*/
inline
std::vector<double>
wrap_abscissas(Rcpp::NumericVector abscissas, double a, double b)    //dim: row of x
{ 
  //check that domain extremes are consistent
  if(a>=b)
  {
    std::string error_message1 = "Left extreme of the domain has to be lower than the right one";
    throw std::invalid_argument(error_message1);
  }
  
  //sorting the abscissas values (security check)
  std::vector<double> abscissas_wrapped = Rcpp::as<std::vector<double>>(abscissas);
  std::sort(abscissas_wrapped.begin(),abscissas_wrapped.end());
  
  //checking that the passed points are inside the domain
  if(abscissas_wrapped[0] < a || abscissas_wrapped.back() > b)
  {
    std::string error_message2 = "The points in which there are the discrete evaluations of the curves have to be in the interval (" + std::to_string(a) + "," + std::to_string(b) + ")";
    throw std::invalid_argument(error_message2);
  }
    
  return abscissas_wrapped;
}


inline
std::map<std::string,std::size_t>
wrap_basis_number_and_order(Rcpp::Nullable<int> basis_number, Rcpp::Nullable<int> basis_order, std::size_t knots_number)
{
  //NUMERO DI BASI E' ORDER + KNOTS (n_nodes) - 1
  std::map<std::string,std::size_t> returning_element;

  //basis number unknown, order unknown: default values
  if(basis_number.isNull() && basis_order.isNull())
  {
    std::size_t order = 3;  //default is a cubic b_spline
    std::size_t n_basis = order + knots_number - static_cast<std::size_t>(1);

    returning_element.insert(std::make_pair(FDAGWR_FEATS::n_basis_string,n_basis));
    returning_element.insert(std::make_pair(FDAGWR_FEATS::order_basis_string,order));
  }

  //basis number unknown, order known
  if (basis_number.isNull() && !basis_order.isNull())
  {
    if (basis_order < 0)
    {
      std::string error_message1 = "Basis order for the response has to be a non-negative integer";
      throw std::invalid_argument(error_message1);
    }

    std::size_t n_basis = static_cast<std::size_t>(basis_order) + knots_number - static_cast<std::size_t>(1);

    returning_element.insert(std::make_pair(FDAGWR_FEATS::n_basis_string,n_basis));
    returning_element.insert(std::make_pair(FDAGWR_FEATS::order_basis_string,basis_order));
  }

  //basis number known, order unknown
  if (!basis_number.isNull() && basis_order.isNull())
  {
    if (static_cast<std::size_t>(basis_number) < knots_number - static_cast<std::size_t>(1))
    {
      std::string error_message2 = "The number of basis for the response has to be at least the number of knots (" + std::to_string(knots_number) + ") - 1";
      throw std::invalid_argument(error_message2);
    }

    std::size_t order = static_cast<std::size_t>(basis_number) - knots_number + static_cast<std::size_t>(1);

    returning_element.insert(std::make_pair(FDAGWR_FEATS::n_basis_string,basis_number));
    returning_element.insert(std::make_pair(FDAGWR_FEATS::order_basis_string,order));
  }

  //both basis number and order known
  if (!basis_number.isNull() && !basis_order.isNull())
  {
    returning_element.insert(std::make_pair(FDAGWR_FEATS::n_basis_string,static_cast<std::size_t>(basis_number)));
    returning_element.insert(std::make_pair(FDAGWR_FEATS::order_basis_string,static_cast<std::size_t>(basis_order)));
  }
  
  return returning_element;
}


/*
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
std::map<std::string,std::vector<std::size_t>
wrap_basis_numbers_and_orders(Rcpp::Nullable<Rcpp::IntegerVector> basis_numbers, Rcpp::Nullable<Rcpp::IntegerVector> basis_orders, std::size_t knots_number)
{

  
  //NUMERO DI BASI E' ORDER + KNOTS (n_nodes) - 1
  if(basis_number.isNull() && basis_order.isNull())

}
*/


inline
double
wrap_penalization(double lambda)
{
  if (lambda < 0)
  {
    std::string error_message = "Penalization term for the response has to be non-negative";
    throw std::invalid_argument(error_message);
  }

  return lambda;  
}


template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
std::vector<double>
wrap_penalizations(Rcpp::NumericVector lambdas)
{
  std::vector<double> lambdas_wrapped = Rcpp::as<std::vector<double>>(lambdas);

  auto min_lambda = std::min_element(lambdas_wrapped.begin(), lambdas_wrapped.end());

  if (*min_lambda < 0)
  {
    // type of the covariates for which the penalization is used
    std::string covariates_type = covariate_type<fdagwr_cov_t>();
    std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),
                   [](unsigned char c) { return std::tolower(c);});

    std::string error_message = "Penalization terms for " + covariates_type + " covariates have to be non-negative";
    throw std::invalid_argument(error_message);
  }

  return lambdas_wrapped;  
}


template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
double
wrap_bandwith(double bandwith)
{
  if (bandwith <= 0)
  {
    // type of the covariates for which the kernel bandwith is used
    std::string covariates_type = covariate_type<fdagwr_cov_t>();
    std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),
                   [](unsigned char c) { return std::tolower(c);});

    std::string error_message = "Kernel bandwith for " + covariates_type + " covariates has to be positive";
    throw std::invalid_argument(error_message);
  }

  return bandwith;  
}


/*!
* @brief Wrapping the number of threads for OMP
* @param num_threads indicates how many threads to be used by multi-threading directives.
* @return the number of threads
* @details if omp is not included: will return 1. If not, a number going from 1 up to the maximum cores available by the machine used (default, or if the input is smaller than 1 or bigger than the maximum number of available cores)
* @note omp requested
*/
inline
int
wrap_num_thread(Rcpp::Nullable<int> num_threads)
{
#ifndef _OPENMP

  return 1;
#else

  //getting maximum number of cores in the machine
  int max_n_t = omp_get_num_procs();
  
  if(num_threads.isNull())
  {
    return max_n_t;
  }
  else
  {
    int n_t = Rcpp::as<int>(num_threads);
    if(n_t < 1 || n_t > max_n_t){  return max_n_t;}
    
    return n_t;
  }
#endif
}


#endif  /*FDAGWR_WRAP_PARAMS_HPP*/