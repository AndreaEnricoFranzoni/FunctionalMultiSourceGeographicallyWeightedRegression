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


#include "traits_fdagwr.hpp"
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
  //std::tuple< std::vector<fdagwr_traits::Dense_Matrix>, std::vector<std::string>> _output_;
  Rcpp::Nullable<Rcpp::CharacterVector> cov_names_R = cov_coeff_list.names();


  if (cov_names.isNull())
  {
      std::vector<std::string> covariates_names;
      std::size_t number_cov = cov_coeff_list.size();
      covariates_names.reserve(number_cov);

      std::string covariates_type = covariate_conversion<fdagwr_cov_t>;

      for(std::size_t i = 0; i < number_cov; ++i){
        covariates_names.emplace_back("Cov" + covariates_type + std::to_string(i+1));
      }

      return covariates_type;
  }
  
  std::vector<std::string> covariates_names = as<std::vector<std::string>>(cov_names_R);
  return covariates_names;
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