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

#ifndef FDAGWR_READ_DATA_HPP
#define FDAGWR_READ_DATA_HPP

#include <RcppEigen.h>


#include "traits_fdagwr.hpp"
#include "removing_nan.hpp"



/*!
* @file data_reader.hpp
* @brief Contains the method to read data
* @author Andrea Enrico Franzoni
*/


/*!
* @brief Function that reads data from R containers, substitute actual NaNs, handle dummy NaNs, removing them but saving their position, and wraps it into C++ objects
* @param X Rcpp::NumericMatrix as passed in input to the R-interfaced function
* @param MA_t how to handle not-dummy NaNs (if substituting with pointwise fts mean or 0)
* @return a pair containing the mapped matrix and a vector with the positions of the dummy NaNs (rows of the original matrix)
* @note Depends on RcppEigen for interfacing with R containers
*/
//
// [[Rcpp::depends(RcppEigen)]]
template<typename T, REM_NAN rem_nan_t> 
fdagwr_traits::Dense_Matrix
reader_data(Rcpp::NumericMatrix X)
{
  //taking the dimensions: n_row is the number of "covariates" (evaluation points), n_col is the number of statistical units
  int n_row = X.nrow();
  int n_col = X.ncol();
    
  //Eigen::Map to map data into KO_Traits::StoringMatrix (Eigen::MatrixXd)  (!!to be modified if the trait is not an Eigen object anymore!!)
  fdagwr_traits::Dense_Matrix x = Eigen::Map<fdagwr_traits::Dense_Matrix>(X.begin(),n_row,n_col);
  
  //check if there are NaNs (NaNs due to missed measurements, not dummy)
  auto check_nan = std::find_if(x.reshaped().cbegin(),x.reshaped().cend(),[](T el){return std::isnan(el);});
  
  //if there are nans: remove them
  if (check_nan!=x.reshaped().end())
  {
    if constexpr(rem_nan_t == REM_NAN::MR)     //replacing nans with the mean
    {
      removing_nan<T,REM_NAN::MR> data_clean(std::move(x));
      data_clean.remove_nan();
      return data_clean.data();       
    }
    if constexpr(rem_nan_t == REM_NAN::ZR)     //replacing nans with 0s
    {
      removing_nan<T,REM_NAN::ZR> data_clean(std::move(x));
      data_clean.remove_nan();
      return data_clean.data();
    }
  }
  
  return x;
}

#endif /*FDAGWR_READ_DATA_HPP*/