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

#include "removing_nan.hpp"
#include <RcppEigen.h>


/*!
* @file removing_nan_cleaner_imp.hpp
* @brief Implementation of dummy NaNs removal
* @author Andrea Enrico Franzoni
*/


/*!
* @brief Function checking if a row is entirely made by NaNs
* @param row row to be checked
* @param tot_cols total number of columns of the row
* @return true if all NaNs in the row, false otherwise
*/
inline
bool 
is_row_all_nan(const Rcpp::NumericMatrix::ConstRow& row, int tot_cols) 
{
  return std::count_if(row.cbegin(),row.cend(),[](auto el){return std::isnan(el);}) == tot_cols  ?  true  :  false;
}


/*!
* @brief Identifying rows of only NaNs
* @param x Rcpp::NumericMatrix
* @return a std::set containing the position, with respect to the original matrix, of the to be retained rows
* @note Depends on RcppEigen for interfacing with R containers
*/
//
//  [[Rcpp::depends(RcppEigen)]]
std::set<int>
rows_entire_NaNs(const Rcpp::NumericMatrix &x)
{
  if (x.nrow() == 0 || x.ncol() == 0) 
  {
    std::string error_message1 = "Empty data matrix";
    throw std::invalid_argument(error_message1);
  }
    
  int tot_cols = x.ncol();
  std::set<int> rows_to_be_kept;
  for (int i = 0; i < x.nrow(); ++i) { if (!is_row_all_nan(x.row(i),tot_cols)) {  rows_to_be_kept.insert(i);}}
    
  if (rows_to_be_kept.empty())
  {
    std::string error_message2 = "Only-NaNs data matrix";
    throw std::invalid_argument(error_message2);
  }
    
  return rows_to_be_kept;
}


// function to remove the all-NaNs rows
/*!
* @brief Removing rows of only NaNs
* @param x Rcpp::NumericMatrix
* @param rows_to_be_kept position, wrt to 'x', of the rows to be kept
* @return a Rcpp::NumericMatrix  without the rows of only NaNs
* @note Depends on RcppEigen for interfacing with R containers
*/
//
//  [[Rcpp::depends(RcppEigen)]]
Rcpp::NumericMatrix 
removing_NaNS_entire_rows(const Rcpp::NumericMatrix &x,const std::set<int> &rows_to_be_kept)
{
  Rcpp::NumericMatrix x_clean(rows_to_be_kept.size(),x.ncol());
  int counter_row_clean = 0;
  std::for_each(rows_to_be_kept.cbegin(),rows_to_be_kept.cend(),[&x,&x_clean,&counter_row_clean](auto el){x_clean.row(counter_row_clean)=x.row(el); counter_row_clean++;});
  
  return x_clean;
}