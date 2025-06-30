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


/*!
* @file removing_nan_imp.hpp
* @brief Implementation of non-dummy NaNs removal
* @author Andrea Enrico Franzoni
*/


/*!
* @brief Mean replacing: substituting non-dummy NaNs with the mean of the row
* @details 'REM_NAN::MR' dispatch
*/
template<typename T,REM_NAN MA_t>
void
removing_nan<T,MA_t>::row_removal(Eigen::Block<Eigen::Matrix<T,-1,-1>,1>& row, MAT<REM_NAN::MR>)
{
  //counting how many non-NaNs
  const auto n_not_nan = std::count_if(row.begin(),row.end(),[](T el){return !std::isnan(el);});
  std::vector<T> el_not_nan;
  el_not_nan.reserve(n_not_nan);
  //saving non-NaNs values
  std::copy_if(row.begin(),row.end(),std::back_inserter(el_not_nan),[](T el){return !std::isnan(el);});
  //evaluating their mean
  const T mean = std::accumulate(el_not_nan.begin(),el_not_nan.end(),static_cast<T>(0),std::plus{})/static_cast<T>(n_not_nan);
  el_not_nan.clear();
  //replacing
  std::replace_if(row.begin(),row.end(),[](T el){return std::isnan(el);},mean);
}  


/*!
* @brief Zeros replacing: substituting non-dummy NaNs with 0s
* @details 'REM_NAN::ZR' dispatch
*/
template<typename T,REM_NAN MA_t>
void
removing_nan<T,MA_t>::row_removal(Eigen::Block<Eigen::Matrix<T,-1,-1>,1>& row, MAT<REM_NAN::ZR>)
{
  std::replace_if(row.begin(),row.end(),[](T el){return std::isnan(el);},static_cast<T>(0)); 
}