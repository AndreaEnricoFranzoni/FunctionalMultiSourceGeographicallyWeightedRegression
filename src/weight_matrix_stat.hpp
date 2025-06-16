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



#ifndef FDAGWR_WEIGHT_MATRIX_STATIONARY_HPP
#define FDAGWR_WEIGHT_MATRIX_STATIONARY_HPP


#include "weight_matrix.hpp"


/*!
* @file weight_matrix_stat.hpp
* @brief Construct the stationary weight matrix for performing the geographically weighted regression
* @author Andrea Enrico Franzoni
*/


template< KERNEL_FUNC kernel_func >  
class weight_matrix_stationary : public weight_matrix_base< weight_matrix_stationary<kernel_func>, kernel_func >
{
public:


  
  /*!
  * @brief Constructor if number of PPCs k is known (k_imp=K_IMP::YES), for derived class: constructs firstly CV_base<CV_alpha,...>
  * @param Data fts data matrix
  * @param strategy splitting training/validation strategy
  * @param params input space for regularization parameter
  * @param k number of retained PPCs
  * @param pred_f function to make validation set prediction (overloading with k imposed)
  * @param number_threads number of threads for OMP
  * @details Universal constructor: move semantic used to optimazing handling big size objects
  */
  weight_matrix_stationary(const std::vector<double> data,
                           int n,
                           int number_threads)
                    : weight_matrix_base<weight_matrix_stationary,kernel_func>(data,n,number_threads)    {}
};



#endif  /*FDAGWR_WEIGHT_MATRIX_STATIONARY_HPP*/