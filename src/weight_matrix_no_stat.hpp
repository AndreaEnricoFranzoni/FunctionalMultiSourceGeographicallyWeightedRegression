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


#ifndef FDAGWR_WEIGHT_MATRIX_NON_STATIONARY_HPP
#define FDAGWR_WEIGHT_MATRIX_NON_STATIONARY_HPP

#include "weight_matrix.hpp"


/*!
* @file weight_matrix_no_stat.hpp
* @brief Construct the non stationary weight matrix for performing the geographically weighted regression. Weights consist of functional reconstruction weights times the spatial ones
* @author Andrea Enrico Franzoni
*/


template< KERNEL_FUNC kernel_func >  
class weight_matrix_non_stationary : public weight_matrix_base< weight_matrix_non_stationary<kernel_func>, kernel_func >
{
public:

  /*!
  * @brief Constructor for the non stationary weight matrix: each weight consists of the reconstruction functional weight times the spatial ones
  * @param stationary_weight stationary weight, for each statistical unit
  * @param n number of statistical units
  * @param number_threads number of threads for OMP
  */
  weight_matrix_non_stationary(const std::vector<double> weight_stat,
                               const std::vector<double> weight_no_stat,
                               std::size_t n,
                               int number_threads)
                    : weight_matrix_base<weight_matrix_non_stationary,kernel_func>(n,number_threads) {}
};


#endif  /*FDAGWR_WEIGHT_MATRIX_NON_STATIONARY_HPP*/