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


#include "weight_matrix.hpp"

/*!
* @file kernel_functions_eval.hpp
* @brief Implementation of tag-dispatched function for evaluating the kernel functions
* @author Andrea Enrico Franzoni
*/


/*!
* @brief Kernel Gaussian function evaluation, given a bandwith, of a given distance
* @param distance distance for which is needed the kernel function evaluation
* @param bandwith bandwith of the gaussian kernel function
* @details 'KERNEL_FUNC::GAUSSIAN' dispatch via std::integral_constant.
*/
template< class D, FDAGWR_COVARIATES_TYPES stationarity_t, KERNEL_FUNC kernel_func >
double
weight_matrix_base<D,stationarity_t,kernel_func>::kernel_eval(double distance, double bandwith, KERNEL_FUNC_T<KERNEL_FUNC::GAUSSIAN>)
const
{
  //gaussian kernel function evaluation
  return gaussian_kernel<double>(distance,bandwith);
}