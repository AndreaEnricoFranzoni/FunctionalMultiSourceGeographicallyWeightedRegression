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

#ifndef FDAGWR_FUNC_WEIGHT_MATRIX_STATIONARY_HPP
#define FDAGWR_FUNC_WEIGHT_MATRIX_STATIONARY_HPP


#include "functional_weight_matrix.hpp"
#include "weight_matrix_stat.hpp"


/*!
* @file functional_weight_matrix_stat.hpp
* @brief Construct the functional stationary weight matrix for performing the geographically weighted regression. Weights only consist of functional reconstruction weights
* @author Andrea Enrico Franzoni
*/


template< KERNEL_FUNC kernel_func >  
class functional_weight_matrix_stationary : public functional_weight_matrix_base< functional_weight_matrix_stationary<kernel_func>, kernel_func >
{

public:

  /*!
  * @brief Constructor for the functional stationary weight matrix: feach weight only consists of the reconstruction functional weight
  * @param weight_stat stationary weight, for each statistical unit, for each absissa
  * @param abscissa abscissa of the functional object
  * @param n number of statistical units
  * @param number_threads number of threads for OMP
  */
  functional_weight_matrix_stationary(const std::vector<std::vector<double>> & weights_stat,
                                      const std::vector<double> & abscissa,
                                      std::size_t n, 
                                      std::size_t T,
                                      int number_threads)
                    : functional_weight_matrix_base<functional_weight_matrix_stationary,kernel_func>(n,T,number_threads)
                    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->number_threads())
#endif
                        for(std::size_t i = 0; i < this->T(); ++i){
                            //given an abscissa, costruct its corresponding weight (stationary) matrix
                            weight_matrix_stationary<kernel_func> stat_wei(weights_stat[i],this->n(),this->number_threads());
                            //add it in the right place in the functional weight matrix
                            this->functional_weights().insert(std::make_pair(abscissa[i],stat_wei.weights())); 
                        }
                    }
};

#endif  /*FDAGWR_FUNC_WEIGHT_MATRIX_STATIONARY_HPP*/