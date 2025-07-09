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


#ifndef FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_STATIONARY_HPP
#define FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_STATIONARY_HPP

#include "functional_weight_matrix.hpp"


/*!
* @file weight_matrix_stat.hpp
* @brief Construct the stationary weight matrix for performing the geographically weighted regression. Weights only consist of functional reconstruction weights
* @author Andrea Enrico Franzoni
*/


template< FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::STATIONARY, KERNEL_FUNC kernel_func >  
class functional_weight_matrix_stationary : public functional_weight_matrix_base< functional_weight_matrix_stationary<stationarity_t,kernel_func>, stationarity_t, kernel_func >
{
private:

    /*!Vector of diagonal matrices storing the weights*/
    WeightMatrixType<stationarity_t> m_weights;

public:

    /*!
    * @brief Constructor for the stationary weight matrix: each weight only consists of the reconstruction functional weight
    * @param weight_stat stationary weight, for each statistical unit (abscissas x units)
    * @param n number of statistical units
    * @param number_threads number of threads for OMP
    */
    functional_weight_matrix_stationary(const fdagwr_traits::Dense_Matrix& coeff_stat_weights,
                             int number_threads)
                      : 
  
                      functional_weight_matrix_base<functional_weight_matrix_stationary,stationarity_t,kernel_func>(coeff_stat_weights,
                                                                                                                    number_threads) 
                      {   
                        static_assert(stationarity_t == FDAGWR_COVARIATES_TYPES::STATIONARY,
                                      "Functional weight matrix for stationary covariates needs FDAGWR_COVARIATES_TYPES::STATIONARY as template parameter");
                      }

    inline
    void
    computing_weights()
    {

      m_weights.resize(this->number_abscissa_evaluations());

#ifdef _OPENMP
#pragma omp parallel for shared(this->number_abscissa_evaluations(),this->coeff_stat_weights()) num_threads(this->number_threads())
#endif
      for(std::size_t i = 0; i < this->number_abscissa_evaluations(); ++i)
      {
        fdagwr_traits::Diag_Matrix weight_given_abscissa(this->coeff_stat_weights().row(i).transpose());
        m_weights[i] = weight_given_abscissa;
      }
    }
};

#endif  /*FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_STATIONARY_HPP*/