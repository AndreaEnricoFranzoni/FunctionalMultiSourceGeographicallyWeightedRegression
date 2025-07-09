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


#ifndef FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_NON_STATIONARY_HPP
#define FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_NON_STATIONARY_HPP

#include "functional_weight_matrix.hpp"


/*!
* @file functional_weight_matrix_no_stat.hpp
* @brief Construct the non stationary weight matrix for performing the geographically weighted regression. Weights consist of functional reconstruction weights and spatial weights
* @author Andrea Enrico Franzoni
*/



template< FDAGWR_COVARIATES_TYPES stationarity_t, KERNEL_FUNC kernel_func >  
class functional_weight_matrix_non_stationary : public functional_weight_matrix_base< functional_weight_matrix_non_stationary<stationarity_t,kernel_func>, stationarity_t, kernel_func >
{

private:

    /*!Vector of diagonal matrices storing the weights*/
    WeightMatrixType<stationarity_t> m_weights;

    distance_matrix<DISTANCE_MEASURE::EUCLIDEAN> m_distance_matrix;

    double m_kernel_bandwith;



public:

    /*!
    * @brief Constructor for the non stationary weight matrix: each weight consists of the reconstruction functional weight and spatial weight
    * @param weight_stat stationary weight, for each statistical unit
    * @param n number of statistical units
    * @param number_threads number of threads for OMP
    */
    template< typename DIST_MATRIX_OBJ >
    functional_weight_matrix_non_stationary(const fdagwr_traits::Dense_Matrix & coeff_stat_weights,
                                            DIST_MATRIX_OBJ&& distance_matrix,
                                            double kernel_bwt,
                                            int number_threads)

                                : 
                                  functional_weight_matrix_base<functional_weight_matrix_non_stationary,stationarity_t,kernel_func>(coeff_stat_weights,
                                                                                                                                    number_threads),
                                  m_distance_matrix{std::forward<DIST_MATRIX_OBJ>(distance_matrix)},
                                  m_kernel_bandwith(kernel_bwt) 
                                {   
                                    std::cout << "Constructing a non stationary weight matrix" << std::endl;
                                }

    inline
    void
    computing_weights()
    {

      m_weights.resize(this->number_statistical_units());

      for(std::size_t i = 0; i < this->number_statistical_units(); ++i)
      {
        m_weights[i].resize(this->number_abscissa_evaluations());

        for (std::size_t j = 0; j < this->number_abscissa_evaluations(); ++j)
        {
          fdagwr_traits::Diag_Matrix weight_given_abscissa(this->coeff_stat_weights().row(j).transpose());
          m_weights[i][j] = weight_given_abscissa;
        }
        
      }
    }
};

#endif  /*FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_NON_STATIONARY_HPP*/