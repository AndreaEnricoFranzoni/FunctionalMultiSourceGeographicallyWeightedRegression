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


template< class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis, FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::STATIONARY >  
    requires fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
class functional_weight_matrix_stationary : public functional_weight_matrix_base< functional_weight_matrix_stationary<domain_type,basis_type>, domain_type, basis_type, stationarity_t >
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
    functional_weight_matrix_stationary(const functional_data<domain_type,basis_type> &y_recostruction_weights_fd,
                                        int number_threads)
                      : 
                      functional_weight_matrix_base<functional_weight_matrix_stationary,domain_type,basis_type>(y_recostruction_weights_fd,
                                                                                                                number_threads) 
                      {   
                        static_assert(stationarity_t == FDAGWR_COVARIATES_TYPES::STATIONARY,
                                      "Functional weight matrix for stationary covariates needs FDAGWR_COVARIATES_TYPES::STATIONARY as template parameter");
                      }

    /*!
    * @brief Getter for the functional stationary weight matrix
    * @return the private m_weights
    */
    const WeightMatrixType<stationarity_t>& weights() const {return m_weights;}

    /*!
    * @brief Function to compute stationary weights
    */
    inline
    void
    computing_weights()
    {
      //to shared the values with OMP
      auto n_stat_units = this->n();
      //preparing the container for the functional stationary weight matrix
      m_weights.resize(n_stat_units);

#ifdef _OPENMP
#pragma omp parallel for shared(n_stat_units) num_threads(this->number_threads())
#endif
      for(std::size_t i = 0; i < n_stat_units; ++i)
      {
        FDAGWR_TRAITS::f_type w_i = [i](const double & loc){return this->y_recostruction_weights_fd().eval(loc,i);};
        m_weights[i] = w_i;
      }
    }
};

#endif  /*FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_STATIONARY_HPP*/