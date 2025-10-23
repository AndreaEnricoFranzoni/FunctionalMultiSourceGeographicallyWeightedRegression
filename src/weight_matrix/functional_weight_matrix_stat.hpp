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


#ifndef FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_STATIONARY_HPP
#define FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_STATIONARY_HPP

#include "functional_weight_matrix.hpp"


/*!
* @file functional_weight_matrix_stat.hpp
* @brief Construct the stationary weight matrix for performing the geographically weighted regression. Weights only consist of functional reconstruction weights
* @author Andrea Enrico Franzoni
*/



/*!
* @class functional_weight_matrix_stationary
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @tparam domain_type geometry of the basis domain
* @tparam basis_type type of basis used to construct the functional weights
* @tparam stationarity_t covariate type
* @brief Class for constructing the functional weight matrix for stationary covariates
* @details The functional matrix is a diagonal matrix of functions, n x n (n number of statistical units), where each diagonal element consists in the functional response reconstruction weight of the corresponding statistical unit
*/
template< typename INPUT = double, typename OUTPUT = double, class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis, FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::STATIONARY >  
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>) && fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
class functional_weight_matrix_stationary : public functional_weight_matrix_base< functional_weight_matrix_stationary<INPUT,OUTPUT,domain_type,basis_type>, INPUT, OUTPUT, domain_type, basis_type >
{
    /*!std::function object stored*/
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    /*!std::function input type*/
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

private:
    /*!Vector of n functions, element i-th representing the weight for reconstructing element i-th*/
    WeightMatrixType<INPUT,OUTPUT,stationarity_t> m_weights;

public:

    /*!
    * @brief Constructor
    * @param y_recostruction_weights_fd functional response reconstruction weights
    * @param number_threads number of threads for OMP
    */
    functional_weight_matrix_stationary(const functional_data<domain_type,basis_type> &y_recostruction_weights_fd,
                                        int number_threads)
                      : 
                      functional_weight_matrix_base<functional_weight_matrix_stationary,INPUT,OUTPUT,domain_type,basis_type>(y_recostruction_weights_fd,
                                                                                                                             number_threads) 
                      {   
                        static_assert(stationarity_t == FDAGWR_COVARIATES_TYPES::STATIONARY,
                                      "Functional weight matrix for stationary covariates needs FDAGWR_COVARIATES_TYPES::STATIONARY as template parameter");
                      }

    /*!
    * @brief Getter for the functional stationary weight matrix
    * @return the private m_weights
    */
    const WeightMatrixType<INPUT,OUTPUT,stationarity_t>& weights() const {return m_weights;}

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
        //element W(i,i) is the reconstruction weight of unit i-th
        F_OBJ w_i = [i,this](F_OBJ_INPUT loc){return this->y_recostruction_weights_fd().eval(loc,i);};
        m_weights[i] = w_i;
      }
    }

    /*!
    * @brief Function to compute stationary weights within training and prediction sets
    * @note empty, since these weights do not vary in the space
    */
    inline
    void
    compute_weights_pred()
    {}
};

#endif  /*FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_STATIONARY_HPP*/