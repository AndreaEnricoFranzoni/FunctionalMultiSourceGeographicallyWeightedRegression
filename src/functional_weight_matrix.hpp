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


#ifndef FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_HPP
#define FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_HPP


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include "concepts"

#include <cassert>


/*!
* @file functional_weight_matrix.hpp
* @brief Construct the weight matrix for performing the geographically weighted regression
* @author Andrea Enrico Franzoni
*/



/*!
* @brief Indicating the container to storing the functional weight matrix
*/
template <FDAGWR_COVARIATES_TYPES stationarity_t>
using WeightMatrixType = std::conditional<stationarity_t == FDAGWR_COVARIATES_TYPES::STATIONARY,
                                          std::vector< FDAGWR_TRAITS::f_type >,        //se stazionario, ogni elemento del vettore corrisponde ad un valore dell'ascissa, e di conseguenza vi è la giusta matrice peso
                                          std::vector< std::vector< FDAGWR_TRAITS::f_type >>>::type;



/*!
* @class weight_matrix_base
* @brief Template class for constructing the weight matrix: a squared matrix containing the weight for each unit
* @tparam D type of the derived class (for static polymorphism thorugh CRTP):
*         - stationary: 'weight_matrix_stat'
*         - non stationary: 'weight_matrix_no_stat'
* @tparam kernel_func kernel function for the evaluation of the weights (enumerator)
* @details It is the base class. Polymorphism is known at compile time thanks to Curiously Recursive Template Pattern (CRTP) 
*/
template< class D, class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis >
    requires fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
class functional_weight_matrix_base
{

private:
    /*!Matrix containing the stationary weights: abscissa x units*/
    functional_data<domain_type,basis_type> m_y_recostruction_weights;
    /*!Number of statistical units*/
    std::size_t m_n;
    /*!Number of threads for OMP*/
    int m_number_threads;

public:
    /*!
    * @brief Constructor for the weight matrix (diagonal matrix containing the weight for each unit)
    * @param n number of statistical units
    * @param number_threads number of threads for OMP
    */
    functional_weight_matrix_base(const functional_data<domain_type,basis_type> &y_recostruction_weights,
                                  int number_threads)
        :      
            m_y_recostruction_weights(y_recostruction_weights),
            m_n(y_recostruction_weights.n()), 
            m_number_threads(number_threads)  
        {}

    /*!
    * @brief Getter for the coefficient-stationary-weights matrix
    * @return the private m_coeff_stat_weights
    */
    const functional_data<domain_type,basis_type>& y_recostruction_weights() const {return m_y_recostruction_weights;}

    /*!
    * @brief Getter for the number of available evaluations of the stationary weights
    * @return the private m_number_abscissa_evaluations
    */
    std::size_t n() const {return m_n;}

    /*!
    * @brief Getter for the number of threads for OMP
    * @return the private m_number_threads
    */
    inline int number_threads() const {return m_number_threads;}

    /*!
    * @brief Computing weights accordingly if they are only stationary or not
    * @details Entails downcasting of base class with a static cast of pointer to the derived-known-at-compile-time class, CRTP fashion
    */
    inline
    void 
    compute_weights() 
    {
        static_cast<D*>(this)->computing_weights();   //solving depends on child class: downcasting with CRTP of base to derived
    }
};

#endif  /*FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_HPP*/