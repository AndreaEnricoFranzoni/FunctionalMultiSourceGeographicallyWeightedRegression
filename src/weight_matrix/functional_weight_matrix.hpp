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


#ifndef FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_HPP
#define FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_HPP


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"
#include "../functional_matrix/functional_matrix_storing_type.hpp"
#include "../functional_matrix/functional_matrix_utils.hpp"

#include <cassert>


/*!
* @file functional_weight_matrix.hpp
* @brief Construct the functional weight matrix for performing functional weighted regression
* @author Andrea Enrico Franzoni
*/



/*!
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @tparam stationarity_t covariate type
* @brief std::conditional to indicate the container used to storie the functional weight matrix. Vector of functional matrices for stationary covariates, containing the weights. Vector, one element for each statistical unit, of vector of functional matrices for non-stationary covariates
*/
template <typename INPUT = double, typename OUTPUT = double, FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::STATIONARY>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using WeightMatrixType = std::conditional<stationarity_t == FDAGWR_COVARIATES_TYPES::STATIONARY,
                                          std::vector< FUNC_OBJ< INPUT,OUTPUT > >,        
                                          std::vector< std::vector< FUNC_OBJ< INPUT,OUTPUT > >>>::type;



/*!
* @class functional_weight_matrix_base
* @tparam D type of the derived class (for static polymorphism thorugh CRTP):
*         - stationary: 'functional_weight_matrix_stationary'
*         - non stationary: 'functional_weight_matrix_non_stationary'
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @tparam domain_type geometry of the basis domain
* @tparam basis_type type of basis used to construct the functional weights
* @brief Base class for constructing the functional weight matrix
* @details It is the base class. Polymorphism is known at compile time thanks to Curiously Recursive Template Pattern (CRTP) 
*/
template< class D, typename INPUT = double, typename OUTPUT = double, class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>) && fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
class functional_weight_matrix_base
{

private:
    /*!Functional response reconstruction weights*/
    functional_data<domain_type,basis_type> m_y_recostruction_weights_fd;
    /*!Number of statistical units*/
    std::size_t m_n;
    /*!Number of threads for OMP*/
    int m_number_threads;

public:

    /*!
    * @brief Constructor
    * @param y_recostruction_weights_fd functional response reconstruction weights
    * @param number_threads number of threads for OMP
    */
    functional_weight_matrix_base(const functional_data<domain_type,basis_type> &y_recostruction_weights_fd,
                                  int number_threads)
        :      
            m_y_recostruction_weights_fd(y_recostruction_weights_fd),
            m_n(m_y_recostruction_weights_fd.n()), 
            m_number_threads(number_threads)  
        {}

    /*!
    * @brief Getter for the functional response reconstruction weights
    * @return the private m_y_recostruction_weights_fd
    */
    const functional_data<domain_type,basis_type>& y_recostruction_weights_fd() const {return m_y_recostruction_weights_fd;}

    /*!
    * @brief Getter for the number of statistical units
    * @return the private m_n
    */
    std::size_t n() const {return m_n;}

    /*!
    * @brief Getter for the number of threads for OMP
    * @return the private m_number_threads
    */
    inline int number_threads() const {return m_number_threads;}

    /*!
    * @brief Computing functional weights accordingly if they are only stationary or not
    * @details Entails downcasting of base class with a static cast of pointer to the derived-known-at-compile-time class, CRTP fashion
    */
    inline
    void 
    compute_weights() 
    {
        static_cast<D*>(this)->computing_weights();   //solving depends on child class: downcasting with CRTP of base to derived
    }

    /*!
    * @brief Computing functional weights (non-stationary) within the training set and the prediction set
    */
    inline
    void 
    compute_weights_pred() 
    {
        static_cast<D*>(this)->compute_weights_pred();   //solving depends on child class: downcasting with CRTP of base to derived
    }
};

#endif  /*FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_HPP*/