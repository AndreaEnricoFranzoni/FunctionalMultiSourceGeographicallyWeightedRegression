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


#ifndef FDAGWR_FUNC_WEIGHT_MATRIX_HPP
#define FDAGWR_FUNC_WEIGHT_MATRIX_HPP


#include "traits_fdagwr.hpp"


#ifdef _OPENMP
#include <omp.h>
#endif


/*!
* @file functional_weight_matrix.hpp
* @brief Construct the functional weight matrix for performing the geographically weighted regression
* @author Andrea Enrico Franzoni
*/


//define the type of the functional weight matrix
typedef std::map<double,fdagwr_traits::Sparse_Matrix> func_weight_mat_t


/*!
* @class functional_weight_matrix_base
* @brief Template class for constructing the fucntional weight matrix: for each abscissa available, a squared matrix containing the weight for each unit
* @tparam D type of the derived class (for static polymorphism thorugh CRTP):
*         - stationary: 'functional_weight_matrix_stationary'
*         - non stationary: 'functional_weight_matrix_no_stationary'
* @tparam kernel_func kernel function for the evaluation of the weights (enumerator)
* @details It is the base class. Polymorphism is known at compile time thanks to Curiously Recursive Template Pattern (CRTP) 
*/
template< class D, KERNEL_FUNC kernel_func >
class functional_weight_matrix_base
{

private:
    /*!Functional weight matrix: key is the abscissa point, value is the weight matrix for that abscissa*/
    func_weight_mat_t m_functional_weights;

    /*!Number of statistical units*/
    std::size_t m_n;

    /*!Number of abscissa points of the functional object known*/
    std::size_t m_T;

    /*!Number of threads for OMP*/
    int m_number_threads;


public:

    /*!
    * @brief Constructor for the functional weight matrix
    * @param n number of statistical units
    * @param number_threads number of threads for OMP
    */
    functional_weight_matrix_base(std::size_t n, std::size_t T, int number_threads)
        : m_n(n), m_T(T), m_number_threads(number_threads)  {}

    /*!
    * @brief Getter for the weight matrix
    * @return the private m_data
    */
    func_weight_mat_t functional_weights() const {return m_functional_weights;}

    /*!
    * @brief Setter for the weight matrix
    * @return the private m_data
    */
    inline func_weight_mat_t & functional_weights() {return m_functional_weights;}

    /*!
    * @brief Getter for the number of statistical units
    * @return the private m_n
    */
    std::size_t n() const {return m_n;}

    /*!
    * @brief Getter for the number of abscissa points
    * @return the private m_T
    */
    std::size_t T() const {return m_T;}

    /*!
    * @brief Getter for the number of OMP threads
    * @return the private m_number_threads
    */
    std::size_t number_threads() const {return m_number_threads;}

};

#endif  /*FDAGWR_FUNC_WEIGHT_MATRIX_HPP*/