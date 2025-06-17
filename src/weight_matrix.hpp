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


#ifndef FDAGWR_WEIGHT_MATRIX_HPP
#define FDAGWR_WEIGHT_MATRIX_HPP

#include "traits_fdagwr.hpp"
#include "kernel_functions.hpp"


#ifdef _OPENMP
#include <omp.h>
#endif


/*!
* @file weight_matrix.hpp
* @brief Construct the weight matrix for performing the geographically weighted regression
* @author Andrea Enrico Franzoni
*/



/*!
* Doing tag dispatching for the correct way of evaluating the non stationary weights (kernel function for the distances)
* @tparam err_eval: template parameter for the error evaluation strategy
*/
template <KERNEL_FUNC kernel_func>
using KERNEL_FUNC_T = std::integral_constant<KERNEL_FUNC, kernel_func>;



/*!
* @class weight_matrix_base
* @brief Template class for constructing the weight matrix: a squared matrix containing the weight for each unit
* @tparam D type of the derived class (for static polymorphism thorugh CRTP):
*         - stationary: 'weight_matrix_stat'
*         - non stationary: 'weight_matrix_no_stat'
* @tparam kernel_func kernel function for the evaluation of the weights (enumerator)
* @details It is the base class. Polymorphism is known at compile time thanks to Curiously Recursive Template Pattern (CRTP) 
*/
template< class D, KERNEL_FUNC kernel_func >
class weight_matrix_base
{

private:
    /*!Matrix storing the weights in the diagonal*/
    fdagwr_traits::Sparse_Matrix m_weights;

    /*!Number of statistical units*/
    std::size_t m_n;

    /*!Number of threads for OMP*/
    int m_number_threads;

    /*!
    * @brief Evaluation of the kernel function for the non stationary weights
    * @param distance distance between two locations
    * @param bandwith kernel bandwith
    * @return the evaluation of the kernel function
    */
    double kernel_eval(double distance, double bandwith, KERNEL_FUNC_T<KERNEL_FUNC::GAUSSIAN>) const;


public:

    /*!
    * @brief Constructor for the weight matrix (diagonal matrix containing the weight for each unit)
    * @param n number of statistical units
    * @param number_threads number of threads for OMP
    */
    weight_matrix_base(std::size_t n, int number_threads)
        : m_weights(n,n), m_n(n), m_number_threads(number_threads)  {}

    /*!
    * @brief Getter for the weight matrix
    * @return the private m_data
    */
    fdagwr_traits::Sparse_Matrix weights() const {return m_weights;}

    /*!
    * @brief Setter for the weight matrix
    * @return the private m_data
    */
    inline fdagwr_traits::Sparse_Matrix & weights() {return m_weights;}

    /*!
    * @brief Getter for the number of statistical units
    * @return the private m_n
    */
    std::size_t n() const {return m_n;}

    /*!
    * @brief Setter for the number of OMP threads
    * @return the private m_n
    */
    std::size_t number_threads() const {return m_number_threads;}

    /*!
    * @brief Evaluation of kernel function for the non-stationary weights. Tag-dispacther.
    * @param distance distance between two locations
    * @param bandwith kernel bandwith
    * @return the evaluation of the kernel function
    */
    double kernel_eval(double distance, double bandwith) const { return kernel_eval(distance,bandwith,KERNEL_FUNC_T<kernel_func>{});};
 
};

#include "kernel_functions_eval.hpp"

#endif  /*FDAGWR_WEIGHT_MATRIX_HPP*/