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


#ifdef _OPENMP
#include <omp.h>
#endif


/*!
* @file weight_matrix.hpp
* @brief Construct the weight matrix for performing the geographically weighted regression
* @author Andrea Enrico Franzoni
*/


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
    fdagwr_traits::Sparse_Matrix m_data;

    /*!Number of statistical units*/
    std::size_t m_n;

    /*!Number of threads for OMP*/
    int m_number_threads;


public:

    /*!
    * @brief Constructor for the weight matrix (diagonal matrix containing the weight for each unit)
    * @param data stationary weight, for each statistical unit
    * @param n number of statistical units
    * @param number_threads number of threads for OMP
    */
    weight_matrix_base(std::size_t n, int number_threads)
        : m_data(n,n), m_n(n), m_number_threads(number_threads)  {}

    /*!
    * @brief Getter for the weight matrix
    * @return the private m_data
    */
    fdagwr_traits::Sparse_Matrix data() const {return m_data;}

    /*!
    * @brief Getter for the number of statistical units
    * @return the private m_n
    */
    std::size_t n() const {return m_n;}

};

#endif  /*FDAGWR_WEIGHT_MATRIX_HPP*/