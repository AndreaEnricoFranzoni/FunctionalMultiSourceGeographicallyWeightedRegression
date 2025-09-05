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


#ifndef FUNCTIONAL_MATRIX_PRODUCT_HPP
#define FUNCTIONAL_MATRIX_PRODUCT_HPP

#include "functional_matrix_storing_type.hpp"
#include "functional_matrix.hpp"
#include "functional_matrix_diagonal.hpp"
#include "functional_matrix_operators.hpp"
#include "traits_fdagwr.hpp"


/*!
* @brief Row-by-col product within two functional matrices M1*M2
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_product(const functional_matrix<INPUT,OUTPUT> &M1,
           const functional_matrix<INPUT,OUTPUT> &M2,
           int number_threads)
{
    if (M1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(M1.rows(),M2.cols());

#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(M1,M2,prod) num_threads(number_threads)
    for (std::size_t i = 0; i < prod.rows(); ++i){
        for (std::size_t j = 0; j < prod.cols(); ++j){            
            prod(i,j) = static_cast<functional_matrix<INPUT,OUTPUT>>(M1.get_row(i)*(M2.get_col(j).transpose())).reduce();}}   //static_cast allows to use immediately .reduce() method
#endif       

    return prod;
}


#endif  /*FUNCTIONAL_MATRIX_PRODUCT_HPP*/