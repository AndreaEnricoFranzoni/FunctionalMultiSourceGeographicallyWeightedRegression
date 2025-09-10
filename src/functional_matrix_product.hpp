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

#include <Eigen/Dense>
#include <numberic>
#include <algorithm>
#include <iterator>


#ifdef _OPENMP
#include <omp.h>
#endif


/*!
* @brief Row-by-col product within two functional matrices M1*M2
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix<INPUT,OUTPUT> &M1,
        const functional_matrix<INPUT,OUTPUT> &M2,
        int number_threads)
{
    std::cout << "Dense x dense second version" << std::endl;
    if (M1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(M1.rows(),M2.cols());

    F_OBJ f_null = [](F_OBJ_INPUT x){return static_cast<OUTPUT>(0);};
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_sum = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)+f2(x);};};
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_prod = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)*f2(x);};};

/*
#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(M1,M2,prod) num_threads(number_threads)
    for (std::size_t i = 0; i < prod.rows(); ++i){
        for (std::size_t j = 0; j < prod.cols(); ++j){            
            prod(i,j) = static_cast<functional_matrix<INPUT,OUTPUT>>(M1.row(i)*(M2.col(j).transpose())).reduce();}}   //static_cast allows to use immediately .reduce() method
#endif       
*/
    for(std::size_t i = 0; i < prod.rows(); ++i){
        for(std::size_t j = 0; j < prod.cols(); ++j){
            prod(i,j) = std::transform_reduce(M1.get_row(i).cbegin(),   //dot product within the row i-th of M1 and the col j-th of M2
                                              M1.get_row(i).cend(),
                                              M2.get_col(j).cbegin(),
                                              f_null,                                    //starting point
                                              f_sum,          //reduce operation
                                              f_prod);}}       //transform operation within the two ranges

    return prod;
}



/*!
* @brief Row-by-col product within a functional matrix M1 and a diagoanl functional matrix M2
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix<INPUT,OUTPUT> &M1,
        const functional_matrix_diagonal<INPUT,OUTPUT> &D2,
        int number_threads)
{
    if (M1.cols() != D2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    //function that operates the product within two functions
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_prod = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)*f2(x);};};

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(M1.rows(),D2.cols());

#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(M1,D2,prod,f_prod) num_threads(number_threads)
    for (std::size_t i = 0; i < prod.rows(); ++i){
        for (std::size_t j = 0; j < prod.cols(); ++j){            
            prod(i,j) = f_prod(M1(i,j),D2(j,j));}}  //dense x diagonal: in prod, prod(i,j) = dense(i,j)*diagonal(j,j)
#endif  

    return prod;
}



/*!
* @brief Row-by-col product within a diagonal functional matrix M1 and a functional matrix M2
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix_diagonal<INPUT,OUTPUT> &D1,
        const functional_matrix<INPUT,OUTPUT> &M2,
        int number_threads)
{
    if (D1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    //function that operates summation within two functions
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_prod = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)*f2(x);};};

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(D1.rows(),M2.cols());

#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(D1,M2,prod,f_prod) num_threads(number_threads)
    for (std::size_t i = 0; i < prod.rows(); ++i){
        for (std::size_t j = 0; j < prod.cols(); ++j){            
            prod(i,j) = f_prod(D1(i,i),M2(i,j));}}  //diagonal x dense: in prod, prod(i,j) = diagonal(i,i)*dense(i,j)
#endif  

    return prod;
}



/*!
* @brief Row-by-col product within two diagonal functional matrices D1*D2
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix_diagonal<INPUT,OUTPUT>
fm_prod(const functional_matrix_diagonal<INPUT,OUTPUT> &D1,
        const functional_matrix_diagonal<INPUT,OUTPUT> &D2)
{
    if (D1.cols() != D2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    return static_cast<functional_matrix_diagonal<INPUT,OUTPUT>>(D1*D2);
}



/*!
* @brief Row-by-col product within a functional matrix M1 and a scalar matrix M2
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix<INPUT,OUTPUT> &M1,
        const Eigen::MatrixXd &M2,
        int number_threads)
{
    
    if (M1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //converting the scalar matrix into one of constant functions
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;
    std::function<F_OBJ(const double &)> scalar_to_const_f = [](const double &a){return [a](F_OBJ_INPUT x){return static_cast<OUTPUT>(a);};};

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(M1.rows(),M2.cols());
    
#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(M1,M2,prod) num_threads(number_threads)
    for (std::size_t j = 0; j < prod.cols(); ++j){
        //transforming the j-th col of the scalar matrix into a row vector (coherent for ETs on fm) of functional objects
        std::vector<F_OBJ> scalar_f_vec;
        scalar_f_vec.resize(M2.rows());
        std::transform(M2.col(j).cbegin(),
                       M2.col(j).cend(),
                       scalar_f_vec.begin(),
                       scalar_to_const_f);      //iterators on Eigen::MatrixXd traverse M2 column-wise (coherent with how elements are stored into a functional_matrix)
        functional_matrix<INPUT,OUTPUT> col_j(scalar_f_vec,1,M2.rows());
        
        for (std::size_t i = 0; i < prod.rows(); ++i){    
            prod(i,j) = static_cast<functional_matrix<INPUT,OUTPUT>>(M1.row(i)*col_j).reduce();}}   //static_cast allows to use immediately .reduce() method
#endif        

    return prod;
}



/*!
* @brief Row-by-col product within a scalar matrix M1 and a functional matrix M2
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const Eigen::MatrixXd &M1,
        const functional_matrix<INPUT,OUTPUT> &M2,
        int number_threads)
{
    
    if (M1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //converting the scalar matrix into one of constant functions
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;
    std::function<F_OBJ(const double &)> scalar_to_const_f = [](const double &a){return [a](F_OBJ_INPUT x){return static_cast<OUTPUT>(a);};};

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(M1.rows(),M2.cols());
    
#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(M1,M2,prod) num_threads(number_threads)
    for (std::size_t i = 0; i < prod.rows(); ++i){
        //transforming the i-th row of the scalar matrix into a row vector (already made a column-vector for ETs on fm) of functional objects
        std::vector<F_OBJ> scalar_f_vec;
        scalar_f_vec.resize(M1.cols());
        std::transform(M1.row(i).cbegin(),
                       M1.row(i).cend(),
                       scalar_f_vec.begin(),
                       scalar_to_const_f);      //iterators on Eigen::MatrixXd traverse M2 column-wise (coherent with how elements are stored into a functional_matrix)
        functional_matrix<INPUT,OUTPUT> row_i(scalar_f_vec,M2.rows(),1);
        
        for (std::size_t j = 0; j < prod.cols(); ++j){    
            prod(i,j) = static_cast<functional_matrix<INPUT,OUTPUT>>(row_i*M2.col(j)).reduce();}}   //static_cast allows to use immediately .reduce() method
#endif        

    return prod;
}

#endif  /*FUNCTIONAL_MATRIX_PRODUCT_HPP*/