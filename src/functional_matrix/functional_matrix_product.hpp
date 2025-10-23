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


#ifndef FUNCTIONAL_MATRIX_PRODUCT_HPP
#define FUNCTIONAL_MATRIX_PRODUCT_HPP

#include "functional_matrix_storing_type.hpp"
#include "functional_matrix.hpp"
#include "functional_matrix_diagonal.hpp"
#include "functional_matrix_operators.hpp"

#include <Eigen/Dense>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <exception>


#ifdef _OPENMP
#include <omp.h>
#endif



/*!
* @file functional_matrix_product.hpp
* @brief Contains the overloaded functions to perform the matrix product within matrices of std::function
* @author Andrea Enrico Franzoni
*/


/*!
* @brief Matrix product within two functional matrices M1*M2
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param M1 functional matrix, m1xn1
* @param M2 functional matrix, m2xn2
* @param number_threads number of threads for going in parallel with OMP
* @return the functional matrix product, m1xn2
* @note n1 has to be equal to m2, otherwise an exception is thrown
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix<INPUT,OUTPUT> &M1,
        const functional_matrix<INPUT,OUTPUT> &M2,
        int number_threads)
{
    //checking matrices dimensions
    if (M1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //std::function input type
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;
    //initial point for f_sum
    F_OBJ f_null = [](F_OBJ_INPUT x){return static_cast<OUTPUT>(0);};
    //reducing operation for transform_reduce
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_sum = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)+f2(x);};};
    //binary operation for transform_reduce
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_prod = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)*f2(x);};};
    
    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(M1.rows(),M2.cols());

#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(M1,M2,prod,f_null,f_sum,f_prod) num_threads(number_threads)
#endif  
    for(std::size_t i = 0; i < prod.rows(); ++i){
        for(std::size_t j = 0; j < prod.cols(); ++j){
            //dot product within the row i-th of M1 and the col j-th of M2: using the views, access to row and cols is O(1)
            prod(i,j) = std::transform_reduce(M1.row(i).cbegin(),   
                                              M1.row(i).cend(),
                                              M2.col(j).cbegin(),
                                              f_null,                    //initial value (null function)
                                              f_sum,                     //reduce operation
                                              f_prod);}}                 //transform operation within the two ranges

    return prod;
}



/*!
* @brief Matrix product within two functional matrices M1*SM2, M1 dense and SM2 sparse
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param M1 functional matrix, m1xn1
* @param SM2 sparse functional matrix, m2xn2
* @return the functional matrix product, m1xn2
* @note n1 has to be equal to m2, otherwise an exception is thrown
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix<INPUT,OUTPUT> &M1,
        const functional_matrix_sparse<INPUT,OUTPUT> &SM2)
{
    //checking matrices dimensions
    if (M1.cols() != SM2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //std::function input type
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    //initial point for f_sum
    F_OBJ f_null = [](F_OBJ_INPUT x){return static_cast<OUTPUT>(0);};
    //reducing operation for transform_reduce
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_sum = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)+f2(x);};};
    //binary operation for transform_reduce
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_prod = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)*f2(x);};};

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(M1.rows(),SM2.cols(),f_null);


    //fixing the col in the sparse matrix, computing all the products with that columns
    for(std::size_t j = 0; j < prod.cols(); ++j){
        //if there is at least one element in col j-th        
        if(SM2.check_col_presence(j)){
            //computing all the products with col j-th
            for(std::size_t i = 0; i < prod.rows(); ++i){
                //products and summations are computed only with respect to the non-null elements
                for(std::size_t ii = SM2.cols_idx()[j]; ii < SM2.cols_idx()[j+1]; ++ii){
                    //row that contains an element in col j-th
                    std::size_t row_col_j = SM2.rows_idx()[ii];
                    prod(i,j) = f_sum(prod(i,j),f_prod(M1(i,row_col_j),SM2(row_col_j,j)));}}}}

    return prod;
}



/*!
* @brief Matrix product within two functional matrices SM1*M2, SM1 sparse and M2 dense
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param SM1 sparse functional matrix, m1xn1
* @param M2 functional matrix, m2xn2
* @return the functional matrix product, m1xn2
* @note n1 has to be equal to m2, otherwise an exception is thrown
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix_sparse<INPUT,OUTPUT> &SM1,
        const functional_matrix<INPUT,OUTPUT> &M2)
{
    //checking matrices dimensions
    if (SM1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //std::function input type
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    //initial point for f_sum
    F_OBJ f_null = [](F_OBJ_INPUT x){return static_cast<OUTPUT>(0);};
    //reducing operation for transform_reduce
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_sum = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)+f2(x);};};
    //binary operation for transform_reduce
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_prod = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)*f2(x);};};

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(SM1.rows(),M2.cols(),f_null);

    //loop su tutte le colonne di SM1 perchè la matrice sparsa va passata columnwise
    for(std::size_t j_s = 0; j_s < SM1.cols(); ++j_s){
        //loop sulle righe non-nulle della colonna j-th 
        for(std::size_t row_i_idx = SM1.cols_idx()[j_s]; row_i_idx < SM1.cols_idx()[j_s+1]; ++row_i_idx){
            //row not null
            std::size_t i = SM1.rows_idx()[row_i_idx];
            //updating step-by-step that element in the product
            for (std::size_t j = 0; j < prod.cols(); ++j){
                prod(i,j) = f_sum( prod(i,j), f_prod(SM1(i,j_s),M2(j_s,j)) );}}}

    return prod;
}




/*!
* @brief Matrix product within two functional matrices M1*D2, M1 dense and D2 diagonal
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param M1 functional matrix, m1xn1
* @param D2 diagonal functional matrix, nxn
* @param number_threads number of threads for going in parallel with OMP
* @return the functional matrix product, m1xn
* @note n1 has to be equal to n, otherwise an exception is thrown
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix<INPUT,OUTPUT> &M1,
        const functional_matrix_diagonal<INPUT,OUTPUT> &D2,
        int number_threads)
{
    //checking matrices dimensions
    if (M1.cols() != D2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //std::function input type
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;
    //function that operates the product within two functions
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_prod = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)*f2(x);};};

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(M1.rows(),D2.cols());

#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(M1,D2,prod,f_prod) num_threads(number_threads)
#endif 
    for (std::size_t i = 0; i < prod.rows(); ++i){
        for (std::size_t j = 0; j < prod.cols(); ++j){            
            prod(i,j) = f_prod(M1(i,j),D2(j,j));}}  //dense x diagonal: in prod, prod(i,j) = dense(i,j)*diagonal(j,j) 

    return prod;
}



/*!
* @brief Matrix product within two functional matrices D1*M2, D1 diagonal and M2 dense
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param D1 diagonal functional matrix, nxn
* @param M2 functional matrix, m2xn2
* @param number_threads number of threads for going in parallel with OMP
* @return the functional matrix product, nxn2
* @note n has to be equal to m2, otherwise an exception is thrown
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix_diagonal<INPUT,OUTPUT> &D1,
        const functional_matrix<INPUT,OUTPUT> &M2,
        int number_threads)
{
    //checking matrices dimensions
    if (D1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //std::function input type
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;
    //function that operates summation within two functions
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_prod = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)*f2(x);};};

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(D1.rows(),M2.cols());

#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(D1,M2,prod,f_prod) num_threads(number_threads)
#endif  
    for (std::size_t i = 0; i < prod.rows(); ++i){
        for (std::size_t j = 0; j < prod.cols(); ++j){            
            prod(i,j) = f_prod(D1(i,i),M2(i,j));}}  //diagonal x dense: in prod, prod(i,j) = diagonal(i,i)*dense(i,j)

    return prod;
}



/*!
* @brief Matrix product within two functional matrices D1*D2, D1 diagonal and D2 diagonal
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param D1 diagonal functional matrix, n1xn1
* @param D2 diagonal functional matrix, n2xn2
* @return the functional matrix product, n1xn2
* @note n1 has to be equal to n2, otherwise an exception is thrown
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix_diagonal<INPUT,OUTPUT>
fm_prod(const functional_matrix_diagonal<INPUT,OUTPUT> &D1,
        const functional_matrix_diagonal<INPUT,OUTPUT> &D2)
{
    //checking matrices dimensions
    if (D1.cols() != D2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //using ETs to perform the product
    return static_cast<functional_matrix_diagonal<INPUT,OUTPUT>>(D1*D2);
}



/*!
* @brief Matrix product within two functional matrices M1*S2, M1 dense and S2 scalar
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param M1 functional matrix, m1xn1
* @param S2 scalar matrix, m2xn2
* @param number_threads number of threads for going in parallel with OMP
* @return the functional matrix product, m1xn2
* @note n1 has to be equal to m2, otherwise an exception is thrown
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix<INPUT,OUTPUT> &M1,
        const Eigen::Matrix<OUTPUT,Eigen::Dynamic,Eigen::Dynamic> &S2,
        int number_threads)
{
    //checking matrices dimensions
    if (M1.cols() != S2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");    

    return fm_prod<INPUT,OUTPUT>(M1,scalar_to_functional<INPUT,OUTPUT>(S2),number_threads);
}



/*!
* @brief Matrix product within two functional matrices S1*M2, S1 scalar and M2 dense
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param S1 scalar matrix, m1xn1
* @param M2 functional matrix, m2xn2
* @param number_threads number of threads for going in parallel with OMP
* @return the functional matrix product, m1xn2
* @note n1 has to be equal to m2, otherwise an exception is thrown
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const Eigen::Matrix<OUTPUT,Eigen::Dynamic,Eigen::Dynamic> &S1,
        const functional_matrix<INPUT,OUTPUT> &M2,
        int number_threads)
{
    //checking matrices dimensions
    if (S1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    return fm_prod<INPUT,OUTPUT>(scalar_to_functional<INPUT,OUTPUT>(S1),M2,number_threads);
}



/*!
* @brief Matrix product within two functional matrices SM1*S2, SM1 sparse and S2 scalar
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param SM1 sparse function matrix, m1xn1
* @param S2 scalar matrix, m2xn2
* @return the functional matrix product, m1xn2
* @note n1 has to be equal to m2, otherwise an exception is thrown
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const functional_matrix_sparse<INPUT,OUTPUT> &SM1,
        const Eigen::Matrix<OUTPUT,Eigen::Dynamic,Eigen::Dynamic> &S2)
{
    //checking matrices dimensions
    if (SM1.cols() != S2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");    

    return fm_prod<INPUT,OUTPUT>(SM1,scalar_to_functional<INPUT,OUTPUT>(S2));
}



/*!
* @brief Matrix product within two functional matrices S1*SM2, S1 scalar and SM2 sparse
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param S1 scalar function matrix, m1xn1
* @param SM2 sparse functional matrix, m2xn2
* @return the functional matrix product, m1xn2
* @note n1 has to be equal to m2, otherwise an exception is thrown
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline
functional_matrix<INPUT,OUTPUT>
fm_prod(const Eigen::Matrix<OUTPUT,Eigen::Dynamic,Eigen::Dynamic> &S1,
        const functional_matrix_sparse<INPUT,OUTPUT> &SM2)
{
    //checking matrices dimensions
    if (S1.cols() != SM2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    return fm_prod<INPUT,OUTPUT>(scalar_to_functional<INPUT,OUTPUT>(S1),SM2);
}

#endif  /*FUNCTIONAL_MATRIX_PRODUCT_HPP*/