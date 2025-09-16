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
#include <numeric>
#include <algorithm>
#include <iterator>
#include <exception>
#include <iostream>


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
    //checking matrices dimensions
    if (M1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //type stored by the functional matrix
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //input type of the elements of the functional matrix
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
    for(std::size_t i = 0; i < prod.rows(); ++i){
        for(std::size_t j = 0; j < prod.cols(); ++j){
            //dot product within the row i-th of M1 and the col j-th of M2: using the views, access to row and cols is O(1)
            prod(i,j) = std::transform_reduce(M1.row(i).cbegin(),   
                                              M1.row(i).cend(),
                                              M2.col(j).cbegin(),
                                              f_null,                    //initial value (null function)
                                              f_sum,                     //reduce operation
                                              f_prod);}}                 //transform operation within the two ranges
#endif  

    return prod;
}



/*!
* @brief Row-by-col product within a functional matrices M1 and a sparse functional matrix M2
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

    //type stored by the functional matrix
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //input type of the elements of the functional matrix
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    //initial point for f_sum
    F_OBJ f_null = [](F_OBJ_INPUT x){return static_cast<OUTPUT>(0);};
    //reducing operation for transform_reduce
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_sum = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)+f2(x);};};
    //binary operation for transform_reduce
    std::function<F_OBJ(F_OBJ,F_OBJ)> f_prod = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)*f2(x);};};

    //resulting matrix
    functional_matrix<INPUT,OUTPUT> prod(M1.rows(),SM2.cols(),f_null);

    for(std::size_t j = 0; j < prod.cols(); ++j){
        //the number of elements in the col j-th of the sparse matrix
        std::size_t start_col_j = SM2.cols_idx()[j];
        std::size_t end_col_j = SM2.cols_idx()[j+1];
        
        if(start_col_j <  end_col_j){
            for(std::size_t i = 0; i < prod.rows(); ++i){
                //for each element, making the product looping only of the non null elements
                //for(auto non_null_row = std::next(SM2.rows_idx().cbegin(),start_col_j); non_null_row != std::next(SM2.rows_idx().cbegin(),end_col_j); ++non_null_row){
                //    prod(i,j) = f_sum(prod(i,j),f_prod(M1(i,*non_null_row),SM2(*non_null_row,j)));}
                
                //std::for_each(std::next(SM2.rows_idx().cbegin(),start_col_j),
                //              std::next(SM2.rows_idx().cbegin(),end_col_j),
                //              [&prod,i,j,&M1,&SM2,&f_sum,&f_prod](const std::size_t &row_no_null){ prod(i,j) = f_sum(prod(i,j),f_prod(M1(i,row_no_null),SM2(row_no_null,j)));});

                std::for_each(std::next(SM2.rows_idx().cbegin(),start_col_j),
                              std::next(SM2.rows_idx().cbegin(),end_col_j),
                              [i,j](const std::size_t &row_no_null_idx){std::cout << "A prod(" << i <<","<<j<<") concorre l'elemento (" << i << ","<<row_no_null_idx<<") della densa e ("<<row_no_null_idx<<","<<j<<") della sparsa"<<std::endl;});

/*
                std::cout << "i=" << i << ", j=" << j <<": start_col_j: " << start_col_j << ", end_col_j: " << end_col_j << std::endl;
                auto f_ = std::next(SM2.rows_idx().cbegin(),start_col_j);
                std::cout << "Inizio: " << SM2((*f_),j)(0.3) << std::endl;
                auto f__ = std::next(SM2.rows_idx().cbegin(),end_col_j - 1);
                std::cout << "Fine: " << SM2((*f__),j)(0.3) << std::endl;
*/
                }
        }
    }

    return prod;
}



/*!
* @brief Row-by-col product within a functional matrices M1 and a sparse functional matrix SM2
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

    //type stored by the functional matrix
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //input type of the elements of the functional matrix
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
        //the number of elements in the col j-th of the sparse matrix
        std::size_t start_col_j = SM1.cols_idx()[j_s];
        std::size_t end_col_j = SM1.cols_idx()[j_s+1];
        //loop sulle righe non-nulle della colonna j-th 
        for(auto non_null_row = std::next(SM1.rows_idx().cbegin(),start_col_j); non_null_row != std::next(SM1.rows_idx().cbegin(),end_col_j); ++non_null_row){
            //cosa vado ad aggiornare nel prodotto? In corrispondenza delle riga non nulla non_null_row-th,
            //devo fare un ulteriore loop sulle colonne di M2, in cui vado a fare i prodotti singoli, sommando verso la fine
            for (std::size_t j = 0; j < prod.cols(); ++j){
                //actual products
                prod(*non_null_row,j) = f_sum( prod(*non_null_row,j), f_prod(SM1(*non_null_row,j_s),M2(j_s,j)) );}}}

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
    //checking matrices dimensions
    if (M1.cols() != D2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //type stored by the functional matrix
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //input type of the elements of the functional matrix
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
    //checking matrices dimensions
    if (D1.cols() != M2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //type stored by the functional matrix
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //input type of the elements of the functional matrix
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
    //checking matrices dimensions
    if (D1.cols() != D2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");

    //using ETs to perform the product
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
        const Eigen::Matrix<OUTPUT,Eigen::Dynamic,Eigen::Dynamic> &S2,
        int number_threads)
{
    //checking matrices dimensions
    if (M1.cols() != S2.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for functional matrix product");    

    return fm_prod<INPUT,OUTPUT>(M1,scalar_to_functional<INPUT,OUTPUT>(S2),number_threads);
}



/*!
* @brief Row-by-col product within a scalar matrix M1 and a functional matrix M2
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

#endif  /*FUNCTIONAL_MATRIX_PRODUCT_HPP*/