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


#ifndef FUNCTIONAL_MATRIX_INTO_WRAPPER_HPP
#define FUNCTIONAL_MATRIX_INTO_WRAPPER_HPP

#include "functional_matrix_storing_type.hpp"
#include "functional_matrix.hpp"
#include "functional_matrix_sparse.hpp"
#include "functional_matrix_diagonal.hpp"
#include "../utility/traits_fdagwr.hpp"



/*!
* @file functional_matrix_into_wrapper.hpp
* @brief Contains functions to wrap different functional objects into matrices of std::function
* @author Andrea Enrico Franzoni
*/



/*!
* @brief Function to wrap a functional data object functional_data into a column (row) vector intended as a functional matrix object functional_matrix
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @tparam domain_type geometry of basis domain
* @tparam basis_type basis type used for projecting the fdatum
* @param fd functional datum, with n statistical units
* @param number_threads number of threads for going in parallel with OMP
* @param as_column if true, the fdatum is represented in a functional_matrix nx1. Otherwise in a 1xn
* @return a functional matrix storing the fdatum as std::function, nx1 or 1xn
*/
template< typename INPUT = double, typename OUTPUT = double, class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>) && fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
inline
functional_matrix<INPUT,OUTPUT>
wrap_into_fm(const functional_data<domain_type,basis_type> &fd,
             int number_threads,
             bool as_column = true)
{
    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //std::function input type
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    std::vector< F_OBJ > f;
    f.resize(fd.n());

#ifdef _OPENMP
#pragma omp parallel for shared(fd) num_threads(number_threads)
#endif
    for(std::size_t unit_i = 0; unit_i < fd.n(); ++unit_i)
    {
        f[unit_i] = [unit_i,&fd](F_OBJ_INPUT x){return fd.eval(x,unit_i);};
    }

    if(as_column == true)
    {
        functional_matrix<INPUT,OUTPUT> fm(std::move(f),fd.n(),1);
        return fm;
    }
    else
    {
        functional_matrix<INPUT,OUTPUT> fm(std::move(f),1,fd.n());
        return fm;   
    }
}



/*!
* @brief Function to wrap a functional data covariates object functional_data_covariates into a functional matrix object functional_matrix
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @tparam domain_type geometry of basis domain
* @tparam basis_type basis type used for projecting the fdatum
* @param X functional data covariates, n statistical units and q covariates
* @param number_threads number of threads for going in parallel with OMP
* @return a matrix of functions, nxq
* @note It stores the functions objects column wise
*/
template< typename INPUT = double, typename OUTPUT = double, class domain_type = FDAGWR_TRAITS::basis_geometry, FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::STATIONARY >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>) && fdagwr_concepts::as_interval<domain_type>
inline
functional_matrix<INPUT,OUTPUT>
wrap_into_fm(const functional_data_covariates<domain_type,stationarity_t> &X,
             int number_threads)
{
    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //std::function input type
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    std::vector< F_OBJ > f;
    f.resize(X.n()*X.q());

#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(X) num_threads(number_threads)
#endif
    for (std::size_t unit_i = 0; unit_i < X.n(); ++unit_i) 
    {
        for (std::size_t cov_j = 0; cov_j < X.q(); ++cov_j) 
        {
            f[cov_j*X.n() + unit_i] = [unit_i,cov_j,&X](F_OBJ_INPUT x){return X.eval(x,cov_j,unit_i);};
        }
    }

    functional_matrix<INPUT,OUTPUT> fm(std::move(f),X.n(),X.q());
    return fm;
}



/*!
* @brief Function to wrap a basis system into a sparse functional matrix functional_matrix_sparse of dimension n x (n*L)
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @tparam domain_type geometry of basis domain
* @tparam basis_type basis type used for projecting the fdatum
* @param bs the basis system
* @param n the number of statistical units
* @param L the total number of basis
* @return a functional_matrix_sparse object, diagonal block matrix, each block of dimension 1xL, containing the basis of the of the basis system, that are constantly repeated
* @note the same basis system is repeated n times (n rows), each time as a single block. The first row contains the L basis, and then (n-1)*L 0s.
*       The second row starts at column L+1 (the first L containing 0s), contains the same L basis, and the 0s. And so on.
*       It is used for the representation of phi in the fwr algorithm.
*/
template< typename INPUT = double, typename OUTPUT = double, class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>) && fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
inline
functional_matrix_sparse<INPUT,OUTPUT>
wrap_into_fm(const basis_type<domain_type> &bs,
             std::size_t n,
             std::size_t L)
{
    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //std::function input type
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    //one row for each unit
    std::size_t rows = n;
    //the total number of cols is the product within the number of units and the number of basis
    std::size_t cols = n*L;
    //the construction of the matrix by block, only one element for each column
    std::size_t nnz = n*L;

    //containers storing elements and indices
    std::vector< F_OBJ > f;
    f.reserve(nnz);
    std::vector<std::size_t> row_idx;
    row_idx.reserve(nnz);
    std::vector<std::size_t> col_idx;
    col_idx.resize(cols + 1);
    std::iota(col_idx.begin(),col_idx.end(),static_cast<std::size_t>(0));   //cumulative number of elements in the cols is simply an increasing count of naturals

    //filling f
    for (std::size_t unit_i = 0; unit_i < n; ++unit_i){
        for (std::size_t l_i = 0; l_i < L; ++l_i){
            row_idx.emplace_back(unit_i);
            f.emplace_back([l_i,&bs](F_OBJ_INPUT x){return bs.eval_base(x)(0,l_i);});}}

    functional_matrix_sparse<INPUT,OUTPUT> fm(f,rows,cols,row_idx,col_idx);
    return fm;  
}



/*!
* @brief Function to wrap a collection of systems of basis into a sparse functional matrix functional_matrix_sparse
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @tparam domain_type geometry of basis domain
* @tparam basis_type basis type used for projecting the fdatum
* @param bs a collection of basis systems
* @return a functional sparse matrix, block diagonal, q x L. q is the number of basis systems, L the total number of basis. 
*         Each block has dimension 1xLj, where Lj is the number of basis of basis system j-th.
*/
template< typename INPUT = double, typename OUTPUT = double, class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>) && fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
inline
functional_matrix_sparse<INPUT,OUTPUT>
wrap_into_fm(const basis_systems<domain_type,basis_type> &bs)
{
    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //std::function input type
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    //one row for each covariate
    std::size_t rows = bs.q();
    //the total number of cols is the sum of the number of basis of all the covariates
    std::size_t cols = std::reduce(bs.numbers_of_basis().cbegin(),bs.numbers_of_basis().cend(),static_cast<std::size_t>(0));
    //the construction of the matrix by block, only one element for each column
    std::size_t nnz = cols;

    //containers storing elements and indices
    std::vector< F_OBJ > f;
    f.reserve(nnz);
    std::vector<std::size_t> row_idx;
    row_idx.reserve(nnz);
    std::vector<std::size_t> col_idx;
    col_idx.resize(cols + 1);
    std::iota(col_idx.begin(),col_idx.end(),static_cast<std::size_t>(0));   //cumulative number of elements in the cols is simply an increasing count of naturals

    //filling f
    for (std::size_t cov_i = 0; cov_i < bs.q(); ++cov_i){
        for(std::size_t base_j = 0; base_j < bs.numbers_of_basis()[cov_i]; ++base_j){
            //row cov_i-th contains a number of elements equal to the number of basis for the covariate cov_i-th
            row_idx.push_back(cov_i);
            //storing the basis accordingly to the type
            f.emplace_back([cov_i,base_j,&bs](F_OBJ_INPUT x){return bs.systems_of_basis()[cov_i].eval_base(x)(0,base_j);});}}
    
    functional_matrix_sparse<INPUT,OUTPUT> fm(f,rows,cols,row_idx,col_idx);
    return fm;
}



/*!
* @brief Function to wrap a functional stationary weight matrix object functional_weight_matrix_stationary into a functional diagonal matrix object functional_matrix_diagonal
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @tparam domain_type geometry of basis domain
* @tparam basis_type basis type used for projecting the fdatum
* @tparam stationarity_t if the weights are stationary
* @return a functional diagonal matrix, containing the n weights. n is the number of statistical units
*/
template< typename INPUT = double, typename OUTPUT = double, class domain_type = FDAGWR_TRAITS::basis_geometry,  template <typename> class basis_type = bsplines_basis, FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::STATIONARY >
    requires (std::integral<INPUT> || std::floating_point<INPUT>) && (std::integral<OUTPUT> || std::floating_point<OUTPUT>) && fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
inline
functional_matrix_diagonal<INPUT,OUTPUT>
wrap_into_fm(const functional_weight_matrix_stationary<INPUT,OUTPUT,domain_type,basis_type,stationarity_t> &W,
             int number_threads)
{
    static_assert(stationarity_t == FDAGWR_COVARIATES_TYPES::STATIONARY,
                  "Functional weight matrix for stationary covariates needs FDAGWR_COVARIATES_TYPES::STATIONARY as template parameter");

    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;

    std::vector< F_OBJ > f;
    std::size_t n = W.n();
    f.resize(n);

#ifdef _OPENMP
#pragma omp parallel for shared(n,W) num_threads(number_threads)
#endif
    for(std::size_t unit_i = 0; unit_i < n; ++unit_i)
    {
        f[unit_i] = W.weights()[unit_i];
    }

    functional_matrix_diagonal<INPUT,OUTPUT> fm(f,n);
    return fm;
}



/*!
* @brief Function to wrap a functional non-stationary weight matrix object functional_weight_matrix_stationary into a functional diagonal matrix object functional_matrix_diagonal
* @note It stores the functions objects diagonally, as an n x n matrix, where n is the number of statistical units
* @todo WRITE IT
*/
/*!
* @brief Function to wrap a functional non-stationary weight matrix object into a vector of diagonal functional matrices
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @tparam domain_type geometry of basis domain
* @tparam basis_type basis type used for projecting the fdatum
* @tparam stationarity_t if the weights are stationary
* @return an std::vector of size n of functional diagonal matrix, containing the n weights in the diagonal, in the position i-th, for unit i-th. n is the number of statistical units
*/
template< typename INPUT = double, typename OUTPUT = double, class domain_type = FDAGWR_TRAITS::basis_geometry,  template <typename> class basis_type = bsplines_basis, FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::NON_STATIONARY >
    requires (std::integral<INPUT> || std::floating_point<INPUT>) && (std::integral<OUTPUT> || std::floating_point<OUTPUT>) && fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
inline
std::vector< functional_matrix_diagonal<INPUT,OUTPUT> >
wrap_into_fm(const functional_weight_matrix_non_stationary<INPUT,OUTPUT,domain_type,basis_type,stationarity_t> &W,
             int number_threads)
{
    static_assert(stationarity_t == FDAGWR_COVARIATES_TYPES::NON_STATIONARY   ||
                  stationarity_t == FDAGWR_COVARIATES_TYPES::EVENT            ||
                  stationarity_t == FDAGWR_COVARIATES_TYPES::STATION,
                  "Functional weight matrix for non stationary covariates needs FDAGWR_COVARIATES_TYPES::NON_STATIONARY or FDAGWR_COVARIATES_TYPES::EVENT or FDAGWR_COVARIATES_TYPES::STATION as template parameter");

    //std::function object stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;

    std::size_t n_matrices = W.weights().size();
    std::size_t n = W.n();  //number of statistical units
    //storing all the functional diagonal matrices
    std::vector<functional_matrix_diagonal<INPUT,OUTPUT>> fm;
    fm.reserve(n_matrices);

    for(std::size_t i = 0; i < n_matrices; ++i)
    {
        std::vector< F_OBJ > f;
        f.resize(n);

#ifdef _OPENMP
#pragma omp parallel for shared(i,n,W) num_threads(number_threads)
#endif
        for(std::size_t unit_i = 0; unit_i < n; ++unit_i)
        {
            f[unit_i] = W.weights()[i][unit_i];
        }

        //fm.emplace_back(std::move(f),n);
        //fm.emplace_back(f,n);
        functional_matrix_diagonal<INPUT,OUTPUT> fm_i(f,n);
        fm.push_back(fm_i);
    }

    return fm;
}

#endif  /*FUNCTIONAL_MATRIX_INTO_WRAPPER_HPP*/