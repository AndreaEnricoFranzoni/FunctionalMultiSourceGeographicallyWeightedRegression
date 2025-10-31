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


#include "fwr_operator_computing.hpp"



/*!
* @file fwr_operator_computing_imp.hpp
* @brief Contains the implementation of the class for computing operators needed in FWR, wrapping and dewrapping the basis expansion coefficients of the functional regression coefficients, evaluating the latters 
* @author Andrea Enrico Franzoni
*/



/*!
* @brief Compute [J_i + R]^(-1), where, for each unit i-th:
*        - J_i = int_a_b(base_t * X_t * W_i * X * base), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
*        - R the penalization matrix, diagonal block-matrix, such that each block contains the inner product within a given derivative of basis systems, one block for each basis system
* @param base_t containing basis systems, one system for each row, one basis for each column, transpost
* @param X_t functional covariates, transpost
* @param W vector of functional diagonal matrices, containing, for each unit, functional weights
* @param X functioal covaraites
* @param base containing basis systems, one system for each row, one basis for each column
* @return vector with the partial PivLU of each J_i + R
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> >
fwr_operator_computing<INPUT,OUTPUT>::compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                                                      const functional_matrix<INPUT,OUTPUT> &X_t,
                                                      const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                                      const functional_matrix<INPUT,OUTPUT> &X,
                                                      const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                                      const FDAGWR_TRAITS::Sparse_Matrix &R)
const
{
    //the vector contains factorization of the matrix
    std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > penalty;
    penalty.resize(W.size());

    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);   //necessary to compute the sum later

#ifdef _OPENMP
#pragma omp parallel for shared(penalty,base,base_t,X,X_t,W,_R_,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {
        //dimension: L x L, where L is the number of basis
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(fm_prod(base_t,X_t),W[i],m_number_threads),X,m_number_threads),base);

        //performing integration and factorization
        penalty[i] = Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >( fm_integration(integrand) + _R_ );    
        // penalty[i].solve(M) equivale a fare elemento penalty[i], che è una matrice inversa, times M
    }
    
    return penalty;
}

/*!
* @brief Compute [J_i + R]^(-1), where, for each unit i-th:
*        - J_i = int_a_b(X_crossed_t* W_i * X_crossed), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
*        - R the penalization matrix, diagonal block-matrix, such that each block contains the inner product within a given derivative of basis systems, one block for each basis system
* @param X_crossed_t containing functional covariates from which the project of the previous part of the model estimation is taken out, transpost
* @param W vector of functional diagonal matrices, containing, for each unit, functional weights
* @param X_crossed containing functional covariates from which the project of the previous part of the model estimation is taken out
* @param R penalization matrix
* @return vector with the partial PivLU of each J_i + R
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > >
fwr_operator_computing<INPUT,OUTPUT>::compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                                                      const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                                      const functional_matrix<INPUT,OUTPUT> &X_crossed,
                                                      const FDAGWR_TRAITS::Sparse_Matrix &R) 
const
{
    //the vector contains factorization of the matrix
    std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > penalty;
    penalty.resize(W.size());

    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);   //necessary to compute the sum later

#ifdef _OPENMP
#pragma omp parallel for shared(penalty,X_crossed_t,X_crossed,W,_R_,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {
        //dimension: L x L, where L is the number of basis
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_crossed_t,W[i],m_number_threads),X_crossed,m_number_threads);

        //performing integration and factorization
        penalty[i] = Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >( fm_integration(integrand) + _R_ );    
        // penalty[i].solve(M) equivale a fare elemento penalty[i], che è una matrice inversa, times M
    }
    
    return penalty;
}

/*!
* @brief Compute [J + R]^(-1), where:
*        - J = int_a_b(X_crossed_t* W * X_crossed), where W is the functional weight. The integral of the matrix is element-wise
*        - R the penalization matrix, diagonal block-matrix, such that each block contains the inner product within a given derivative of basis systems, one block for each basis system
* @param X_crossed_t containing functional covariates from which the project of the previous part of the model estimation is taken out, transpost
* @param W functional diagonal matrix, containing functional weights
* @param X_crossed containing functional covariates from which the project of the previous part of the model estimation is taken out
* @param R penalization matrix
* @return partial PivLU of J + R
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >
fwr_operator_computing<INPUT,OUTPUT>::compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                                                      const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                                                      const functional_matrix<INPUT,OUTPUT> &X_crossed,
                                                      const FDAGWR_TRAITS::Sparse_Matrix &R) 
const
{
    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);
    functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_crossed_t,W,m_number_threads),X_crossed,m_number_threads);

    //performing integration and factorization
    return Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >( fm_integration(integrand) + _R_ ); 
}

/*!
* @brief Compute [J + R]^(-1), where:
*        - J = int_a_b(base_t * X_t* W * X * base), where W is the functional weight. The integral of the matrix is element-wise
*        - R the penalization matrix, diagonal block-matrix, such that each block contains the inner product within a given derivative of basis systems, one block for each basis system
* @param base_t containing basis systems, one system for each row, one basis for each column, transpost
* @param X_t functional covariates, transpost
* @param W functional diagonal matrix, containing functional weights
* @param X functioal covaraites
* @param base containing basis systems, one system for each row, one basis for each column
* @param R penalization matrix
* @return partial PivLU of J + R
* @note is used when having only stationary covariates
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >
fwr_operator_computing<INPUT,OUTPUT>::compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                                                      const functional_matrix<INPUT,OUTPUT> &X_t,
                                                      const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                                                      const functional_matrix<INPUT,OUTPUT> &X,
                                                      const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                                      const FDAGWR_TRAITS::Sparse_Matrix &R)
const
{   
    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);
    functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(fm_prod(base_t,X_t),W,m_number_threads),X,m_number_threads),base);

    //performing integration and factorization
    return Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >( this->fm_integration(integrand) + _R_ ); 
}

/*!
* @brief Compute a scalar (matrix) operator as, for each unit
*        int_a_b(base_lhs * X_lhs * W_i * X_rhs * base_rhs), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
* @param base_lhs containing basis systems, one system for each row, one basis for each column
* @param X_lhs functional data matrix
* @param W vector of functional diagonal matrices, containing, for each unit, functional weights
* @param X_rhs functional data matrix
* @param base_rhs containing basis systems, one system for each row, one basis for each column
* @param penalty vector of partial PivLU decomposition, containing the penalties [J_i + R]^(-1) as computed by the functions above
* @return a matrix containing the element-wise integration
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fwr_operator_computing<INPUT,OUTPUT>::compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                                                       const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                                       const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                                       const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                                       const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                                                       const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > _operator_;
    _operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(_operator_,penalty,base_lhs,X_lhs,X_rhs,base_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(fm_prod(base_lhs,X_lhs),W[i],m_number_threads),X_rhs,m_number_threads),base_rhs);

        //performing integration and multiplication with the penalty (inverse factorized)
        _operator_[i] = penalty[i].solve( fm_integration(integrand) ); 
    }

    return _operator_;
}

/*!
* @brief Compute a scalar (matrix) operator as, for each unit
*        [J_i + R]^(-1) * int_a_b(base_lhs * X_lhs * W_i * base_rhs), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
* @param base_lhs containing basis systems, one system for each row, one basis for each column
* @param X_lhs functional data matrix
* @param W vector of functional diagonal matrices, containing, for each unit, functional weights
* @param base_rhs containing basis systems, one system for each row, one basis for each column
* @param penalty vector of partial PivLU decomposition, containing the penalties [J_i + R]^(-1) as computed by the functions above
* @return a matrix containing the element-wise integration
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fwr_operator_computing<INPUT,OUTPUT>::compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                                                       const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                                       const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                                       const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                                                       const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > _operator_;
    _operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(_operator_,penalty,base_lhs,X_lhs,base_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(base_lhs,X_lhs),W[i],m_number_threads),base_rhs);

        //performing integration and multiplication with the penalty (inverse factorized)
        _operator_[i] = penalty[i].solve( fm_integration(integrand) ); 
    }

    return _operator_;
}

/*!
* @brief Compute a scalar (matrix) operator as, for each unit
*        [J_i + R]^(-1) * int_a_b(X_lhs * W_i * X_rhs * base_rhs), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
* @param X_lhs functional data matrix
* @param W vector of functional diagonal matrices, containing, for each unit, functional weights
* @param X_rhs functional data matrix
* @param base_rhs containing basis systems, one system for each row, one basis for each column
* @param penalty vector of partial PivLU decomposition, containing the penalties [J_i + R]^(-1) as computed by the functions above
* @return a matrix containing the element-wise integration
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fwr_operator_computing<INPUT,OUTPUT>::compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                                       const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                                       const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                                       const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                                                       const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > _operator_;
    _operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(_operator_,penalty,X_lhs,X_rhs,base_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(X_lhs,W[i],m_number_threads),X_rhs,m_number_threads),base_rhs);

        //performing integration and multiplication with the penalty (inverse factorized)
        _operator_[i] = penalty[i].solve( fm_integration(integrand) ); 
    }

    return _operator_;
}

/*!
* @brief Compute a scalar (matrix) operator as, for each unit
*        [J_i + R]^(-1) * int_a_b(X_lhs * W_i * base_rhs), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
* @param X_lhs functional data matrix
* @param W vector of functional diagonal matrices, containing, for each unit, functional weights
* @param base_rhs containing basis systems, one system for each row, one basis for each column
* @param penalty vector of partial PivLU decomposition, containing the penalties [J_i + R]^(-1) as computed by the functions above
* @return a matrix containing the element-wise integration
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fwr_operator_computing<INPUT,OUTPUT>::compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                                       const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                                       const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                                                       const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > _operator_;
    _operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(_operator_,penalty,X_lhs,base_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_lhs,W[i],m_number_threads),base_rhs);

        //performing integration and multiplication with the penalty (inverse factorized)
        _operator_[i] = penalty[i].solve( fm_integration(integrand) ); 
    }

    return _operator_;
}

/*!
* @brief Compute a scalar (matrix) operator as, for each unit
*        [J_i + R]^(-1) * int_a_b(X_lhs * W_i * X_rhs), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
* @param X_lhs functional data matrix
* @param W vector of functional diagonal matrices, containing, for each unit, functional weights
* @param X_rhs functional data matrix
* @param penalty vector of partial PivLU decomposition, containing the penalties [J_i + R]^(-1) as computed by the functions above
* @return a matrix containing the element-wise integration
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fwr_operator_computing<INPUT,OUTPUT>::compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                                       const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                                       const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                                       const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > _operator_;
    _operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(_operator_,penalty,X_lhs,X_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_lhs,W[i],m_number_threads),X_rhs,m_number_threads);

        //performing integration and multiplication with the penalty (inverse factorized)
        _operator_[i] = penalty[i].solve( fm_integration(integrand) ); 
    }

    return _operator_;
}

/*!
* @brief Compute a scalar (matrix) operator as, for each unit
*        [J_i + R]^(-1) * int_a_b(base_lhs * X_lhs * W_i * X_rhs), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
* @param base_lhs containing basis systems, one system for each row, one basis for each column
* @param X_lhs functional data matrix
* @param W vector of functional diagonal matrices, containing, for each unit, functional weights
* @param X_rhs functional data matrix
* @param penalty vector of partial PivLU decomposition, containing the penalties [J_i + R]^(-1) as computed by the functions above
* @return a matrix containing the element-wise integration
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fwr_operator_computing<INPUT,OUTPUT>::compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                                                       const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                                       const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                                       const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                                       const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > _operator_;
    _operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(_operator_,penalty,base_lhs,X_lhs,X_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(base_lhs,X_lhs),W[i],m_number_threads),X_rhs,m_number_threads);

        //performing integration and multiplication with the penalty (inverse factorized)
        _operator_[i] = penalty[i].solve( fm_integration(integrand) ); 
    }

    return _operator_;
}

/*!
* @brief Compute a scalar (matrix) operator as [J + R]^(-1) * int_a_b(X_lhs * W_i * X_rhs), where W is the functional weight. The integral of the matrix is element-wise
* @param X_lhs functional data matrix
* @param W functional diagonal matrix, containing functional weights
* @param X_rhs functional data matrix
* @param penalty partial PivLU decomposition, containing the penalty [J + R]^(-1) as computed by the functions above
* @return a matrix containing the element-wise integration
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
FDAGWR_TRAITS::Dense_Matrix
fwr_operator_computing<INPUT,OUTPUT>::compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                                       const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                                                       const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                                       const Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > &penalty) 
const
{
    //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
    functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_lhs,W,m_number_threads),X_rhs,m_number_threads);

    //performing integration and multiplication with the penalty (inverse factorized)
    return penalty.solve( fm_integration(integrand) ); 
}

/*!
* @brief Compute a scalar (matrix) operator as [J + R]^(-1) * int_a_b(base_lhs * X_lhs * W_i * X_rhs), where W_i is the functional weight. The integral of the matrix is element-wise
* @param base_lhs containing basis systems, one system for each row, one basis for each column
* @param X_lhs functional data matrix
* @param W functional diagonal matrix, containing functional weights
* @param X_rhs functional data matrix
* @param penalty partial PivLU decomposition, containing the penalty [J + R]^(-1) as computed by the functions above
* @return a matrix containing the element-wise integration
* @note for FWR when having only stationary covariates
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
FDAGWR_TRAITS::Dense_Matrix
fwr_operator_computing<INPUT,OUTPUT>::compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                                                        const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                                        const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                                                        const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                                        const Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > &penalty) 
const
{
    //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
    functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(base_lhs,X_lhs),W,m_number_threads),X_rhs,m_number_threads);

    //performing integration and multiplication with the penalty (inverse factorized)
    return penalty.solve( fm_integration(integrand) ); 
}

/*!
* @brief Compute a functional operator, intended as a matrix where:
*        each row is the respective row of the functional covaraite, multiplied by the base system, and then by the operator computed
* @param X functional data covariates
* @param base basis system
* @param _operator_ an operator computed as above
* @return the functional operator, as functional matrix       
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
functional_matrix<INPUT,OUTPUT> 
fwr_operator_computing<INPUT,OUTPUT>::compute_functional_operator(const functional_matrix<INPUT,OUTPUT> &X,
                                                                   const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                                                   const std::vector< FDAGWR_TRAITS::Dense_Matrix > &_operator_) 
const
{
    //number of rows of the functional operator
    std::size_t m = X.rows();
    //number of cols of the functional operator
    std::size_t n = _operator_[0].cols();
    //result
    functional_matrix<INPUT,OUTPUT> func_operator(m,n);

    functional_matrix<INPUT,OUTPUT> x_times_base = fm_prod(X,base);

#ifdef _OPENMP
#pragma omp parallel for shared(func_operator,x_times_base,_operator_,m,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < m; ++i){
        //trnasforming the scalar matrix into a functional one, with constant functions
        std::vector< FUNC_OBJ<INPUT,OUTPUT> > row_i_v(x_times_base.row(i).cbegin(),x_times_base.row(i).cend());
        functional_matrix<INPUT,OUTPUT> row_i(row_i_v,1,base.cols());
        functional_matrix<INPUT,OUTPUT> row_i_prod = fm_prod(row_i,_operator_[i],m_number_threads);
        func_operator.row_sub(row_i_prod.as_vector(),i);
    }

    return func_operator;
}

/*!
* @brief For stationary covariates, wrap the basis expansion coefficients of the functional regression coefficients, from column vector containing all the the coefficients to a vector containing only the coefficients for the covariate
* @param b a matrix, column matrix, of dimension Lx1, containing the basis expansion coefficients of the functional regression coefficients, one covariate after the other
* @param L_j vector containing the number of basis of each regression coefficient. Its sum is L
* @param q number of covariates
* @return a vector of matrices, the i-th matrix of dimension L_j[i]x1, with the basis expansion coefficients of the regression coefficient of the i-th stationary covariate
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fwr_operator_computing<INPUT,OUTPUT>::wrap_operator(const FDAGWR_TRAITS::Dense_Matrix& b,
                                                     const std::vector<std::size_t>& L_j,
                                                     std::size_t q)
const
{
    //input coherency
    assert((L_j.size() == q) && (b.cols() == 1) && (b.rows() == std::reduce(L_j.cbegin(),L_j.cend(),static_cast<std::size_t>(0))));
    //container
    std::vector< FDAGWR_TRAITS::Dense_Matrix > B;
    B.reserve(q);
    for(std::size_t j = 0; j < q; ++j)
    {
        //for each stationary covariates
        std::size_t start_idx = std::reduce(L_j.cbegin(),std::next(L_j.cbegin(),j),static_cast<std::size_t>(0));
        //taking the right coefficients of the basis expansion
        FDAGWR_TRAITS::Dense_Matrix B_j = b.block(start_idx,0,L_j[j],1);
        B.push_back(B_j);
    }

    return B;
}

/*!
* @brief For non-stationary covariates, wrap the basis expansion coefficients of the functional regression coefficients, from column vector containing all the the coefficients to a vector containing only the coefficients for the covariate, for each statistical unit
* @param b a vector with n matrices, column matrices, of dimension Lx1, containing the basis expansion coefficients of the functional regression coefficients, one covariate after the other, for each of the statistical unit
* @param L_j vector containing the number of basis of each regression coefficient. Its sum is L
* @param q number of covariates
* @param n number of statistical units
* @return a vector of dimension q. Element i-th is a vector of n matrices, each one of dimension L_j[i]x1, with the basis expansion coefficients of the regression coefficient of the i-th stationary covariate, for each unit
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > >
fwr_operator_computing<INPUT,OUTPUT>::wrap_operator(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
                                                     const std::vector<std::size_t>& L_j,
                                                     std::size_t q,
                                                     std::size_t n) 
const
{
    //n is the number of units for training
    //input coherency
    assert((b.size() == n) && (L_j.size() == q));
    for(std::size_t i = 0; i < b.size(); ++i){     assert((b[i].cols() == 1) && (b[i].rows() == std::reduce(L_j.cbegin(),L_j.cend(),static_cast<std::size_t>(0))));}
    //container
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> B;    
    B.reserve(q);
    for(std::size_t j = 0; j < q; ++j)
    {
        //for each event-dependent covariates
        std::size_t start_idx = std::reduce(L_j.cbegin(),std::next(L_j.cbegin(),j),static_cast<std::size_t>(0));
        std::vector< FDAGWR_TRAITS::Dense_Matrix > B_j;
        B_j.reserve(b.size());
        //for all the units
        for(std::size_t i = 0; i < b.size(); ++i){
            //taking the right coefficients of the basis expansion
            FDAGWR_TRAITS::Dense_Matrix B_j_i = b[i].block(start_idx,0,L_j[j],1);
            B_j.push_back(B_j_i);}
        B.push_back(B_j);
    }

    return B;
}

/*!
* @brief For stationary covariates, dewrap the basis expansion coefficients for functional regression coefficients and puts them in a column matrix
* @param b vector containing, for each covariate, a column matrix containing the basis expansion coefficients  of the functional regression coefficients of that covariate
* @param L_j vector containing the number of basis for each covariate basis expansion of the functional regression coefficients
* @return a matrix, of dimension Lx1, L the sum of the element in L_j, containing the basis expansion coefficients of the functional regression coefficients, one after the other
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
FDAGWR_TRAITS::Dense_Matrix 
fwr_operator_computing<INPUT,OUTPUT>::dewrap_operator(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
                                                       const std::vector<std::size_t>& L_j) 
const
{
    //input coherency
    assert(b.size() == L_j.size());
    for(std::size_t i = 0; i < b.size(); ++i){  assert((b[i].cols()==1) && (b[i].rows()==L_j[i]));}

    FDAGWR_TRAITS::Dense_Matrix b_dewrapped(std::reduce(L_j.cbegin(),L_j.cend(),static_cast<std::size_t>(0)),1);

    for(std::size_t j = 0; j < L_j.size(); ++j)
    {
        std::size_t start_idx = std::reduce(L_j.cbegin(),std::next(L_j.cbegin(),j),static_cast<std::size_t>(0));
        b_dewrapped.block(start_idx,0,L_j[j],1) = b[j];
    }

    return b_dewrapped;
}

/*!
* @brief For non-stationary covariates, for each statistical unit, dewrap the basis expansion coefficients for functional regression coefficients and puts them in a column matrix
* @param b vector, of dimension q, containing, in element i-th, a vector of matrices, of dimension L_j[i]x1, for each statistical unit, the basis expansion coefficients of i-th covariate functional regression coefficients
* @param L_j vector containing the number of basis for each covariate basis expansion of the functional regression coefficients
* @param n the number of statistical units
* @return a vector of matrix, containing for each statistical unit, the basis expansion coefficients of functional regression coefficients, as column one after the other, in a Lx1, L sum of elements in L_j
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fwr_operator_computing<INPUT,OUTPUT>::dewrap_operator(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& b,
                                                       const std::vector<std::size_t>& L_j,
                                                       std::size_t n) 
const
{
    //input coherency
    assert(b.size() == L_j.size());
    std::size_t q = L_j.size();
    for(std::size_t j = 0; j < q; ++j){ assert(b[j].size() == n);}

    std::vector< FDAGWR_TRAITS::Dense_Matrix > b_dewrapped;
    b_dewrapped.reserve(n);

    for(std::size_t i = 0; i < n; ++i){

        std::vector< FDAGWR_TRAITS::Dense_Matrix > b_i;
        b_i.reserve(q);
        for(std::size_t j = 0; j < q; ++j){     b_i.push_back(b[j][i]);}
        b_dewrapped.push_back(this->dewrap_operator(b_i,L_j));
    }

    return b_dewrapped;
}

/*!
* @brief Evaluation the stationary betas, as basis expansion coefficients and basis, over a grid of points
* @param B vector of size q, containing for each covariate, as a column matrix, the basis expansion coefficients for the functional regression coefficient
* @param basis_B matrix of dimension qxL, where each row, for each covariate contains the basis for that functional regression coefficient. Each row contains only that basis, shifted, form position L_j[i-1] up to L_j[i] 
* @param L_j vector containing the number of basis for each covariate basis expansion of the functional regression coefficients
* @param q number of covariates
* @param abscissas vector of points over which evaluating the betas
* @return a vector of size q, such that element i-th is a vector containing the evaluations of beta of covariate i-th 
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< std::vector< OUTPUT >>
fwr_operator_computing<INPUT,OUTPUT>::eval_func_betas(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& B,
                                                      const functional_matrix_sparse<INPUT,OUTPUT>& basis_B,
                                                      const std::vector<std::size_t>& L_j,
                                                      std::size_t q,
                                                      const std::vector< INPUT >& abscissas) 
const
{
    
    //input coherency
    assert((B.size() == q) && (L_j.size() == q) && (basis_B.rows() == q) && (basis_B.cols() == std::reduce(L_j.cbegin(),L_j.cend(),static_cast<std::size_t>(0))));
    for(std::size_t j = 0; j < q; ++j){     assert((B[j].rows() == L_j[j]) && (B[j].cols() == 1));}
    //container
    std::vector< std::vector< OUTPUT >> beta;
    beta.reserve(B.size());
    //aliases
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    for(std::size_t j = 0; j < B.size(); ++j)
    {
        //retrieving the basis
        std::vector< F_OBJ > basis_j_v;
        basis_j_v.reserve(B[j].rows());
        std::size_t start_idx = std::reduce(L_j.cbegin(),std::next(L_j.cbegin(),j),static_cast<std::size_t>(0));
        std::size_t end_idx = start_idx + L_j[j];
        for(std::size_t k = start_idx; k < end_idx; ++k){   basis_j_v.push_back(basis_B(j,k));}
        functional_matrix<INPUT,OUTPUT> basis_j(basis_j_v,1,B[j].rows());

        //compute the beta
        FUNC_OBJ<INPUT,OUTPUT> beta_j = fm_prod<INPUT,OUTPUT>(basis_j,B[j],m_number_threads)(0,0);
        //eval the beta
        std::vector< OUTPUT > beta_j_ev; 
        beta_j_ev.resize(abscissas.size());
        std::transform(abscissas.cbegin(),abscissas.cend(),beta_j_ev.begin(),[&beta_j](F_OBJ_INPUT x){return beta_j(x);});
        beta.push_back(beta_j_ev);
    }

    return beta;
}

/*!
* @brief Evaluation the non-stationary betas, as basis expansion coefficients and basis, over a grid of points, for each statistical unit
* @param B vector of size q, each element is of size n and contains, for each unit, the basis expansion coefficients, as L_j_i x 1 matrix, of the functional regression coefficient for the respective covariate
* @param basis_B matrix of dimension qxL, where each row, for each covariate contains the basis for that functional regression coefficient. Each row contains only that basis, shifted, form position L_j[i-1] up to L_j[i] 
* @param L_j vector containing the number of basis for each covariate basis expansion of the functional regression coefficients
* @param q number of covariates
* @param n number of statistical units
* @param abscissas vector of points over which evaluating the betas
* @return a vector of size q, element i-th is a vector with an element for each unit, containing the evaluation of beta of covariate i-th for that unit over abscissas
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< std::vector< std::vector< OUTPUT >>>
fwr_operator_computing<INPUT,OUTPUT>::eval_func_betas(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& B,
                                                      const functional_matrix_sparse<INPUT,OUTPUT>& basis_B,
                                                      const std::vector<std::size_t>& L_j,
                                                      std::size_t q,
                                                      std::size_t n,
                                                      const std::vector< INPUT >& abscissas)
const
{
    //input coherency
    assert((B.size() == q) && (L_j.size() == q) && (basis_B.rows() == q) && (basis_B.cols() == std::reduce(L_j.cbegin(),L_j.cend(),static_cast<std::size_t>(0))));
    for(std::size_t j = 0; j < B.size(); ++j){  
        assert(B[j].size() == n);   
        for(std::size_t i = 0; i < B[j].size(); ++i){     assert((B[j][i].rows() == L_j[j]) && (B[j][i].cols() == 1));}}
        
    //container
    std::vector< std::vector< std::vector< OUTPUT >>> beta;
    beta.reserve(B.size());
    //aliases
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;
    

    for(std::size_t j = 0; j < B.size(); ++j)
    {
        //retrieving the basis
        std::vector< F_OBJ > basis_j_v;
        basis_j_v.reserve(L_j[j]);
        std::size_t start_idx = std::reduce(L_j.cbegin(),std::next(L_j.cbegin(),j),static_cast<std::size_t>(0));
        std::size_t end_idx = start_idx + L_j[j];
        for(std::size_t k = start_idx; k < end_idx; ++k){   basis_j_v.push_back(basis_B(j,k));}
        functional_matrix<INPUT,OUTPUT> basis_j(basis_j_v,1,L_j[j]);

        //evaluating the betas in every unit
        std::vector< std::vector<OUTPUT> > beta_j_ev;
        beta_j_ev.reserve(B[j].size());
        for(std::size_t i = 0; i < B[j].size(); ++i)
        {
            //compute the beta j-th for unit i-th
            FUNC_OBJ<INPUT,OUTPUT> beta_j_i = fm_prod<INPUT,OUTPUT>(basis_j,B[j][i],m_number_threads)(0,0);
            //eval the beta
            std::vector< OUTPUT > beta_j_i_ev; 
            beta_j_i_ev.resize(abscissas.size());
            std::transform(abscissas.cbegin(),abscissas.cend(),beta_j_i_ev.begin(),[&beta_j_i](F_OBJ_INPUT x){return beta_j_i(x);});
            beta_j_ev.push_back(beta_j_i_ev);
        }

        beta.push_back(beta_j_ev);
    }

    return beta;
}

/*!
* @brief Evaluating the stationary betas, as functional matrix, over a grid of points
* @param beta a qx1 matrix of functions containing the functional regression coefficients for stationary covariates
* @param q number of covariates
* @param abscissa vector of points over which evaluating the betas
* @return a vector of size q, such that element i-th is a vector containing the evaluations of beta of covariate i-th 
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< std::vector<OUTPUT> >
fwr_operator_computing<INPUT,OUTPUT>::eval_func_betas(const functional_matrix<INPUT,OUTPUT> &beta,
                                                      std::size_t q,
                                                      const std::vector<INPUT> &abscissa) 
const
{
    //input coherency
    assert((beta.rows() == q) && (beta.cols() == 1));
    //number of evaluations
    std::size_t n_abs = abscissa.size();

    //reserving
    std::vector< std::vector<OUTPUT>> beta_ev;    
    beta_ev.reserve(q);        

    for (std::size_t j = 0; j < q; ++j)
    {
        std::vector<OUTPUT> beta_j_ev;
        beta_j_ev.resize(n_abs);

#ifdef _OPENMP
#pragma omp parallel for shared(beta_j_ev,j,abscissa,n_abs) num_threads(m_number_threads)
#endif
        for(std::size_t i = 0; i < n_abs; ++i)
        {
            beta_j_ev[i] = beta(j,0)(abscissa[i]);
        }

        beta_ev.push_back(beta_j_ev);
    }

    return beta_ev;
}

/*!
* @brief Evaluating the non-stationary betas, as functional matrix, over a grid of points
* @param beta vector, one element for each statistical unit, containing a qx1 matrix of functions containing the functional regression coefficients for non-stationary covariates
* @param q number of covariates
* @param abscissa vector of points over which evaluating the betas
* @return a vector of size q, element i-th is a vector with an element for each unit, containing the evaluation of beta of covariate i-th for that unit over abscissa
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< std::vector< std::vector<OUTPUT>>>
fwr_operator_computing<INPUT,OUTPUT>::eval_func_betas(const std::vector< functional_matrix<INPUT,OUTPUT>> &beta,
                                                      std::size_t q,
                                                      const std::vector<INPUT> &abscissa)
const
{
    std::size_t n_pred = beta.size();
    std::size_t n_abs  = abscissa.size();
    //input coherency
    for(std::size_t i = 0; i < n_pred; ++i){    assert((beta[i].rows() == q) && (beta[i].cols() == 1));}
    

    //reserving
    std::vector< std::vector< std::vector<OUTPUT>> > beta_ev;    
    beta_ev.reserve(q);  
    
    for(std::size_t j = 0; j < q; ++j)
    {
        std::vector< std::vector<OUTPUT>> beta_j_ev;
        beta_j_ev.reserve(n_pred);

        for(std::size_t i = 0; i < n_pred; ++i)
        {
            std::vector<OUTPUT> beta_j_i_ev;
            beta_j_i_ev.resize(n_abs);

#ifdef _OPENMP
#pragma omp parallel for shared(beta_j_i_ev,j,i,abscissa,n_abs) num_threads(m_number_threads)
#endif
            for(std::size_t i_ev = 0; i_ev < n_abs; ++i_ev)
            {
                beta_j_i_ev[i_ev] = beta[i](j,0)(abscissa[i_ev]);
            }

            beta_j_ev.push_back(beta_j_i_ev);
        }

        beta_ev.push_back(beta_j_ev);
    }

    return beta_ev;
}