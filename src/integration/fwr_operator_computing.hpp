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


#ifndef FWR_OPERATOR_COMPUTING_HPP
#define FWR_OPERATOR_COMPUTING_HPP


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"

#include "../functional_matrix/functional_matrix.hpp"
#include "../functional_matrix/functional_matrix_sparse.hpp"
#include "../functional_matrix/functional_matrix_diagonal.hpp"
#include "../functional_matrix/functional_matrix_product.hpp"
#include "../functional_matrix/functional_matrix_operators.hpp"

#include "functional_data_integration.hpp"


/*!
* @file fwr_operator_computing.hpp
* @brief Contains the class for computing operators needed in FWR, wrapping and dewrapping the basis expansion coefficients of the functional regression coefficients, evaluating the latters 
* @author Andrea Enrico Franzoni
*/


/*!
* @class fwr_operator_computing
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Computing operators and functional operators in FWR. Wrapping and dewrapping basis expansion coefficients of the functional regression coefficients. Evaluating the latters
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_operator_computing
{
private:
    /*!Object to perform the integration using rectangle quadrature rule*/
    fd_integration m_integrating;
    /*!Number of threads for OMP*/
    int m_number_threads;

    /*!
    * @brief Integrating element-wise a functional matrix
    * @param integrand matrix containing std::function objects
    * @return a matrix containing the integration of all the integrands
    */
    inline
    FDAGWR_TRAITS::Dense_Matrix
    fm_integration(const functional_matrix<INPUT,OUTPUT> &integrand)
    const
    {
        //storing integrals results
        std::vector<OUTPUT> result_integrand;
        result_integrand.resize(integrand.size());

#ifdef _OPENMP
#pragma omp parallel for shared(integrand,m_integrating,result_integrand) num_threads(m_number_threads)
#endif
        for(std::size_t i = 0; i < integrand.size(); ++i){
            result_integrand[i] = m_integrating.integrate(integrand.as_vector()[i]);}

        FDAGWR_TRAITS::Dense_Matrix result_integration = Eigen::Map< FDAGWR_TRAITS::Dense_Matrix >(result_integrand.data(), integrand.rows(), integrand.cols());

        return result_integration;
    }


public:
    /*!
    * @brief Constructor
    * @param a left integration domain extreme
    * @param b right integration domain extreme
    * @param n_intervals_integration number of equally spaced intervals dividing the integration domain
    * @param number_threads number of threads for OMP
    */
    fwr_operator_computing(INPUT a, INPUT b, int n_intervals_integration, int number_threads)
        : m_integrating(a,b,n_intervals_integration), m_number_threads(number_threads) {}


    /*!
    * @brief Compute [J_i + R]^(-1), where, for each unit i-th:
    *        - J_i = int_a_b(base_t * X_t * W_i * X * base), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
    *        - R the penalization matrix, diagonal block-matrix, such that each block contains the inner product within a given derivative of basis systems, one block for each basis system
    * @param base_t containing basis systems, one system for each row, one basis for each column, transpost
    * @param X_t functional covariates, transpost
    * @param W vector of functional diagonal matrices, containing, for each unit, functional weights
    * @param X functioal covaraites
    * @param base containing basis systems, one system for each row, one basis for each column
    * @param R penalization matrix
    * @return vector with the partial PivLU of each J_i + R
    */
    std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > >
    compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                    const functional_matrix<INPUT,OUTPUT> &X_t,
                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                    const functional_matrix<INPUT,OUTPUT> &X,
                    const functional_matrix_sparse<INPUT,OUTPUT> &base,
                    const FDAGWR_TRAITS::Sparse_Matrix &R) const;

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
    std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > >
    compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                    const functional_matrix<INPUT,OUTPUT> &X_crossed,
                    const FDAGWR_TRAITS::Sparse_Matrix &R) const;

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
    Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >
    compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                    const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                    const functional_matrix<INPUT,OUTPUT> &X_crossed,
                    const FDAGWR_TRAITS::Sparse_Matrix &R) const;

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
    * @note for FWR when having only stationary covariates
    */
    Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >
    compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                    const functional_matrix<INPUT,OUTPUT> &X_t,
                    const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                    const functional_matrix<INPUT,OUTPUT> &X,
                    const functional_matrix_sparse<INPUT,OUTPUT> &base,
                    const FDAGWR_TRAITS::Sparse_Matrix &R) const;

    /*!
    * @brief Compute a scalar (matrix) operator as, for each unit
    *        [J_i + R]^(-1) * int_a_b(base_lhs * X_lhs * W_i * X_rhs * base_rhs), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
    * @param base_lhs containing basis systems, one system for each row, one basis for each column
    * @param X_lhs functional data matrix
    * @param W vector of functional diagonal matrices, containing, for each unit, functional weights
    * @param X_rhs functional data matrix
    * @param base_rhs containing basis systems, one system for each row, one basis for each column
    * @param penalty vector of partial PivLU decomposition, containing the penalties [J_i + R]^(-1) as computed by the functions above
    * @return a matrix containing the element-wise integration
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

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
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

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
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

    /*!
    * @brief Compute a scalar (matrix) operator as, for each unit
    *        [J_i + R]^(-1) * int_a_b(X_lhs * W_i * base_rhs), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
    * @param X_lhs functional data matrix
    * @param W vector of functional diagonal matrices, containing, for each unit, functional weights
    * @param base_rhs containing basis systems, one system for each row, one basis for each column
    * @param penalty vector of partial PivLU decomposition, containing the penalties [J_i + R]^(-1) as computed by the functions above
    * @return a matrix containing the element-wise integration
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;
    
    /*!
    * @brief Compute a scalar (matrix) operator as, for each unit
    *        [J_i + R]^(-1) * int_a_b(X_lhs * W_i * X_rhs), where W_i is the functional weight of the i-th units. The integral of the matrix is element-wise
    * @param X_lhs functional data matrix
    * @param W vector of functional diagonal matrices, containing, for each unit, functional weights
    * @param X_rhs functional data matrix
    * @param penalty vector of partial PivLU decomposition, containing the penalties [J_i + R]^(-1) as computed by the functions above
    * @return a matrix containing the element-wise integration
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

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
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

    /*!
    * @brief Compute a scalar (matrix) operator as [J + R]^(-1) * int_a_b(X_lhs * W_i * X_rhs), where W is the functional weight. The integral of the matrix is element-wise
    * @param X_lhs functional data matrix
    * @param W functional diagonal matrix, containing functional weights
    * @param X_rhs functional data matrix
    * @param penalty partial PivLU decomposition, containing the penalty [J + R]^(-1) as computed by the functions above
    * @return a matrix containing the element-wise integration
    */
    FDAGWR_TRAITS::Dense_Matrix
    compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > &penalty) const;

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
    FDAGWR_TRAITS::Dense_Matrix
    compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > &penalty) const;

    /*!
    * @brief Compute a functional operator, intended as a matrix where:
    *        each row is the respective row of the functional covaraite, multiplied by the base system, and then by the operator computed
    * @param X functional data covariates
    * @param base basis system
    * @param _operator_ an operator computed as above
    * @return the functional operator, as functional matrix       
    */
    functional_matrix<INPUT,OUTPUT> 
    compute_functional_operator(const functional_matrix<INPUT,OUTPUT> &X,
                                const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                const std::vector< FDAGWR_TRAITS::Dense_Matrix > &_operator_) const;

    /*!
    * @brief For stationary covariates, wrap the basis expansion coefficients of the functional regression coefficients, from column vector containing all the the coefficients to a vector containing only the coefficients for the covariate
    * @param b a matrix, column matrix, of dimension Lx1, containing the basis expansion coefficients of the functional regression coefficients, one covariate after the other
    * @param L_j vector containing the number of basis of each regression coefficient. Its sum is L
    * @param q number of covariates
    * @return a vector of matrices, the i-th matrix of dimension L_j[i]x1, with the basis expansion coefficients of the regression coefficient of the i-th stationary covariate
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    wrap_operator(const FDAGWR_TRAITS::Dense_Matrix& b,
                  const std::vector<std::size_t>& L_j,
                  std::size_t q) const;

    /*!
    * @brief For non-stationary covariates, wrap the basis expansion coefficients of the functional regression coefficients, from column vector containing all the the coefficients to a vector containing only the coefficients for the covariate, for each statistical unit
    * @param b a vector with n matrices, column matrices, of dimension Lx1, containing the basis expansion coefficients of the functional regression coefficients, one covariate after the other, for each of the statistical unit
    * @param L_j vector containing the number of basis of each regression coefficient. Its sum is L
    * @param q number of covariates
    * @param n number of statistical units
    * @return a vector of dimension q. Element i-th is a vector of n matrices, each one of dimension L_j[i]x1, with the basis expansion coefficients of the regression coefficient of the i-th stationary covariate, for each unit
    */
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>
    wrap_operator(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
                  const std::vector<std::size_t>& L_j,
                  std::size_t q,
                  std::size_t n) const;

    /*!
    * @brief For stationary covariates, dewrap the basis expansion coefficients for functional regression coefficients and puts them in a column matrix
    * @param b vector containing, for each covariate, a column matrix containing the basis expansion coefficients  of the functional regression coefficients of that covariate
    * @param L_j vector containing the number of basis for each covariate basis expansion of the functional regression coefficients
    * @return a matrix, of dimension Lx1, L the sum of the element in L_j, containing the basis expansion coefficients of the functional regression coefficients, one after the other
    */
    FDAGWR_TRAITS::Dense_Matrix 
    dewrap_operator(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
                    const std::vector<std::size_t>& L_j) const;

    /*!
    * @brief For non-stationary covariates, for each statistical unit, dewrap the basis expansion coefficients for functional regression coefficients and puts them in a column matrix
    * @param b vector, of dimension q, containing, in element i-th, a vector of matrices, of dimension L_j[i]x1, for each statistical unit, the basis expansion coefficients of i-th covariate functional regression coefficients
    * @param L_j vector containing the number of basis for each covariate basis expansion of the functional regression coefficients
    * @param n the number of statistical units
    * @return a vector of matrix, containing for each statistical unit, the basis expansion coefficients of functional regression coefficients, as column one after the other, in a Lx1, L sum of elements in L_j
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    dewrap_operator(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& b,
                    const std::vector<std::size_t>& L_j,
                    std::size_t n) const;

    /*!
    * @brief Evaluation the stationary betas, as basis expansion coefficients and basis, over a grid of points
    * @param B vector of size q, containing for each covariate, as a column matrix, the basis expansion coefficients for the functional regression coefficient
    * @param basis_B matrix of dimension qxL, where each row, for each covariate contains the basis for that functional regression coefficient. Each row contains only that basis, shifted, form position L_j[i-1] up to L_j[i] 
    * @param L_j vector containing the number of basis for each covariate basis expansion of the functional regression coefficients
    * @param q number of covariates
    * @param abscissas vector of points over which evaluating the betas
    * @return a vector of size q, such that element i-th is a vector containing the evaluations of beta of covariate i-th 
    */
    std::vector< std::vector< OUTPUT >>
    eval_func_betas(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& B,
                    const functional_matrix_sparse<INPUT,OUTPUT>& basis_B,
                    const std::vector<std::size_t>& L_j,   
                    std::size_t q,
                    const std::vector< INPUT >& abscissas) const;

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
    std::vector< std::vector< std::vector< OUTPUT >>>
    eval_func_betas(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& B,
                    const functional_matrix_sparse<INPUT,OUTPUT>& basis_B,
                    const std::vector<std::size_t>& L_j,
                    std::size_t q,
                    std::size_t n,
                    const std::vector< INPUT >& abscissas) const;
    
    /*!
    * @brief Evaluating the stationary betas, as functional matrix, over a grid of points
    * @param beta a qx1 matrix of functions containing the functional regression coefficients for stationary covariates
    * @param q number of covariates
    * @param abscissa vector of points over which evaluating the betas
    * @return a vector of size q, such that element i-th is a vector containing the evaluations of beta of covariate i-th 
    */
    std::vector< std::vector<OUTPUT> >
    eval_func_betas(const functional_matrix<INPUT,OUTPUT> &beta,
                    std::size_t q,
                    const std::vector<INPUT> &abscissa) const;

    /*!
    * @brief Evaluating the non-stationary betas, as functional matrix, over a grid of points, for each statistical unit
    * @param beta vector, one element for each statistical unit, containing a qx1 matrix of functions containing the functional regression coefficients for non-stationary covariates
    * @param q number of covariates
    * @param abscissa vector of points over which evaluating the betas
    * @return a vector of size q, element i-th is a vector with an element for each unit, containing the evaluation of beta of covariate i-th for that unit over abscissa
    */
    std::vector< std::vector< std::vector<OUTPUT>>>
    eval_func_betas(const std::vector< functional_matrix<INPUT,OUTPUT>> &beta,
                    std::size_t q,
                    const std::vector<INPUT> &abscissa) const;

};

#include "fwr_operator_computing_imp.hpp"

#endif  /*FWR_OPERATOR_COMPUTING_HPP*/