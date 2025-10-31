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
    * @brief Wrap b, for stationary covariates (da colonna, li mette in un vettore, coefficienti per ogni covariate)
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    wrap_operator(const FDAGWR_TRAITS::Dense_Matrix& b,
                  const std::vector<std::size_t>& L_j,
                  std::size_t q) const;

    /*!
    * @brief Wrap b, for non-stationary covariates (da colonna, li mette in un vettore, coefficienti per ogni covariate, che sono vettori, coefficienti per ogni unità)
    */
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>
    wrap_operator(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
                  const std::vector<std::size_t>& L_j,
                  std::size_t q,
                  std::size_t n) const;

    /*!
    * @brief Dewrap b, for stationary covariates: me li incolonna tutti 
    */
    FDAGWR_TRAITS::Dense_Matrix 
    dewrap_operator(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
                    const std::vector<std::size_t>& L_j) const;

    /*!
    * @brief Dewrap b, for non-stationary covariates: me li incolonna tutti
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    dewrap_operator(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& b,
                    const std::vector<std::size_t>& L_j,
                    std::size_t n) const;

    /*!
    * @brief Evaluation of the betas, for stationary covariates, from coefficients (incolonnati) + basi
    */
    std::vector< std::vector< OUTPUT >>
    eval_func_betas(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& B,
                    const functional_matrix_sparse<INPUT,OUTPUT>& basis_B,
                    const std::vector<std::size_t>& L_j,   
                    std::size_t q,
                    const std::vector< INPUT >& abscissas) const;

    /*!
    * @brief Evaluation of the betas, for non-stationary covariates, from coefficients (incolonnati (ogni elemento: sta per una covariata, con i coeff per ogni unità (n))) + basi
    */
    std::vector< std::vector< std::vector< OUTPUT >>>
    eval_func_betas(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& B,
                    const functional_matrix_sparse<INPUT,OUTPUT>& basis_B,
                    const std::vector<std::size_t>& L_j,
                    std::size_t q,
                    std::size_t n,
                    const std::vector< INPUT >& abscissas) const;
    
    /*!
    * @brief Eval the stationary betas on a grid, as func matrices
    */
    std::vector< std::vector<OUTPUT> >
    eval_func_betas(const functional_matrix<INPUT,OUTPUT> &beta,
                    std::size_t q,
                    const std::vector<INPUT> &abscissa) const;

    /*!
    * @brief Eval the non-stationary betas on a grid
    */
    std::vector< std::vector< std::vector<OUTPUT>>>
    eval_func_betas(const std::vector< functional_matrix<INPUT,OUTPUT>> &beta,
                    std::size_t q,
                    const std::vector<INPUT> &abscissa) const;

};

#include "fwr_operator_computing_imp.hpp"

#endif  /*FWR_OPERATOR_COMPUTING_HPP*/