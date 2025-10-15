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


#ifndef FGWR_OPERATOR_COMPUTING_HPP
#define FGWR_OPERATOR_COMPUTING_HPP


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"

#include "../functional_matrix/functional_matrix.hpp"
#include "../functional_matrix/functional_matrix_sparse.hpp"
#include "../functional_matrix/functional_matrix_diagonal.hpp"
#include "../functional_matrix/functional_matrix_product.hpp"
#include "../functional_matrix/functional_matrix_operators.hpp"

#include "functional_data_integration.hpp"



template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fgwr_operator_computing
{
private:
    /*!Object to perform the integration using trapezoidal quadrature rule*/
    fd_integration m_integrating;
    /*!Number of threads for OMP*/
    int m_number_threads;

    /*!
    * @brief Integrating element-wise a functional matrix
    */
    inline
    FDAGWR_TRAITS::Dense_Matrix
    fm_integration(const functional_matrix<INPUT,OUTPUT> &integrand)
    const
    {
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
    * @param number_threads number of threads for OMP
    */
    fgwr_operator_computing(INPUT a, INPUT b, int n_intervals_integration, double target_error, int max_iterations, int number_threads)
        : m_integrating(a,b,n_intervals_integration,target_error,max_iterations), m_number_threads(number_threads) {}


    /*!
    * @brief Compute all the [J_2_tilde_i + R]^(-1): 
    * @note FATTO
    */
    std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > >
    compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                    const functional_matrix<INPUT,OUTPUT> &X_t,
                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                    const functional_matrix<INPUT,OUTPUT> &X,
                    const functional_matrix_sparse<INPUT,OUTPUT> &base,
                    const FDAGWR_TRAITS::Sparse_Matrix &R) const;

    /*!
    * @brief Compute [J_tilde_i + R]^(-1)
    * @note FATTO
    */
    std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > >
    compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                    const functional_matrix<INPUT,OUTPUT> &X_crossed,
                    const FDAGWR_TRAITS::Sparse_Matrix &R) const;

    /*!
    * @brief Compute [J + Rc]^(-1) quando non ci sono solo stazionarie
    * @note FATTO
    */
    Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >
    compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                    const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                    const functional_matrix<INPUT,OUTPUT> &X_crossed,
                    const FDAGWR_TRAITS::Sparse_Matrix &R) const;

    /*!
    * @brief Compute [J + Rc]^(-1) per FGWR solo stazionarie
    * @note FATTO
    */
    Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >
    compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                    const functional_matrix<INPUT,OUTPUT> &X_t,
                    const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                    const functional_matrix<INPUT,OUTPUT> &X,
                    const functional_matrix_sparse<INPUT,OUTPUT> &base,
                    const FDAGWR_TRAITS::Sparse_Matrix &R) const;

    /*!
    * @brief Compute an operator
    * @note FATTO
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

    /*!
    * @brief Compute an operator
    * @note FATTO
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

    /*!
    * @brief Compute an operator
    * @note FATTO
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

    /*!
    * @brief Compute an operator
    * @note FATTO
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;
    
    /*!
    * @brief Compute an operator
    * @note FATTO
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

    /*!
    * @brief Compute an operator
    * @note FATTO
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

    /*!
    * @brief Compute the operator for stationary coefficients
    * @note FATTO
    */
    FDAGWR_TRAITS::Dense_Matrix
    compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > &penalty) const;

    /*!
    * @brief Compute the operator for stationary coefficients
    * @note FATTO
    */
    FDAGWR_TRAITS::Dense_Matrix
    compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > &penalty) const;

    /*!
    * @brief Compute a functional operator
    * @note FATTO
    */
    functional_matrix<INPUT,OUTPUT> 
    compute_functional_operator(const functional_matrix<INPUT,OUTPUT> &X,
                                const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                const std::vector< FDAGWR_TRAITS::Dense_Matrix > &_operator_) const;

    /*!
    * @brief Wrap b, for stationary covariates
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    wrap_operator(const FDAGWR_TRAITS::Dense_Matrix& b,
                  const std::vector<std::size_t>& L_j,
                  std::size_t q) const;

    /*!
    * @brief Wrap b, for non-stationary covariates
    */
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>
    wrap_operator(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
                  const std::vector<std::size_t>& L_j,
                  std::size_t q,
                  std::size_t n) const;

};

#include "fgwr_operator_computing_imp.hpp"

#endif  /*FGWR_OPERATOR_COMPUTING_HPP*/