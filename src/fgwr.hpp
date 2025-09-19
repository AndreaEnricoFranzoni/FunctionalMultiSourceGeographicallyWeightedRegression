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


#ifndef FGWR_ALGO_HPP
#define FGWR_ALGO_HPP

#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"

#include "functional_matrix.hpp"
#include "functional_matrix_sparse.hpp"
#include "functional_matrix_diagonal.hpp"
#include "functional_matrix_product.hpp"
#include "functional_matrix_operators.hpp"

#include "functional_data_integration.hpp"

#include <iostream>
/*!
* @brief Virtual interface to perform the 
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fgwr
{
private:
    /*!Object to perform the integration using trapezoidal quadrature rule*/
    fd_integration m_integrating;
    /*!Number of statistical units*/
    std::size_t m_n;
    /*!Number of threads for OMP*/
    int m_number_threads;

public:
    /*!
    * @brief Constructor
    * @param number_threads number of threads for OMP
    */
    fgwr(INPUT a, INPUT b, int n_intervals_integration, double target_error, int max_iterations, std::size_t n, int number_threads)
        : m_integrating(a,b,n_intervals_integration,target_error,max_iterations), m_n(n), m_number_threads(number_threads) {}

    /*!
    * @brief Virtual destructor
    */
    virtual ~fgwr() = default;

    /*!
    * @brief Getter for the number of statistical units
    * @return the private m_n
    */
    inline std::size_t n() const {return m_n;}

    /*!
    * @brief Getter for the number of threads for OMP
    * @return the private m_number_threads
    */
    inline int number_threads() const {return m_number_threads;}

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
        //integrating every element of the functional matrix
        std::transform(cbegin(integrand),
                       cend(integrand),
                       result_integrand.begin(),
                       [this](const FUNC_OBJ<INPUT,OUTPUT> &f){ return this->m_integrating.integrate(f);});

        FDAGWR_TRAITS::Dense_Matrix result_integration = Eigen::Map< FDAGWR_TRAITS::Dense_Matrix >(result_integrand.data(), integrand.rows(), integrand.cols());

        return result_integration;
    }

    /*!
    * @brief Compute all the [J_2_tilde_i + R]^(-1): 
    * @note FATTO
    */
    std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > >
    compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base,
                    const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                    const functional_matrix<INPUT,OUTPUT> &X,
                    const functional_matrix<INPUT,OUTPUT> &X_t,
                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
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
    * @brief Compute [J + Rc]^(-1)
    */
    Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >
    compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed,
                    const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                    const functional_matrix_diagonal<INPUT,OUTPUT> &W,
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
    */
    FDAGWR_TRAITS::Dense_Matrix
    compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
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
                                const std::vector< FDAGWR_TRAITS::Dense_Matrix > &operator_) const;

    /*!
    * @brief Virtual method to compute the Functional Geographically Weighted Regression
    */
    virtual inline void compute() = 0;
};


#include "fgwr_imp.hpp"

#endif  /*FGWR_ALGO_HPP*/