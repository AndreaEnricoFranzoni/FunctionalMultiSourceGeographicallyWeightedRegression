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

#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"

#include "../integration/fgwr_operator_computing.hpp"
#include "../functional_matrix/functional_matrix_smoothing.hpp"
#include "../basis/basis_include.hpp"
#include "../utility/parameters_wrapper_fdagwr.hpp"

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
    fgwr_operator_computing<INPUT,OUTPUT> m_operator_comp;
    /*!Abscissa points over which there are the evaluations of the raw fd*/
    std::vector<INPUT> m_abscissa_points;
    /*!Number of statistical units used to fit the model*/
    std::size_t m_n;
    /*!Number of threads for OMP*/
    int m_number_threads;

public:
    /*!
    * @brief Constructor
    * @param number_threads number of threads for OMP
    */
    fgwr(INPUT a, INPUT b, int n_intervals_integration, double target_error, int max_iterations, const std::vector<INPUT> & abscissa_points, std::size_t n, int number_threads)
        : m_operator_comp(a,b,n_intervals_integration,target_error,max_iterations,number_threads), m_abscissa_points(abscissa_points), m_n(n), m_number_threads(number_threads) {}

    /*!
    * @brief Virtual destructor
    */
    virtual ~fgwr() = default;

    /*!
    * @brief Getter for the compute operator
    */
    const fgwr_operator_computing<INPUT,OUTPUT>& operator_comp() const {return m_operator_comp;}

    /*!
    * @brief Getter for the abscissa points for which the evaluation of the fd is available
    * @return the private
    */
    const std::vector<INPUT>& abscissa_points() const {return m_abscissa_points;}

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

/*
    //brief Wrap b, for stationary covariates
    
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    wrap_b(const FDAGWR_TRAITS::Dense_Matrix& b,
           const std::vector<std::size_t>& L_j,
           std::size_t q) const;

    //@brief Wrap b, for non-stationary covariates
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>
    wrap_b(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
           const std::vector<std::size_t>& L_j,
           std::size_t q) const;
*/

    /*!
    * @brief Evaluation of the betas, for stationary covariates
    */
    std::vector< std::vector< OUTPUT >>
    eval_betas(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& B,
               const functional_matrix_sparse<INPUT,OUTPUT>& basis_B,
               const std::vector<std::size_t>& L_j,   
               std::size_t q,
               const std::vector< INPUT >& abscissas) const;

    /*!
    * @brief Evaluation of the betas, for non-stationary covariates
    */
    std::vector< std::vector< std::vector< OUTPUT >>>
    eval_betas(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& B,
               const functional_matrix_sparse<INPUT,OUTPUT>& basis_B,
               const std::vector<std::size_t>& L_j,
               std::size_t q,
               const std::vector< INPUT >& abscissas) const;

    /*!
    * @brief Virtual method to compute the Functional Geographically Weighted Regression
    */
    virtual inline void compute() = 0;

    /*!
    * @brief Virtual method to compute the betas
    */
    virtual inline void evalBetas() = 0;

    /*!
    * @brief Function to return the coefficients of the betas basis expansion, tuple of different dimension depending on the algo used
    */
    virtual inline BTuple bCoefficients() const = 0;

    /*!
    * @brief Function to return the the betas evaluated, tuple of different dimension depending on the algo used
    */
    virtual inline BetasTuple betas() const = 0;

    /*!
    * @brief Function to return extra objects useful for reporting the functional partial residuals
    */
    virtual inline PartialResidualTuple PRes() const = 0;
};


#include "fgwr_imp.hpp"

#endif  /*FGWR_ALGO_HPP*/