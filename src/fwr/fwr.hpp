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


#ifndef FWR_ALGO_HPP
#define FWR_ALGO_HPP

#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"

#include "../integration/fwr_operator_computing.hpp"
#include "../functional_matrix/functional_matrix_smoothing.hpp"
#include "../basis/basis_include.hpp"
#include "../utility/parameters_wrapper_fdagwr.hpp"

#include <iostream>


/*!
* @brief Virtual interface to perform the 
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr
{
private:
    /*!Object to perform the integration using trapezoidal quadrature rule*/
    fwr_operator_computing<INPUT,OUTPUT> m_operator_comp;
    /*!Abscissa points over which there are the evaluations of the raw fd*/
    std::vector<INPUT> m_abscissa_points;
    /*!Number of statistical units used to fit the model*/
    std::size_t m_n;
    /*!Number of threads for OMP*/
    int m_number_threads;
    /*!Brute force estimation*/
    bool m_brute_force_estimation;

public:
    /*!
    * @brief Constructor
    * @param number_threads number of threads for OMP
    */
    fwr(INPUT a, INPUT b, int n_intervals_integration, double target_error, int max_iterations, const std::vector<INPUT> & abscissa_points, std::size_t n, int number_threads, bool brute_force_estimation)
        : m_operator_comp(a,b,n_intervals_integration,target_error,max_iterations,number_threads), m_abscissa_points(abscissa_points), m_n(n), m_number_threads(number_threads), m_brute_force_estimation(brute_force_estimation) {}

    /*!
    * @brief Virtual destructor
    */
    virtual ~fwr() = default;

    /*!
    * @brief Getter for the compute operator
    */
    const fwr_operator_computing<INPUT,OUTPUT>& operator_comp() const {return m_operator_comp;}

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

    /*!
    * @brief Getter for m_brute_force_estimation
    */
    inline bool bf_estimation() const {return m_brute_force_estimation;}

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

#endif  /*FWR_ALGO_HPP*/