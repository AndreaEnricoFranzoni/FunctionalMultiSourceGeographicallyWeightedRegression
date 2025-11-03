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
* @file fwr.hpp
* @brief Contains the definition of a the virtual base class for the functional weighted regression model
* @author Andrea Enrico Franzoni
*/


/*!
* @class fwr
* @brief Base class for the functional weighted regression model, virtual interface
* @tparam INPUT type of functional data abscissa
* @tparam OUTPUT type of functional data image
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr
{
private:
    /*!Object to perform the integration using midpoint quadrature rule*/
    fwr_operator_computing<INPUT,OUTPUT> m_operator_comp;
    /*!Abscissa points over which there are the evaluations of the raw fd*/
    std::vector<INPUT> m_abscissa_points;
    /*!Number of statistical units used to fit the model*/
    std::size_t m_n;
    /*!Number of threads for OMP*/
    int m_number_threads;
    /*!If false, performing fwr with more than one source in an exact way. If true, estimating the coefficients in cascade*/
    bool m_in_cascade_estimation;

public:
    /*!
    * @brief Constructor
    * @param a left extreme functional data domain 
    * @param a right extreme functional data domain 
    * @param n_intervals_integration number of intervals used by the midpoint quadrature rule
    * @param abscissa_points abscissa points over which there are the evaluations of the raw functional data
    * @param n number of training statistical units
    * @param number_threads number of threads for OMP
    * @param in_cascade_estimation if true, for more than one source covariates, the estimation is made in cascade. If false, exact
    */
    fwr(INPUT a, INPUT b, int n_intervals_integration, const std::vector<INPUT> & abscissa_points, std::size_t n, int number_threads, bool in_cascade_estimation)
        : m_operator_comp(a,b,n_intervals_integration,number_threads), m_abscissa_points(abscissa_points), m_n(n), m_number_threads(number_threads), m_in_cascade_estimation(in_cascade_estimation) {}

    /*!
    * @brief Virtual destructor
    */
    virtual ~fwr() = default;

    /*!
    * @brief Getter for operator that computes operators, scalar and functional
    * @return the private m_operator_comp
    */
    const fwr_operator_computing<INPUT,OUTPUT>& operator_comp() const {return m_operator_comp;}

    /*!
    * @brief Getter for the abscissa points for which the evaluations of the functional data are available
    * @return the private m_abscissa_points
    */
    const std::vector<INPUT>& abscissa_points() const {return m_abscissa_points;}

    /*!
    * @brief Getter for the number of statistical units used for training
    * @return the private m_n
    */
    inline std::size_t n() const {return m_n;}

    /*!
    * @brief Getter for the number of threads for OMP
    * @return the private m_number_threads
    */
    inline int number_threads() const {return m_number_threads;}

    /*!
    * @brief Getter for how the estimation is performed
    * @return the private m_in_cascade_estimation
    */
    inline bool in_cascade_estimation() const {return m_in_cascade_estimation;}

    /*!
    * @brief Virtual method to compute the Functional Weighted Regression basis expansion coefficients of the functional regression coefficients
    */
    virtual inline void compute() = 0;

    /*!
    * @brief Virtual method to evaluate the functional regression coefficients over a grid of points (m_abscissa_points)
    */
    virtual inline void evalBetas() = 0;

    /*!
    * @brief Function to return the basis expansion coefficients of the functional regression coefficitens
    * @return a tuple of different dimension depending on the model fitted
    */
    virtual inline BTuple bCoefficients() const = 0;

    /*!
    * @brief Function to return the the functional regression coefficients evaluated
    * @return a tuple of different dimension depending on the model fitted
    */
    virtual inline BetasTuple betas() const = 0;

    /*!
    * @brief Function to return objects useful for reconstructing the functional partial residuals
    * @return a tuple of different dimension depending on the model fitted
    */
    virtual inline PartialResidualTuple PRes() const = 0;
};

#endif  /*FWR_ALGO_HPP*/