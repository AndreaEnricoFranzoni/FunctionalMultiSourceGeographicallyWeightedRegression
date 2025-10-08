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


#ifndef FGWR_PREDICT_HPP
#define FGWR_PREDICT_HPP


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"

#include "../functional_matrix/functional_matrix.hpp"
#include "../functional_matrix/functional_matrix_sparse.hpp"
#include "../functional_matrix/functional_matrix_diagonal.hpp"
#include "../functional_matrix/functional_matrix_product.hpp"
#include "../functional_matrix/functional_matrix_operators.hpp"
#include "../functional_matrix/functional_matrix_smoothing.hpp"

#include "../integration/functional_data_integration.hpp"
#include "../basis/basis_include.hpp"
#include "../utility/parameters_wrapper_fdagwr.hpp"


template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fgwr_predict
{
private:
    /*!Object to perform the integration using trapezoidal quadrature rule*/
    fd_integration m_integrating;
    /*!Abscissa points over which there are the evaluations of the raw fd*/
    std::vector<INPUT> m_abscissa_points;
    /*!Number of statistical units*/
    std::size_t m_n;
    /*!Number of threads for OMP*/
    int m_number_threads;

public:
    /*!
    * @brief Constructor
    * @param number_threads number of threads for OMP
    */
    fgwr_predict(INPUT a, INPUT b, int n_intervals_integration, double target_error, int max_iterations, const std::vector<INPUT> & abscissa_points, std::size_t n, int number_threads)
                : m_integrating(a,b,n_intervals_integration,target_error,max_iterations), m_abscissa_points(abscissa_points), m_n(n), m_number_threads(number_threads) {}

    /*!
    * @brief Virtual destructor
    */
    virtual ~fgwr() = default;

    /*!
    * @brief Getter for the abscissa points
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
}

#endif  /*FGWR_PREDICT_HPP*/