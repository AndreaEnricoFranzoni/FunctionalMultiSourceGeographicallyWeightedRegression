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
#include "functional_matrix_operators.hpp"


/*!
* @brief Virtual interface to perform the 
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fgwr
{
private:
    /*!Number of threads for OMP*/
    int m_number_threads;

public:
    /*!
    * @brief Constructor
    * @param number_threads number of threads for OMP
    */
    fgwr(int number_threads): m_number_threads(number_threads) {}

    /*!
    * @brief Virtual destructor
    */
    virtual ~fgwr() = default;

    /*!
    * @brief Getter for the number of threads for OMP
    * @return the private m_number_threads
    */
    inline int number_threads() const {return m_number_threads;}

    /*!
    * @brief Compute all the [J_2_tilde_i + R]^(-1)
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix > 
    compute_penalty(const functional_matrix<INPUT,OUTPUT> &base,
                    const functional_matrix<INPUT,OUTPUT> &X,
                    const std::vector< functional_matrix<INPUT,OUTPUT> > &W,
                    const FDAGWR_TRAITS::Dense_Matrix &R) const;

    /*!
    * @brief Compute [J_tilde_i + R]^(-1)
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed,
                    const std::vector< functional_matrix<INPUT,OUTPUT> > &W,
                    const FDAGWR_TRAITS::Dense_Matrix &R) const;

    /*!
    * @brief Compute [J + Rc]^(-1)
    */
    FDAGWR_TRAITS::Dense_Matrix
    compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed,
                    const functional_matrix<INPUT,OUTPUT> &W,
                    const FDAGWR_TRAITS::Dense_Matrix &R) const;

    /*!
    * @brief Compute an operator
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const functional_matrix<INPUT,OUTPUT> &rhs,
                     const std::vector< FDAGWR_TRAITS::Dense_Matrix > &penalty) const;

    /*!
    * @brief Compute an operator
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &rhs,
                     const std::vector< FDAGWR_TRAITS::Dense_Matrix > &penalty) const;

    /*!
    * @brief Compute an operator
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &lhs,
                     const std::vector< functional_matrix<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const functional_matrix<INPUT,OUTPUT> &rhs,
                     const std::vector< FDAGWR_TRAITS::Dense_Matrix > &penalty) const;

    /*!
    * @brief Compute an operator
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &lhs,
                     const std::vector< functional_matrix<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &rhs,
                     const std::vector< FDAGWR_TRAITS::Dense_Matrix > &penalty) const;

    /*!
    * @brief Compute a functional operator
    */
    functional_matrix<INPUT,OUTPUT> 
    compute_functional_operator(const functional_matrix<INPUT,OUTPUT> &X,
                                const functional_matrix<INPUT,OUTPUT> &base,
                                const FDAGWR_TRAITS::Dense_Matrix &operator_) const;

    /*!
    * @brief Virtual method to compute the Functional Geographically Weighted Regression
    */
    virtual inline void compute() const = 0;
};

#endif  /*FGWR_ALGO_HPP*/