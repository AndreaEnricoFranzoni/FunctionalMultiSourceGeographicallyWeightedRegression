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


#ifndef FDAGWR_BSPLINES_BASIS_HPP
#define FDAGWR_BSPLINES_BASIS_HPP


#include "traits_fdagwr.hpp"
#include "bsplines_evaluation.hpp"


template< typename domain >
class bsplines_basis
{

/*!
* @brief Alias for the basis space
* @note calling the constructor of BsSpace in the constructor of the class
*/
using BasisSpace = fdapde::BsSpace<domain>;


private:
    /*!Basis*/
    BasisSpace m_basis;

    /*!Knots*/
    domain m_knots;

    /*!Domain left extreme*/
    double m_a;

    /*!Domain right extreme*/
    double m_b;

    /*!Bsplines degree*/
    std::size_t m_degree;

    /*!Number of bsplines*/
    std::size_t m_number_of_basis;

public:
    /*!Constructor*/
    bsplines_basis(const fdagwr_traits::Dense_Vector & knots,
                   std::size_t bsplines_degree,
                   std::size_t number_of_bsplines)    :   
                        m_knots(knots), 
                        m_a(knots.coeff(0)), 
                        m_b(m_knots.coeff(knots.size()-static_cast<std::size_t>(1))),
                        m_degree(bsplines_degree),
                        m_number_of_basis(number_of_bsplines)
                            {
                                //cheack input consistency
                                assert((void("Number of knots = number of basis - degree + 1"), m_knots.size() == (m_number_of_basis - m_degree + static_cast<std::size_t>(1))));
                                //construct the basis object
                                m_basis(m_knots,m_degree);
                            }

    /*!
    * @brief Getter for the basis
    * @return the private m_basis
    */
    const BasisSpace& basis() const {return m_basis;}

    /*!
    * @brief Getter for the nodes over which the basis are constructed
    */
    const domain& knots() const {return m_knots;}

    /*!
    * @brief Getter for the degree of the basis
    */
    std::size_t degree() const {return m_degree;}

    /*!
    * @brief Getter for the number of basis
    */
    std::size_t number_of_basis() const {return m_number_of_basis;}

    /*!
    * @brief evaluating the system of basis basis_i-th in location location
    */
    inline 
    fdagwr_traits::Dense_Matrix 
    eval_base(double location) 
    const
    {
        //check where the point has to be evaluated     TODO: TOGLIERE IL CHECK PER DISCORSI DI EFFICIENZA?
        if (location < m_a || location > m_b)
        {
            std::string error_message = "The bspline basis can be evaluated only inside its domain, [" + std::to_string(m_a) + "," + std::to_string(m_b) + "]";
            throw std::invalid_argument(error_message);
        }
        //wrap the input into a coherent object for the spline evaluation
        fdagwr_traits::Dense_Matrix loc = fdagwr_traits::Dense_Matrix::Constant(1, 1, location);
        //wrap the output into a dense matrix
        // HA UNA RIGA, N_BASIS COLONNE
        return fdagwr_traits::Dense_Matrix(bsplines_basis_evaluation<domain>(m_basis, loc));
    }
};

#endif  /*FDAGWR_BSPLINES_BASIS_HPP*/