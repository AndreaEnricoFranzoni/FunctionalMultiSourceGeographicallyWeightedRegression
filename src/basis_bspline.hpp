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


#include "basis.hpp"
#include "bsplines_evaluation.hpp"


/*!
* @brief class for bsplines basis
*/
template< typename domain_type = FDAGWR_TRAITS::basis_geometry >
    requires fdagwr_concepts::as_interval<domain_type>
class bsplines_basis :  public basis_base_class<domain_type>
{

/*!
* @brief Alias for the basis space
* @note calling the constructor of BsSpace in the constructor of the class
*/
using BasisSpace = fdapde::BsSpace<domain_type>;

private:
    /*!Bslines*/
    BasisSpace m_basis;

public:
    /*!Default(cubic bsplines)*/
    static constexpr std::size_t bsplines_degree_default = static_cast<std::size_t>(3); 

    /*!Constructor*/
    bsplines_basis(const FDAGWR_TRAITS::Dense_Vector & knots,
                   std::size_t degree,
                   std::size_t number_of_basis)    
            :   
                basis_base_class<domain_type>(knots,degree,number_of_basis),
                m_basis(this->knots(),this->degree())
                {
                    //cheack input consistency
                    assert((void("Number of knots = number of basis - degree + 1"), this->number_knots() == (m_number_of_basis - m_degree + static_cast<std::size_t>(1))));
                }


    /*!
    * @brief Getter for the basis
    * @return the private m_basis
    */
    const BasisSpace& basis() const {return m_basis;}

    /*!
    * @brief evaluating the system of basis basis_i-th in location location. Overriding the method
    */
    inline 
    FDAGWR_TRAITS::Dense_Matrix 
    eval_base(double location) 
    const
    override
    {
        //wrap the input into a coherent object for the spline evaluation
        FDAGWR_TRAITS::Dense_Matrix loc = FDAGWR_TRAITS::Dense_Matrix::Constant(1, 1, location);
        //wrap the output into a dense matrix:      HA UNA RIGA, N_BASIS COLONNE
        return FDAGWR_TRAITS::Dense_Matrix(bsplines_basis_evaluation<domain_type>(m_basis, loc));
    }
};

#endif  /*FDAGWR_BSPLINES_BASIS_HPP*/