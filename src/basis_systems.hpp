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


#ifndef FDAGWR_BASIS_SYSTEM_HPP
#define FDAGWR_BASIS_SYSTEM_HPP


#include "traits_fdagwr.hpp"
#include "basis_evaluation.hpp"


/*!
* @file basis_systems.hpp
* @brief Contains the class to a system of a given type of basis
* @author Andrea Enrico Franzoni
*/


/*!
* @todo SERVIREBBE UN CONCEPT PER IL TIPO DOMAIN
*/
template< typename domain = fdagwr_traits::Domain, BASIS_TYPE basis_type = BASIS_TYPE::BSPLINES > 
class basis_systems{

/*!
* @brief Alias for the basis space
* @note calling the constructor of BsSpace in the constructor of the class
*/
using BasisSpace = fdapde::BsSpace<domain>;


private:
    /*!Vector containing a basis system for each one of the functional covariates*/
    std::vector<BasisSpace> m_systems_of_basis;

    /*!Nodes over which the basis systems are constructed*/
    domain m_interval;

    /*!Order of basis for each covariate*/
    std::vector<std::size_t> m_basis_orders;

    /*!Number of basis for each covariate*/
    std::vector<std::size_t> m_number_of_basis;

    /*!Number of functional covariates: one basis system for each one of them*/
    std::size_t m_q;

public:
    /*!
    * @brief Class constructor
    * @note BASIS ORDERS HAVE TO BE INT >=1 !!!!!!!!
    */
    basis_systems(const fdagwr_traits::Dense_Vector & knots,
                  const std::vector<std::size_t> & basis_orders,
                  const std::vector<std::size_t> & number_of_basis,
                  std::size_t q)            
                  :    
                        m_interval(knots),
                        m_basis_orders(basis_orders),
                        m_number_of_basis(number_of_basis),
                        m_q(q)
                     {
                        //constructing systems of bsplines given knots and orders of the basis  
                        m_systems_of_basis.reserve(m_q);
                        for(std::size_t i = 0; i < m_q; ++i){  m_systems_of_basis.emplace_back(m_interval, m_basis_orders[i]);}
                     }

    /*!
    * @brief Getter for the nodes over which the basis systems are constructed
    */
    const domain& interval() const {return m_interval;}

    /*!
    * @brief Getter for the systems of basis (returning a reference since fdaPDE stores the basis as a pointer to them)
    * @return the private m_systems_of_basis
    */
    const std::vector<BasisSpace>& systems_of_basis() const {return m_systems_of_basis;}

    /*!
    * @brief Getter for the order of basis for each covariate
    * @return the private m_basis_orders
    */
    const std::vector<std::size_t>& basis_orders() const {return m_basis_orders;}

    /*!
    * @brief Getter for the number of basis for each covariate
    * @return the private m_number_of_basis
    */
    const std::vector<std::size_t>& number_of_basis() const {return m_number_of_basis;}

    /*!
    * @brief Getter for the number of basis systems
    * @return the private m_q
    */
    std::size_t q() const {return m_q;}

    /*!
    * @brief evaluating the system of basis basis_i-th in location location
    */
    inline 
    fdagwr_traits::Dense_Matrix 
    eval_base(double location, std::size_t basis_i) 
    const
    {
        //wrap the input into a coherent object for the spline evaluation
        fdagwr_traits::Dense_Matrix loc = fdagwr_traits::Dense_Matrix::Constant(1, 1, location);
        //wrap the output into a dense matrix
        // HA UNA RIGA, N_BASIS COLONNE
        return fdagwr_traits::Dense_Matrix(spline_basis_evaluation<domain>(m_systems_of_basis[basis_i], loc));
    }
};

#endif  /*FDAGWR_BASIS_SYSTEM_HPP*/