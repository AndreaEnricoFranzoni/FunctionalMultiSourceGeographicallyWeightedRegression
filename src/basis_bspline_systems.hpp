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


#include "basis_include.hpp"


/*!
* @file basis_systems.hpp
* @brief Contains the class to a system of a given type of basis
* @author Andrea Enrico Franzoni
*/


/*!
* @todo SERVIREBBE UN CONCEPT PER IL TIPO DOMAIN
*/
template< class domain = fdagwr_traits::Domain, template <typename> class basis_type = bsplines_basis > 
    requires fdagwr_concepts::as_interval<domain> && fdagwr_concepts::as_basis<basis_type>
class basis_systems{


private:
    /*!Nodes over which the basis systems are constructed*/
    domain m_knots;

    /*!Order of basis for each covariate*/
    std::vector<std::size_t> m_basis_degrees;

    /*!Number of basis for each covariate*/
    std::vector<std::size_t> m_numbers_of_basis;

    /*!Number of functional covariates: one basis system for each one of them*/
    std::size_t m_q;

    /*!Vector containing a basis system for each one of the functional covariates*/
    std::vector<basis_type<domain>> m_systems_of_basis;

public:
    /*!
    * @brief Class constructor for a vector of bsplines
    * @note BASIS ORDERS HAVE TO BE INT >=0 !!!!!!!!
    */
    basis_systems(const fdagwr_traits::Dense_Vector & knots,
                  const std::vector<std::size_t> & basis_degrees,
                  const std::vector<std::size_t> & numbers_of_basis,
                  std::size_t q)            
                  :    
                        m_knots(knots),
                        m_basis_degrees(basis_degrees),
                        m_numbers_of_basis(numbers_of_basis),
                        m_q(q)
                     {
                        //constructing systems of bsplines given knots and orders of the basis  
                        m_systems_of_basis.reserve(m_q);
                        for(std::size_t i = 0; i < m_q; ++i){  m_systems_of_basis.emplace_back(m_knots.nodes(), m_basis_degrees[i], m_numbers_of_basis[i]);}
                     }

    /*!
    * @brief Getter for the nodes over which the basis systems are constructed
    */
    const domain& knots() const {return m_knots;}

    /*!
    * @brief Getter for the systems of basis (returning a reference since fdaPDE stores the basis as a pointer to them)
    * @return the private m_systems_of_basis
    */
    const std::vector<basis_type<domain>>& systems_of_basis() const {return m_systems_of_basis;}

    /*!
    * @brief Getter for the order of basis for each covariate
    * @return the private m_basis_orders
    */
    const std::vector<std::size_t>& basis_degrees() const {return m_basis_degrees;}

    /*!
    * @brief Getter for the number of basis for each covariate
    * @return the private m_number_of_basis
    */
    const std::vector<std::size_t>& numbers_of_basis() const {return m_numbers_of_basis;}

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
        return m_systems_of_basis[basis_i].eval(location);
    }
};

#endif  /*FDAGWR_BASIS_SYSTEM_HPP*/