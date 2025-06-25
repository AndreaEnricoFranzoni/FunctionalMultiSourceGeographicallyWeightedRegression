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


/*!
* @file basis_systems.hpp
* @brief Contains the class to a system of a given type of basis
* @author Andrea Enrico Franzoni
*/


template< typename domain_structure, BASIS_TYPE basis_type > 
class basis_systems{

private:
    /*!Vector containing a basis system for each one of the functional covariates*/
    std::vector< fdapde::BsSpace<domain_structure> > m_systems_of_basis;

    /*!Number of basis systems: one for each covariate*/
    std::size_t m_q;


public:

    basis_systems(const std::vector<double> & knots,
                  const std::vector<int> & basis_orders,
                  std::size_t q)            :    m_q(q)
                     {
                        std::cout << "Nel costruttore di basis_systems" << std::endl;
                        m_systems_of_basis.reserve(q);

                        //casting the nodes for the basis system
                        Eigen::Map<const fdagwr_traits::Dense_Vector> nodes(knots.data(), knots.size());
                        std::cout << "Nodi mappati:" << std::endl;
                        for (size_t j = 0; j < knots.size(); ++j)
                        {
                            std::cout << nodes(j) << std::endl;
                        }
                        

                        domain_structure interval(nodes);
                        std::cout << "Intervallo generato" << std::endl;

                        //construct the basis system
                        for(std::size_t i = 0; i < q; ++i){    

                            int order = basis_orders[i];
                            fdapde::BsSpace<domain_structure> Vh(interval, order);
                            std::cout << "i: " << i << std::endl;
                            m_systems_of_basis.push_back(Vh);
                            //m_systems_of_basis.emplace_back(interval,basis_orders[i]);
                            std::cout << "base creata" << std::endl;
                            }  
                     }


    /*!
    * @brief Getter for the systems of basis
    * @return the private m_basis_system
    */
    std::vector< fdapde::BsSpace<domain_structure> > systems_of_basis() const {return m_systems_of_basis;}

    /*!
    * @brief Getter for the number of basis systems
    * @return the private m_q
    */
    std::size_t q() const {return m_q;}
};

#endif  /*FDAGWR_BASIS_SYSTEM_HPP*/