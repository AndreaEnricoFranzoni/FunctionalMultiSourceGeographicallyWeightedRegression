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

#include "fdaPDE-core/fdaPDE/splines.h"


#ifdef _OPENMP
#include <omp.h>
#endif

/*!
* @file basis_systems.hpp
* @brief Contains the class to a system of a given type of basis
* @author Andrea Enrico Franzoni
*/


template< typename triangulation_, BASIS_TYPE basis_type > 
class basis_systems{

private:
    /*!Vector containing a basis system for each one of the functional covariates*/
    std::vector<fdapde::BsSpace<triangulation_>> m_systems_of_basis;

    /*!Number of basis systems: one for each covariate*/
    std::size_t m_q;


public:
    basis_systems(std::size_t q,
                  const std::vector<std::size_t> & order,
                  const std::vector<double> & knots)            :    m_q(q)
                     {
                        m_systems_of_basis.reserve(q);

                        for(std::size_t i = 0; i < q; ++i){
                            
                            //casting the nodes for the basis system
                            Eigen::Map<fdagwr_traits::Dense_Vector> nodes(knots.data(), knots.size(), 1);
                            fdapde::Triangulation<1, 1> interval(nodes);

                            //construct the basis system
                            m_systems_of_basis.emplace_back(interval,order[i]);
                        }  
                     }


    /*!
    * @brief Getter for the systems of basis
    * @return the private m_basis_system
    */
    std::vector<fdapde::BsSpace<triangulation_>> systems_of_basis() const {return m_systems_of_basis;}

    /*!
    * @brief Getter for the number of basis systems
    * @return the private m_q
    */
    std::size_t q() const {return m_q;}
};

#endif  /*FDAGWR_BASIS_SYSTEM_HPP*/