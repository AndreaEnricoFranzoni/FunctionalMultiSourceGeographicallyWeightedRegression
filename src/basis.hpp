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


#ifndef FDAGWR_BASIS_HPP
#define FDAGWR_BASIS_HPP


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include "concepts_fdagwr.hpp"


template< typename domain_type = FDAGWR_TRAITS::basis_geometry >
    requires fdagwr_concepts::as_interval<domain_type>
class basis_base_class
{
private:
    /*!Domain left extreme*/
    double m_a;
    /*!Domain right extreme*/
    double m_b;
    /*!Knots size*/
    std::size_t m_number_knots;
    /*!Knots*/
    domain_type m_knots;
    /*!Basis degree*/
    std::size_t m_degree;
    /*!Number of basis*/
    std::size_t m_number_of_basis;
    /*!Type of basis*/
    std::string m_type;

public:
    /*!Constructor*/
    basis_base_class(const FDAGWR_TRAITS::Dense_Vector & knots,
                    std::size_t degree,
                    std::size_t number_of_basis)    
            :   
                m_a(knots.coeff(0)), 
                m_b(knots.coeff(knots.size()-static_cast<std::size_t>(1))), 
                m_number_knots(knots.size()),
                m_knots(knots),
                m_degree(degree),
                m_number_of_basis(number_of_basis)  
            {}

    /*! 
    * @brief virtual destructor, for polymorphism
    */
    virtual ~basis_base_class() = default;

    /*!
    * @brief Getter for the basis domain left extreme
    */
    double a() const {return m_a;}

    /*!
    * @brief Getter for the basis domain right extreme
    */
    double b() const {return m_b;}

    /*!
    * @brief Getter for the number of knots
    */
    std::size_t number_knots() const {return m_number_knots;}

    /*!
    * @brief Getter for the nodes over which the basis are constructed
    */
    const domain_type& knots() const {return m_knots;}

    /*!
    * @brief Getter for the degree of the basis
    */
    std::size_t degree() const {return m_degree;}

    /*!
    * @brief Getter for the number of basis
    */
    std::size_t number_of_basis() const {return m_number_of_basis;}

    /*!
    * @brief Basis type
    */
    virtual inline std::string type() const = 0;

    /*!
    * @brief Abstract function to evaluate the basis in a location
    */
    virtual inline FDAGWR_TRAITS::Dense_Matrix eval_base(const double &location) const = 0;
};


#endif  /*FDAGWR_BASIS_HPP*/