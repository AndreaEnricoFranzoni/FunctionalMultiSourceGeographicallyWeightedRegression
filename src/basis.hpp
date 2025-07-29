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


#include "traits_fdagwr.hpp"
#include "concepts_fdagwr.hpp"


template< typename domain_type = fdagwr_traits::Domain >
    requires fdagwr_concepts::as_interval<domain_type>
class basis_base_class
{
private:
    /*!Domain left extreme*/
    double m_a;
    /*!Domain right extreme*/
    double m_b;
    /*!Knots*/
    domain_type m_knots;

public:
    /*!Default constructor*/
    basis_base_class() = default;
    /*!Constructor*/
    basis_base_class(const fdagwr_traits::Dense_Vector & knots)    
        :   m_a(knots.coeff(0)), m_b(knots.coeff(knots.size()-static_cast<std::size_t>(1))), m_knots(knots)  {}

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
    std::size_t number_knots() const {return m_knots.size();}

    /*!
    * @brief Getter for the nodes over which the basis are constructed
    */
    const domain_type& knots() const {return m_knots;}

    /*!
    * @brief Abstract function to evaluate the basis in a location
    */
    virtual inline fdagwr_traits::Dense_Matrix eval_base(double location) const = 0;
};


#endif  /*FDAGWR_BASIS_HPP*/