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


#ifndef FDAGWR_FUNCTIONAL_DATA_HPP
#define FDAGWR_FUNCTIONAL_DATA_HPP


#include "functional_datum.hpp"


/*!
* @brief The class describes n-statistical units referring to the same population: the basis system is the same for each one of them 
*/
template< typename domain = fdagwr_traits::Domain, typename basis_type = bsplines_basis<domain> >
class functional_data
{
private:
    /*!Number of statistical units*/
    std::size_t m_n;

    /*!Domain left extreme*/
    double m_a;

    /*!Domain right extreme*/
    double m_b;

    /*!Coefficient of datum basis expansion*/
    std::vector<functional_datum<domain,basis_type>> m_fdata;


public:
    /*!
    * @brief Constructor
    */
    template< typename _COEFF_OBJ_ >
    functional_data(_COEFF_OBJ_ && fdata_coeff,
                    const basis_type& fdata_basis)
        : 
            m_a(fdata_basis.knots().nodes()(0,0)),
            m_b(fdata_basis.knots().nodes()(fdata_basis.knots().nodes().size()-static_cast<std::size_t>(1),0)),
            m_n(fdata_coeff.cols())   
        {
            m_fdata.reserve(m_n);
            for(std::size_t i = 0; i < m_n; ++i){       m_fdata.emplace_back(std::move(fdata_coeff.col(i)),fdata_basis);}
        }

    const std::vector<functional_datum<domain,basis_type>>& fdata() const {return m_fdata;}

    /*!
    * @brief Getter for the number of statistical units
    */
    std::size_t n() const {return m_n;}

    /*!
    * @brief Evaluating the correct statistical unit
    */
    double
    eval(double loc, std::size_t unit_i)
    const
    {
        return m_fdata[unit_i].eval(loc);
    }

};

#endif  /*FDAGWR_FUNCTIONAL_DATA_HPP*/