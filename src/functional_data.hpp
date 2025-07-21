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

#include "traits_fdagwr.hpp"
#include "basis_bspline.hpp"
#include "basis_constant.hpp"


template< typename domain = fdagwr_traits::Domain, typename basis_type = bsplines_basis<domain> >
class functional_data
{
private:
    /*!Domain left extreme*/
    double m_a;

    /*!Domain right extreme*/
    double m_b;

    /*!Coefficient of datum basis expnasion*/
    fdagwr_traits::Dense_Matrix m_fdata_coeff;

    /*!Basis of datum basis expansion*/
    basis_type m_fdata_basis;

public:
    /*!*/
    template< typename _COEFF_OBJ_, typename _BASIS_OBJ_ >
    functional_data(_COEFF_OBJ_ && fdata_coeff,
                    _BASIS_OBJ_ && fdata_basis)
                    : 
                        m_a(fdata_basis.nodes()(0,0)),
                        m_b(fdata_basis.nodes()(fdata_basis.nodes().size()-static_cast<std::size_t>(1),0)),
                        m_fdata_coeff{std::forward(fdata_coeff)},
                        m_fdata_basis{std::forward(fdata_basis)}
                        {
                            std::cout << "Nel functional_data class" << std::endl;
                            auto el = m_fdata_basis.eval_base(0);
                            std::cout << el << std::endl;
                        }

};


#endif  /*FDAGWR_FUNCTIONAL_DATA_HPP*/