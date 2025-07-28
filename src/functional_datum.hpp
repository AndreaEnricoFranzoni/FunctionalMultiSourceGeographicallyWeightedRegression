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


#ifndef FDAGWR_FUNCTIONAL_DATUM_HPP
#define FDAGWR_FUNCTIONAL_DATUM_HPP

#include "traits_fdagwr.hpp"
#include "basis_bspline.hpp"
#include "basis_constant.hpp"


template< class domain = fdagwr_traits::Domain, template <typename> class basis_type = bsplines_basis > 
    requires fdagwr_concepts::as_interval<domain>
class functional_datum
{
private:
    /*!Domain left extreme*/
    double m_a;

    /*!Domain right extreme*/
    double m_b;

    /*!Coefficient of datum basis expansion*/
    fdagwr_traits::Dense_Vector m_fdatum_coeff;

    /*!Basis of datum basis expansion*/
    basis_type<domain> m_fdatum_basis;

public:
    /*!
    * @brief Constructor
    */
    template< typename _COEFF_OBJ_ >
    functional_datum(_COEFF_OBJ_ && fdata_coeff,
                     const basis_type<domain>& fdata_basis)
        : 
            m_a(fdata_basis.knots().nodes()(0,0)),
            m_b(fdata_basis.knots().nodes()(fdata_basis.knots().nodes().size()-static_cast<std::size_t>(1),0)),
            m_fdatum_coeff{std::forward<_COEFF_OBJ_>(fdata_coeff)},
            m_fdatum_basis(fdata_basis)      
        {}

    /*!*/
    const basis_type<domain>& fdatum_basis() const {return m_fdatum_basis;}

    /*!
    * @brief evaluating the functional datum in location loc
    */
    double
    eval(double loc)
    const
    {
        //as it is in fdaPDE
        return m_fdatum_basis.eval_base(loc).row(0) * m_fdatum_coeff;
    }

};


#endif  /*FDAGWR_FUNCTIONAL_DATUM_HPP*/