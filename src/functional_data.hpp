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


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include "concepts_fdagwr.hpp"
#include "basis_include.hpp"



/*!
* @brief The class describes n-statistical units referring to the same population: the basis system is the same for each one of them 
*/
template< class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis > 
    requires fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
class functional_data
{
private:
    /*!Domain left extreme*/
    double m_a;
    /*!Domain right extreme*/
    double m_b;
    /*!Number of statistical units*/
    std::size_t m_n;
    /*!Coefficients of basis expansion*/
    FDAGWR_TRAITS::Dense_Matrix m_fdata_coeff;
    /*!Pointer to the basis*/
    std::unique_ptr<basis_type<domain_type>> m_fdata_basis;

public:
    /*!
    * @brief Constructor
    */
    template< typename _COEFF_OBJ_ >
    functional_data(_COEFF_OBJ_ && fdata_coeff,
                    std::unique_ptr<basis_type<domain_type>> fdata_basis)
            : 
                m_a(fdata_basis->knots().nodes()(0,0)),
                m_b(fdata_basis->knots().nodes()(fdata_basis->number_knots()-static_cast<std::size_t>(1),0)),
                m_n(fdata_coeff.cols()),
                m_fdata_coeff{std::forward<_COEFF_OBJ_>(fdata_coeff)},
                m_fdata_basis(std::move(fdata_basis))  
            {
                //checking that coefficients dimensions are consistent
                assert((void("Number of knots = number of basis - degree + 1"), m_fdata_coeff.rows() == m_fdata_basis->number_of_basis() ));
            }

    /*!
    * @todo SCRIVERE UN COPY CONSTRUCTOR PER GESTIRE IL PUNTATORE ALLA BASE
    */
    functional_data(functional_data<domain_type,basis_type> const &fd)
            : m_a(fd.a()), m_b(fd.b()), m_n(fd.n()), m_fdata_coeff(fd.fdata_coeff())
            {
                std::cout << "Creando fd con il copy constructor" << std::endl;
                m_fdata_basis = std::make_unique<basis_type<domain_type>>(fd.fdata_basis()->knots().nodes(),fd.fdata_basis()->degree(),fd.fdata_basis()->number_of_basis());
                //m_fdata_basis = std::make_unique<bsplines_basis<domain_type>>(fd.fdata_basis()->knots().nodes(),fd.fdata_basis()->degree(),fd.fdata_basis()->number_of_basis());
            }

    /*!
    * @brief Getter for the basis domain left extreme
    */
    double a() const {return m_a;}

    /*!
    * @brief Getter for the basis domain right extreme
    */
    double b() const {return m_b;}

    /*!
    * @brief Getter for the number of statistical units
    */
    std::size_t n() const {return m_n;}

    /*!
    * @brief Getter for the basis expansion coefficients
    */
    const FDAGWR_TRAITS::Dense_Matrix & fdata_coeff() const {return m_fdata_coeff;}

    /*!
    * @brief Getter for the basis 
    */
    const std::unique_ptr<basis_type<domain_type>> & fdata_basis() const {return m_fdata_basis;}

    /*!
    * @brief Evaluating the correct statistical unit
    */
    double
    eval(double loc, std::size_t unit_i)
    const
    {
        return m_fdata_basis->eval_base(loc).row(0) * m_fdata_coeff.col(unit_i);
    }

};

#endif  /*FDAGWR_FUNCTIONAL_DATA_HPP*/