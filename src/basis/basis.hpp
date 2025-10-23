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
// OUT OF OR IN CONNECTION WITH fdagwr OR THE USE OR OTHER DEALINGS IN
// fdagwr.


#ifndef FDAGWR_BASIS_HPP
#define FDAGWR_BASIS_HPP


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"
#include "../utility/concepts_fdagwr.hpp"

#include <cassert>


/*!
* @file basis.hpp
* @brief Contains the definition of basis base class
* @author Andrea Enrico Franzoni
*/



/*!
* @class basis_base_class
* @tparam domain_type the domain over which the basis is constructed
* @brief base class for basis. Virtual methods
*/
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
    /*!
    * Domain of the basis
    * @note the method .nodes() access to knots
    */
    domain_type m_knots;
    /*!Basis degree*/
    std::size_t m_degree;
    /*!Number of basis*/
    std::size_t m_number_of_basis;
    /*!Type of basis*/
    std::string m_type;

public:
    /*!
    * @brief Constructor
    * @param knots an Eigen::VectorXd containing the knots over which the basis is constructed
    * @param degree basis degree
    * @param number_of_basis number of basis
    */
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
    * @brief Getter for the domain over which the basis are constructed
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
    * @return string with the basis type name
    * @note Virtual method
    */
    virtual inline std::string type() const = 0;

    /*!
    * @brief Function to evaluate the basis in a given location
    * @param location the abscissa over which evaluating the basis system
    * @return an Eigen::MatrixXd of dimension 1 x m_number_of_basis that contains the evaluation of each basis in the location
    * @note Virtual method
    */
    virtual inline FDAGWR_TRAITS::Dense_Matrix eval_base(const double &location) const = 0;

    /*!
    * @brief Function to evaluate the basis over a set of locations
    * @param locations an Eigen::MatrixXd of dimension n_locs x 1 that contains the abscissa over which the basis have to be evaluated
    * @return an Eigen::SparseMatrix<double> of dimension n_locs x m_number_of_basis that contains, for each row, the evalaution of each basis in the respective location
    * @note virtual method
    */
    virtual inline FDAGWR_TRAITS::Sparse_Matrix eval_base_on_locs(const FDAGWR_TRAITS::Dense_Matrix &locations) const = 0;

    /*!
    * @brief Function to perform the smoothing over an evaluated functional datum, given the knots for the smoothing
    * @param f_ev an n_locs x 1 matrix with the evaluations of the fdata in correspondence of the smoothing knots
    * @param knots smoothing knots over which evaluating the basis, and for which it is available the evaluation of the functional datum
    * @return a dense matrix of dimension m_number_of_basis x 1, with the coefficients of the basis expansion
    * @note virtual method
    */
    virtual inline FDAGWR_TRAITS::Dense_Matrix smoothing(const FDAGWR_TRAITS::Dense_Matrix & f_ev, const FDAGWR_TRAITS::Dense_Matrix & knots) const = 0;
};


#endif  /*FDAGWR_BASIS_HPP*/