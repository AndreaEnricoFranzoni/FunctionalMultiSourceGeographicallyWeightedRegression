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


#ifndef FDAGWR_CONSTANT_BASIS_HPP
#define FDAGWR_CONSTANT_BASIS_HPP


#include "basis.hpp"


/*!
* @file basis_constant.hpp
* @brief Contains the definition of constant basis derived class
* @author Andrea Enrico Franzoni
*/


/*!
* @class constant_basis
* @tparam domain_type the domain over which the basis is constructed
* @brief derived class for constant basis
*/
template< typename domain_type = FDAGWR_TRAITS::basis_geometry >
    requires fdagwr_concepts::as_interval<domain_type>
class constant_basis :  public basis_base_class<domain_type>
{

public:
    /*!Basis degree*/
    static constexpr std::size_t degree_constant_basis = 0;
    /*!Number of basis*/
    static constexpr std::size_t number_of_basis_constant_basis = 1;
    
    /*!
    * @brief Constructor
    * @param knots Eigen::VectorXd with the knots for constructing the basis
    */
    constant_basis(const FDAGWR_TRAITS::Dense_Vector & knots,
                   std::size_t,
                   std::size_t)    
            :  
                basis_base_class<domain_type>(knots,constant_basis<domain_type>::degree_constant_basis,constant_basis<domain_type>::number_of_basis_constant_basis)
            {}

    /*!
    * @brief Copy constructor
    */
    constant_basis(const constant_basis&) = default;

    /*!
    * @brief Move constructor
    */
    constant_basis(constant_basis&&) noexcept = default;

    /*!
    * @brief Copy assignment
    */
    constant_basis& operator=(const constant_basis&) = default;

    /*!
    * @brief Move assignment
    */
    constant_basis& operator=(constant_basis&&) noexcept = default;

    /*!
    * @brief Basis type
    * @return string with the basis type name
    */
    inline
    std::string 
    type()
    const 
    override
    {
        return "constant";
    }

    /*!
    * @brief Function to evaluate the basis in a given location
    * @param location the abscissa over which evaluating the basis system
    * @return an Eigen::MatrixXd of dimension 1 x m_number_of_basis that contains the evaluation of each basis in the location
    */
    inline 
    FDAGWR_TRAITS::Dense_Matrix 
    eval_base(const double &location) 
    const
    override
    {   
        //wrap the output into a dense matrix: one row, m_number_of_basis(1) cols
        return FDAGWR_TRAITS::Dense_Matrix::Ones(1,1);  //it is a constant basis
    }

    /*!
    * @brief Function to evaluate the basis over a set of locations
    * @param locations an Eigen::MatrixXd of dimension n_locs x 1 that contains the abscissa over which the basis have to be evaluated
    * @return an Eigen::SparseMatrix<double> of dimension n_locs x m_number_of_basis that contains, for each row, the evalaution of each basis in the respective location
    */
    inline 
    FDAGWR_TRAITS::Sparse_Matrix 
    eval_base_on_locs(const FDAGWR_TRAITS::Dense_Matrix &locations) 
    const
    override
    {
        //dim: n_locs x 1 (only one basis)
        FDAGWR_TRAITS::Dense_Matrix evals = FDAGWR_TRAITS::Dense_Matrix::Ones(locations.rows(), 1);
        return evals.sparseView();  // conversion to Eigen::SparseMatrix
    }

    /*!
    * @brief Function to perform the smoothing over an evaluated functional datum, given the knots for the smoothing
    * @param f_ev an n_locs x 1 matrix with the evaluations of the fdata in correspondence of the smoothing knots
    * @param knots smoothing knots over which evaluating the basis, and for which it is available the evaluation of the functional datum
    * @return a dense matrix of dimension m_number_of_basis x 1, with the coefficients of the basis expansion
    */
    inline
    FDAGWR_TRAITS::Dense_Matrix
    smoothing(const FDAGWR_TRAITS::Dense_Matrix & f_ev, 
              const FDAGWR_TRAITS::Dense_Matrix & knots) 
    const
    override
    {
        assert((f_ev.rows() == knots.rows()) && (f_ev.cols() == 1) && (knots.cols() == 1));

        //for constant basis, the smoothing is simply the mean value of the function along the domain
        std::size_t number_knots = knots.rows();
        FDAGWR_TRAITS::Dense_Matrix c = FDAGWR_TRAITS::Dense_Matrix::Zero(1,1);
        for(std::size_t i = 0; i < number_knots; ++i){    c(0,0) += f_ev(i,0);}

        return c/number_knots;
    }
};

#endif  /*FDAGWR_CONSTANT_BASIS_HPP*/