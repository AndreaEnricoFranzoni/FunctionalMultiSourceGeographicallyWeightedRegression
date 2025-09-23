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


#ifndef FDAGWR_CONSTANT_BASIS_HPP
#define FDAGWR_CONSTANT_BASIS_HPP


#include "basis.hpp"


/*!
* @brief Class for constant basis: a straight line at y=1 all along the fd domain
*/
template< typename domain_type = FDAGWR_TRAITS::basis_geometry >
    requires fdagwr_concepts::as_interval<domain_type>
class constant_basis :  public basis_base_class<domain_type>
{
public:
    /*!Degree*/
    static constexpr std::size_t degree_constant_basis = 0;
    /*!Number of basis*/
    static constexpr std::size_t number_of_basis_constant_basis = 1;
    
    /*!Constructor*/
    constant_basis(const FDAGWR_TRAITS::Dense_Vector & knots,
                   std::size_t,
                   std::size_t)    
            :  
                basis_base_class<domain_type>(knots,constant_basis<domain_type>::degree_constant_basis,constant_basis<domain_type>::number_of_basis_constant_basis)
            {}

    /*!
    * @brief Giving the basis type
    * @return std::string
    */
    inline
    std::string 
    type()
    const 
    override
    {
        return "Constant";
    }

    /*!
    * @brief evaluating the system of basis basis_i-th in location location. Overriding the method
    */
    inline 
    FDAGWR_TRAITS::Dense_Matrix 
    eval_base(const double &location) 
    const
    override
    {   
        //wrap the output into a dense matrix: HA UNA RIGA, N_BASIS COLONNE
        return FDAGWR_TRAITS::Dense_Matrix::Ones(1,1);
    }

    /*!
    * @brief evaluating the basis basis_i over a set of locations. Overriding the method
    * @note locations è una FDAGWR_TRAITS::Dense_Matrix of dimensions n_locs x 1
    */
    inline 
    FDAGWR_TRAITS::Sparse_Matrix 
    eval_base_on_locs(const FDAGWR_TRAITS::Dense_Matrix &locations) 
    const
    override
    {
        FDAGWR_TRAITS::Dense_Matrix evals = FDAGWR_TRAITS::Dense_Matrix:Ones(locations.rows(), 1);
        return evals.sparseView();  // conversione a SparseMatrix
    }
};

#endif  /*FDAGWR_CONSTANT_BASIS_HPP*/