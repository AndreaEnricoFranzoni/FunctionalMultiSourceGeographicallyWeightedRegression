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

#ifndef FDAGWR_PENALTIES_POLICIES_HPP
#define FDAGWR_PENALTIES_POLICIES_HPP

#include "../utility/traits_fdagwr.hpp"
#include "../basis/basis_bspline_systems.hpp"


/*!
* @file penalization_matrix_penalties_policies.hpp
* @brief Contains the different types of penalization computations, penalizing different covariates
* @author Andrea Enrico Franzoni
*/


/*!
* @struct SecondDerivativePenalty
* @brief Functor computing the scalar product within second order derivatives of the basis
*/
struct SecondDerivativePenalty
{ 
  /*!
  * @brief Compute the scalar product of the second order derivative within the basis
  * @param bs a basis_systems of bsplines
  * @param system_number the number of the basis system
  */
  Eigen::SparseMatrix<double> 
  operator()(const basis_systems< FDAGWR_TRAITS::basis_geometry, bsplines_basis > &bs, std::size_t system_number) 
  const
  {
    //using fdaPDE
    fdapde::TrialFunction u(bs.systems_of_basis()[system_number].basis()); 
    fdapde::TestFunction  v(bs.systems_of_basis()[system_number].basis());
    // stiff matrix: penalizing the second derivaive
    auto stiff = integral(bs.knots())(dxx(u) * dxx(v));
    
    //assmebling the stiff matrix 
    return stiff.assemble();
  }
};



/*!
* @struct FirstDerivativePenalty
* @brief Functor computing the scalar product within first order derivatives of the basis
*/
struct FirstDerivativePenalty
{ 
  /*!
  * @brief Compute the scalar product of the first order derivative within the basis
  * @param bs a basis_systems of bsplines
  * @param system_number the number of the basis system
  */
  Eigen::SparseMatrix<double> 
  operator()(const basis_systems< FDAGWR_TRAITS::basis_geometry, bsplines_basis > &bs, std::size_t system_number) 
  const
  {
    //using fdaPDE
    fdapde::TrialFunction u(bs.systems_of_basis()[system_number].basis()); 
    fdapde::TestFunction  v(bs.systems_of_basis()[system_number].basis());
    // first_derivative_penalty matrix: penalizing the first order derivaive
    auto first_derivative_penalty = integral(bs.knots())(dx(u) * dx(v));

    //assembling the first_derivative_penalty matrix
    return first_derivative_penalty.assemble();
  }
};



/*!
* @struct ZeroDerivativePenalty
* @brief Functor computing the scalar product within zero order derivatives of the basis
*/
struct ZeroDerivativePenalty
{ 
  /*!
  * @brief Compute the scalar product of the zero order derivative within the basis
  * @param bs a basis_systems of bsplines
  * @param system_number the number of the basis system
  */
  Eigen::SparseMatrix<double> 
  operator()(const basis_systems< FDAGWR_TRAITS::basis_geometry, bsplines_basis > &bs, std::size_t system_number) 
  const
  {
    //using fdaPDE
    fdapde::TrialFunction u(bs.systems_of_basis()[system_number].basis()); 
    fdapde::TestFunction  v(bs.systems_of_basis()[system_number].basis());
    // mass matrix: penalizing the zero order derivaive
    auto mass = integral(bs.knots())(u * v);

    //assembling the mass matrix
    return mass.assemble();
  }
};


#endif  /*FDAGWR_PENALTIES_POLICIES_HPP*/