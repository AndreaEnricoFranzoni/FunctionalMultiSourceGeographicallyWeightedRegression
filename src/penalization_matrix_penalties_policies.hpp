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

#ifndef FDAGWR_PENALTIES_POLICIES_HPP
#define FDAGWR_PENALTIES_POLICIES_HPP

#include "traits_fdagwr.hpp"
#include "basis_systems.hpp"


/*!
* @file penalization_matrix_penalties.hpp
* @author Andrea Enrico Franzoni
*/


/*!
* @struct Case
* @brief Functor returning the values for which computing ratio for computing ADF-test statistic. No lag orders bigger than one considered.
*/
struct ZeroDerivativePenalty
{ 
  Eigen::SparseMatrix<double> 
  operator()(const basis_systems &bs) 
  const
  {
    std::cout << "0 der pen" << std::endl;
    fdapde::TrialFunction u(bs.systems_of_basis()[i]); 
    fdapde::TestFunction  v(bs.systems_of_basis()[i]);
    
    // mass matrix: penalizing the zero order derivaive
    auto mass = integral(bs.interval())(d(u) * d(v));
    Eigen::SparseMatrix<double> M = stiff.assemble();
    
    return M;
  }
};

/*!
* @struct Case
* @brief Functor returning the values for which computing ratio for computing ADF-test statistic. No lag orders bigger than one considered.
*/
struct SecondDerivativePenalty
{ 
  Eigen::SparseMatrix<double> 
  operator()(const basis_systems &bs) 
  const
  {
    std::cout << "2 der pen" << std::endl;
    fdapde::TrialFunction u(bs.systems_of_basis()[i]); 
    fdapde::TestFunction  v(bs.systems_of_basis()[i]);
    
    // stiff matrix: penalizing the second derivaive
    auto stiff = integral(bs.interval())(dxx(u) * dxx(v));
    Eigen::SparseMatrix<double> M = stiff.assemble();
    
    return M;
  }
};


#endif  /*FDAGWR_PENALTIES_POLICIES_HPP*/