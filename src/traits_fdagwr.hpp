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


#ifndef FDAGWR_TRAITS_HPP
#define FDAGWR_TRAITS_HPP


#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "fdaPDE-core/fdaPDE/splines.h"

#include <vector>
#include <array>
#include <tuple>
#include <map>
#include <variant>
#include <type_traits>
#include <cmath>
#include <string>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif


#include <iostream>


/*!
* @file traits_fdagwr.hpp
* @brief Contains customized types and enumerator for customized template parameters, used during the model's fitting
* @author Andrea Enrico Franzoni
*/



/*!
* @struct fdagwr_traits
* @brief Contains the customized types for fts, covariances, PPCs, etc...
* @details Data are stored in dynamic matrices (easily very big dimensions) of doubles
*/
struct fdagwr_traits
{
public:
  
  using Dense_Matrix  = Eigen::MatrixXd;                 ///< Matrix data structure.

  using Sparse_Matrix  = Eigen::SparseMatrix<double>;    ///< Sparse matrix data structure.
  
  using Dense_Vector  = Eigen::VectorXd;                 ///< Vector data structure.
  
  using Dense_Array   = Eigen::ArrayXd;                  ///< Array data structure: more efficient for coefficient-wise operations.

};


struct FDAGWR_FEATS
{
  using FDAGWR_DOMAIN = fdapde::Triangulation<1, 1>;     ///< Domain mesh: unit interval with a fixed number of nodes
};


/*!
* @enum FDAGWR_COVARIATES_TYPES
* @brief different types of functional covariates
*/
enum FDAGWR_COVARIATES_TYPES
{
  STATIONARY = 0,       ///< Covariates not depending on the geographical location
  NON_STATIONARY = 1,   ///< Covariates depending on the geographical location
  EVENT = 2,            ///< Covariates depending on the event geographical location
  STATION = 3,          ///< Covariates not depending on the station geographical location
};


template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
std::string
covariate_conversion()
{
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::STATIONARY )    {   return "Stationary";}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::NON_STATIONARY ){   return "Nonstationary";}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::EVENT )         {   return "Event";}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::STATION )       {   return "Station";}
};



/*!
* @enum KERNEL_FUNC
* @brief Kernel for evaluating the distances within different locations. Functions defined in "kernel_functions.hpp"
*/
enum KERNEL_FUNC
{
  GAUSSIAN = 0,  ///< Gaussian Kernel to evaluate the distances within different locations
};



/*!
* @enum BASIS_TYPE
* @brief type of basis for reconstructing the functional data
*/
enum BASIS_TYPE
{
    BSPLINES = 0,   ///< Bsplines basis
};



#endif  /*FDAGWR_TRAITS_HPP*/