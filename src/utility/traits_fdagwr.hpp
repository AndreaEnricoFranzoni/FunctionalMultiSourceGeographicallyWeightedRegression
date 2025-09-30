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


#include "include_fdagwr.hpp"


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
struct FDAGWR_TRAITS
{
public:
  
  using Dense_Matrix   = Eigen::MatrixXd;                                  ///< Matrix data structure.

  using Sparse_Matrix  = Eigen::SparseMatrix<double>;                     ///< Sparse matrix data structure.
  
  using Dense_Vector   = Eigen::VectorXd;                                  ///< Vector data structure.
  
  using Dense_Array    = Eigen::ArrayXd;                                   ///< Array data structure: more efficient for coefficient-wise operations.

  using Diag_Matrix    = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;    ///< Diagonal matrix (for weights matrices)

  using basis_geometry = fdapde::Triangulation<1, 1>;                      ///< Domain mesh: unit interval with a fixed number of nodes

  using fd_obj_y_type  = double;                                           ///< Functional data image type

  using fd_obj_x_type  = double;                                           ///< Functional data abscissa type

  using f_type = std::function<fd_obj_y_type(fd_obj_x_type const &)>;      ///< Function type
};


/*!
* @brief General features of the workflow
*/
struct FDAGWR_FEATS
{
  static constexpr std::size_t number_of_geographical_coordinates = static_cast<std::size_t>(2); 
 
  static constexpr std::string n_basis_string = "Basis number";

  static constexpr std::string degree_basis_string = "Basis degree";
};



/*!
* @enum Different possible types for fgwr
*/
enum FDAGWR_ALGO
{
  GWR_FMS_ESC = 0,  ///< Multi-source: stationary coefficients -> station-dependent coefficients -> event-dependent coefficients
  GWR_FMS_SEC = 1,  ///< Multi-source: stationary coefficients -> event-dependent coefficients -> station-dependent coefficients
  GWR_FOS = 2,      ///< One-source: stationary coefficients -> geographically-dependent coefficients 
  GWR_FST = 3,      ///< Stationary: stationary coefficients
};


template < FDAGWR_ALGO fdagwr_algo >
std::string
algo_type()
{
  if constexpr ( fdagwr_algo == FDAGWR_ALGO::GWR_FMS_ESC )    {   return "FGWR_FMS_ESC";}
  if constexpr ( fdagwr_algo == FDAGWR_ALGO::GWR_FMS_SEC )    {   return "FGWR_FMS_SEC";}
  if constexpr ( fdagwr_algo == FDAGWR_ALGO::GWR_FOS )        {   return "FGWR_FS";}
  if constexpr ( fdagwr_algo == FDAGWR_ALGO::GWR_FST )        {   return "FGWR";}
};



/*
using CoefficientsTuple = std::variant<
    std::tuple< FDAGWR_TRAITS::Dense_Matrix >, 
    std::tuple< FDAGWR_TRAITS::Dense_Matrix, std::vector< FDAGWR_TRAITS::Dense_Matrix > >, 
    std::tuple< FDAGWR_TRAITS::Dense_Matrix, std::vector< FDAGWR_TRAITS::Dense_Matrix >, std::vector< FDAGWR_TRAITS::Dense_Matrix > >
>;
*/



using BTuple = std::variant<
    std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix > >, 
    std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix >, std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > > >, 
    std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix >, std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > >, std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > > > 
>;


using BetasTuple = std::variant<
    std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > > >, 
    std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > >, std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > > > >, 
    std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > >, std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > > >, std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > > > > 
>;



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
  RESPONSE = 4,         ///< Response
  REC_WEIGHTS = 5,      ///< Response reconstruction weights
};


template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
std::string
covariate_type()
{
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::STATIONARY )    {   return "Stationary";}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::NON_STATIONARY ){   return "Nonstationary";}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::EVENT )         {   return "Event";}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::STATION )       {   return "Station";}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::RESPONSE )      {   return "Response";}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::REC_WEIGHTS )   {   return "Response reconstruction weights";}
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
* @enum PENALIZED_DERIVATIVE
* @brief order od the derivative to be penalized when creating penalization matrix
*/
enum PENALIZED_DERIVATIVE
{
  ZERO   = 0,   ///< Penalizing the basis itself
  FIRST  = 1,   ///< Penalizing the first derivative of the basis
  SECOND = 2,   ///< Penalizing the second derivative of the basis 
};



/*!
* @enum DISTANCE_MEASURE
* @brief measure to evaluate the distances within different location points for a GWR
*/
enum DISTANCE_MEASURE
{
  EUCLIDEAN = 0,  ///< Euclidean distance
};



/*!
* @enum REM_NAN: how to remove NaNs 
* @brief The available strategy for removing non-dummy NaNs
*/
enum REM_NAN
{ 
  NR = 0,      ///<  Not replacing NaN
  MR = 1,      ///< Replacing nans with mean (could change the mean of the distribution)
  ZR = 2,      ///< Replacing nans with 0s (could change the sd of the distribution)
};

#endif  /*FDAGWR_TRAITS_HPP*/