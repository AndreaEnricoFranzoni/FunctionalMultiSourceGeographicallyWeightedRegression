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


#ifndef FDAGWR_TRAITS_HPP
#define FDAGWR_TRAITS_HPP


#include "include_fdagwr.hpp"


/*!
* @file traits_fdagwr.hpp
* @brief Contains customized types and enumerator for customized template parameters, rather than constants strings for output lists constructions, used during the model's fitting
* @author Andrea Enrico Franzoni
*/




/*!
* @struct fdagwr_traits
* @brief Contains the customized types for matrices, functional elements, basis domain
*/
struct FDAGWR_TRAITS
{
public:
  
  using Dense_Matrix   = Eigen::MatrixXd;                                  ///< Matrix data structure.

  using Sparse_Matrix  = Eigen::SparseMatrix<double>;                      ///< Sparse matrix data structure.
  
  using Dense_Vector   = Eigen::VectorXd;                                  ///< Vector data structure.
  
  using Dense_Array    = Eigen::ArrayXd;                                   ///< Array data structure: more efficient for coefficient-wise operations.

  using Diag_Matrix    = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;    ///< Diagonal matrix (for weights matrices)

  using basis_geometry = fdapde::Triangulation<1, 1>;                      ///< Domain mesh: unit interval with a fixed number of nodes for basis construction

  using fd_obj_y_type  = double;                                           ///< Functional data image type

  using fd_obj_x_type  = double;                                           ///< Functional data abscissa type

  using f_type = std::function<fd_obj_y_type(fd_obj_x_type const &)>;      ///< Functional object type
};



/*!
* @struct FDAGWR_FEATS
* @brief Contains constant strings indicating default keys for the maps to wrap the inputs in the main functions
*/
struct FDAGWR_FEATS
{
  static constexpr std::size_t number_of_geographical_coordinates = static_cast<std::size_t>(2);    ///< Dimension of the space (geographical UTM coordinates) over which the non-stationary covariates vary 
 
  static constexpr std::string n_basis_string = "Basis number";                                     ///< Key for the map to store the input basis number

  static constexpr std::string degree_basis_string = "Basis degree";                                ///< Key for the map to store the input basis degree
};



/*!
* @enum Different possible types for Functional Weighted Regression model
*/
enum FDAGWR_ALGO
{
  _FMSGWR_ESC_ = 0,  ///< Multi-Source Geographically Weighted ESC: estimating stationary coefficients -> station-dependent coefficients -> event-dependent coefficients
  _FMSGWR_SEC_ = 1,  ///< Multi-Source Geographically Weighted SEC: estimating stationary coefficients -> event-dependent coefficients -> station-dependent coefficients
  _FMGWR_      = 2,  ///< Mixed Geographically Weighted: estimating stationary coefficients -> geographically-dependent coefficients 
  _FGWR_       = 3,  ///< Geographically Weighted: estimating non-stationary coefficients
  _FWR_        = 4,  ///< Weighted: estimating stationary coefficients
};



/*!
* @brief Function to return the Functional Regression model name
* @tparam fdagwr_algo the Functional Regression model
* @return a string containing the Functional Regression model name
*/
template < FDAGWR_ALGO fdagwr_algo >
constexpr
std::string
algo_type()
{
  if constexpr ( fdagwr_algo == FDAGWR_ALGO::_FMSGWR_ESC_ )    {   return "FMSGWR_ESC";}
  if constexpr ( fdagwr_algo == FDAGWR_ALGO::_FMSGWR_SEC_ )    {   return "FGWR_FMS_SEC";}
  if constexpr ( fdagwr_algo == FDAGWR_ALGO::_FMGWR_ )         {   return "FMGWR";}
  if constexpr ( fdagwr_algo == FDAGWR_ALGO::_FGWR_ )          {   return "FGWR";}
  if constexpr ( fdagwr_algo == FDAGWR_ALGO::_FWR_ )           {   return "FWR";}
};



/*!
* @brief Variant for the right tuple for storing the basis expansion coefficients for the regression coefficients of the FWR accordingly.
*        The first option is for a _FWR_: vector of matrices, one element for each C covariate.
*        The second option is for a _FGWR_: vector of vector of matrices. The outer vector contains one element for each NC covariate, the intern one for each units over which coefficients are eveluated.
*        The third option is for a _FMGWR_: first for C, second for NC covariates.
*        The latter option is for _FMSGWR_: first for C, second for E, third for S.
*/
using BTuple = std::variant<
    std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix > >, 
    std::tuple< std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > > >,
    std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix >, std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > > >, 
    std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix >, std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > >, std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > > > 
>;



/*!
* @struct FDAGWR_B_NAMES
* @brief Contains constant strings indicating basis expansion coefficients for regression coefficients names
*/
struct FDAGWR_B_NAMES
{
  static constexpr std::string bc  = "Bc";        //Constant covariates basis expansion coefficients for regression coefficients

  static constexpr std::string bnc = "Bnc";       //Non constant covariates basis expansion coefficients for regression coefficients

  static constexpr std::string be  = "Be";        //Event-dependent covariates basis expansion coefficients for regression coefficients

  static constexpr std::string bs  = "Bs";        //Station-dependent covariates basis expansion coefficients for regression coefficients
};



/*!
* @brief Variant for the right tuple for storing the discretized regression coefficients of the FWR accordingly, over a grid points of length N.
*        The first option is for a _FWR_: vector of vector of doubles: the outer vector indicates the C covariates, each element is the discretized value of the betaC (length N).
*        The second option is for a _FGWR_: vector of vector of vector of doubles: the outer vector indicates the NC covariates, and, for each statistical units, each element is the discretized value of the betaNC (length N).
*        The third option is for a _FMGWR_: first for C, second for NC covariates.
*        The latter option is for _FMSGWR_: first for C, second for E, third for S.
*/
using BetasTuple = std::variant<
    std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > > >, 
    std::tuple< std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > > > >,
    std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > >, std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > > > >, 
    std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > >, std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > > >, std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type > > > > 
>;



/*!
* @struct FDAGWR_BETAS_NAMES
* @brief Contains constant strings indicating regression coefficients names
*/
struct FDAGWR_BETAS_NAMES
{
  static constexpr std::string beta_c  = "Beta_c";      //Constant covariates regression coefficients

  static constexpr std::string beta_nc = "Beta_nc";     //Non constant covariates regression coefficients

  static constexpr std::string beta_e  = "Beta_e";      //Event-dependent covariates regression coefficients

  static constexpr std::string beta_s  = "Beta_s";      //Station-dependent covariates regression coefficients
};



/*!
* @brief Variant for the right tuple for storing the elements needed to reconstruct the FWR partial residuals, accordingly
*        The first option is for a _FWR_ and _FGWR_: nothing needed
*        The second option is for a _FMGWR_: a matrix used for reconstruct the first partial residuals
*        The latter option is for _FMSGWR_: first for reconstructing the first partial residuals, second and third, one matrix for each unit, for reconstructing the second partial residual
*/
using PartialResidualTuple = std::variant<
    std::monostate,
    std::tuple< FDAGWR_TRAITS::Dense_Matrix >,
    std::tuple< FDAGWR_TRAITS::Dense_Matrix, std::vector< FDAGWR_TRAITS::Dense_Matrix >, std::vector< FDAGWR_TRAITS::Dense_Matrix > >
>;



/*!
* @struct FDAGWR_HELPERS_for_PRED_NAMES
* @brief Contains constant strings with output list elements names
*/
struct FDAGWR_HELPERS_for_PRED_NAMES
{
  static constexpr std::string model_name = "FWR";                          //Model 

  static constexpr std::string estimation_iter = "EstimationTechnique";     //If brute force or exact estimation

  static constexpr std::string bf_estimate = "BruteForceEstimation";        //Brute force estimation

  static constexpr std::string elem_for_pred = "predictor_info";            //For predict function

  static constexpr std::string p_res = "partial_res";                       //Partial residuals

  static constexpr std::string inputs_info = "inputs_info";                 //Elements of the fitted model

  static constexpr std::string q = "number_covariates";                     //Number of covariates

  static constexpr std::string coeff_basis = "basis_coeff";                 //Basis coefficients

  static constexpr std::string n_basis = "basis_num";                       //Basis number

  static constexpr std::string basis_t = "basis_type";                      //Basis type

  static constexpr std::string basis_deg = "basis_deg";                     //Basis degree

  static constexpr std::string basis_knots = "knots";                       //Basis system knots

  static constexpr std::string penalties = "penalizations";                 //Lambdas for penalization

  static constexpr std::string coords = "coordinates";                      //UTM coordinates

  static constexpr std::string bdw_ker = "kernel_bwd_";                     //Kernel bandwith for computing weights

  static constexpr std::string cov = "cov_";                                //Stands for covariates

  static constexpr std::string beta = "beta_";                              //Stands for betas

  static constexpr std::string n = "n";                                     //Number of statistical units

  static constexpr std::string abscissa = "abscissa";                       //Abscissa points

  static constexpr std::string pred = "prediction";                         //Prediction

  static constexpr std::string a = "a";                                     //Left functional data extreme domain

  static constexpr std::string b = "b";                                     //Right functional data extreme domain

  static constexpr std::string p_res_c_tilde_hat = "c_tilde_hat";           //Element for reconstructing the first partial residuals (FMGWR and FMSGWR)

  static constexpr std::string p_res_A__ = "A__";                           //Elements for reconstructing the second partial residuals (FMSGWR)

  static constexpr std::string p_res_B__for_K = "B__for_K";                 //Elements for reconstructing the second partial residuals (FMSGWR)
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
  RESPONSE = 4,         ///< Response
  REC_WEIGHTS = 5,      ///< Response reconstruction weights
};



/*!
* @struct COVARIATES_NAMES
* @brief Contains constant strings with covariates names
*/
struct COVARIATES_NAMES
{
  static constexpr std::string Stationary                      = "Stationary";

  static constexpr std::string Nonstationary                   = "NonStationary";

  static constexpr std::string Event                           = "Event";

  static constexpr std::string Station                         = "Station";

  static constexpr std::string Response                        = "Response";
}; 



/*!
* @brief Function to return covariate type name
* @tparam FDAGWR_COVARIATES_TYPES type of covariates
* @return string with covariate type name
*/
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
constexpr
std::string
covariate_type()
{
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::STATIONARY )    {   return COVARIATES_NAMES::Stationary;}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::NON_STATIONARY ){   return COVARIATES_NAMES::Nonstationary;}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::EVENT )         {   return COVARIATES_NAMES::Event;}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::STATION )       {   return COVARIATES_NAMES::Station;}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::RESPONSE )      {   return COVARIATES_NAMES::Response;}
  if constexpr ( fdagwr_cov_t == FDAGWR_COVARIATES_TYPES::REC_WEIGHTS )   {   return "ResponseReconstructionWeights";}
};



/*!
* @enum KERNEL_FUNC
* @brief Kernel for evaluating the distances within different locations. 
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