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


#ifndef FDAGWR_WRAP_PARAMS_HPP
#define FDAGWR_WRAP_PARAMS_HPP

#include <RcppEigen.h>


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include "../basis/basis_include.hpp"
#include "data_reader.hpp"

#include <stdexcept>


/*!
* @file parameters_wrapper.hpp
* @brief Contains methods to check and wrap R-inputs into fdagwr-coherent ones.
* @author Andrea Enrico Franzoni
*/


/*!
* @brief Checking that an input has the correct dimension, throwing an exception if not
* @tparam fdagwr_cov_t enum indicating the type of the covariate
* @param sz_q dimension that the input must have
* @param sz_input input dimension
* @param quantity string indicating what the input is about in case it is necessary to throw an exception
*/
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
inline
void
check_dim_input(std::size_t sz_q,   
                std::size_t sz_input, 
                const std::string& quantity)
{
  // throwing an exception if dimensions not coinciding
  if(sz_q!=sz_input){
    std::string covariates_type = std::string{covariate_type<fdagwr_cov_t>()};
    std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
    std::string error_message = "For " + covariates_type + " covariates, " + quantity + " size has to be " + std::to_string(sz_q);
    throw std::invalid_argument(error_message);}
}



/*!
* @brief Wrapping the names of the covariates, intended as the name of the element of the R list passed as input. If absent, a default value is put
* @tparam fdagwr_cov_t enum indicating the type of the covariate
* @param cov_coeff_list list passed from R containing the basis expansion coefficients of the covariates indicated by the template parameter
* @return a vector of string containing the names of the covariates, for the type indicated by the template parameter
*/
//
//  [[Rcpp::depends(RcppEigen)]]
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
inline
std::vector<std::string>
wrap_covariates_names(Rcpp::List cov_coeff_list)
{
  // name of the covariates in the input list list
  Rcpp::Nullable<Rcpp::CharacterVector> cov_names_input_list = cov_coeff_list.names();
  // number of covariates 
  std::size_t number_cov = cov_coeff_list.size();
  // type of the covariates
  std::string covariates_type = std::string{covariate_type<fdagwr_cov_t>()};

  //if all the covariates do not own a name: put default names for all of them
  if (cov_names_input_list.isNull()){
      //output container
      std::vector<std::string> covariates_names;
      covariates_names.reserve(number_cov);
      //put a default value
      for(std::size_t i = 0; i < number_cov; ++i){  covariates_names.emplace_back("Covariate" + covariates_type + std::to_string(i+1));}

      return covariates_names;}
  
  //copy the actual names
  std::vector<std::string> covariates_names = as<std::vector<std::string>>(cov_names_input_list);
  //defaulting only missing names
  for(std::size_t i = 0; i < number_cov; ++i){  
        if(covariates_names[i] == "" ){         
            covariates_names[i] = "Covariate" + covariates_type + std::to_string(i+1);}}

  return covariates_names;
}



/*!
* @brief Wrapping the matrices containing the coefficientes of the basis expansion of a given type of covariates
* @tparam fdagwr_cov_t enum indicating the type of the covariate
* @param cov_coeff_list list passed from R containing the matrices of doubles with the basis expansions coefficients. Each row refers to a base, each column to a statistical unit
* @return a vector of Eigen matrices containing the coefficients of the basis expansions of the covariates of the given type (each row refers to a base, each column to a statistical unit)
* @note checking that all the matrices have the same number of columns (statistical units), throwing an exceptions if not
*/
//
//  [[Rcpp::depends(RcppEigen)]]
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
inline
std::vector<FDAGWR_TRAITS::Dense_Matrix>
wrap_covariates_coefficients(Rcpp::List cov_coeff_list)
{
  // number of covariates 
  std::size_t number_cov = cov_coeff_list.size();

  //where to store the covariates coefficients
  std::vector<FDAGWR_TRAITS::Dense_Matrix> covariates_coefficients;
  covariates_coefficients.reserve(number_cov);

  //check and read the coefficients for all the covariates
  for(std::size_t i = 0; i < number_cov; ++i){
    
    //read the data
    covariates_coefficients.push_back(reader_data<double,REM_NAN::MR>(cov_coeff_list[i]));

    //checking that all the coefficients refer to the same amount of statistical units (checking that all the list elements have the same number of columns)
    if(i>0   &&   covariates_coefficients[i-1].cols()!=covariates_coefficients[i].cols()){   
        std::string error_message = "All covariates coefficients have to refer to the same number of statistical units";
        throw std::invalid_argument(error_message);}
  }

  return covariates_coefficients;
}



/*!
* @brief Wrapping the string indicating the basis type for response and response reconstruction weights
* @tparam fdagwr_cov_t enum indicating the type of the covariate
* @param basis_types_name string passed from R
* @return string indicating the basis type
* @note checking that the passed input is in {"bsplines","constant"}, throwing an exception if not
*/
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
inline
std::string
wrap_and_check_basis_type(const std::string &basis_types_name)
{
  //checking if the requested basis type is implemented
  if (std::find(FDAGWR_BASIS_TYPES::_implemented_basis_.cbegin(),FDAGWR_BASIS_TYPES::_implemented_basis_.cend(),basis_types_name) == FDAGWR_BASIS_TYPES::_implemented_basis_.cend()){
      //throwing an exception if not
      std::string covariates_type = std::string{covariate_type<fdagwr_cov_t>()};
      std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
      std::string error_message = "For " + covariates_type + ", basis type " + basis_types_name + " is not acceptable: basis types accepted: ";
      for(auto it = FDAGWR_BASIS_TYPES::_implemented_basis_.cbegin(); it != FDAGWR_BASIS_TYPES::_implemented_basis_.cend(); ++it){
          error_message += *it;
          if (next(it)!=FDAGWR_BASIS_TYPES::_implemented_basis_.cend()){
            error_message += ", ";}}
      throw std::invalid_argument(error_message);}

  return basis_types_name;
}



/*!
* @brief Wrapping the strings indicating the basis types for the intended covariates type
* @tparam fdagwr_cov_t enum indicating the type of the covariate
* @param basis_types_names vector of strings passed from R indicating, in the element i-th, the basis type for covariate i-th
* @param number_of_covariates number of covariates of the intended type
* @return a vector of strings with, in the element i-th, the basis type for covariate i-th. By default, if a NULL vector is passed, return a vector of "bsplines"
* @note checking that the number of basis types passed is of the right dimension, and that all the elements of the passed input are in {"bsplines","constant"}, throwing an exception if not
*/
//
//  [[Rcpp::depends(RcppEigen)]]
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
inline
std::vector<std::string>
wrap_and_check_basis_type(Rcpp::Nullable<Rcpp::CharacterVector> basis_types_names, 
                          std::size_t number_of_covariates)
{
  //default if a NULL is passed 
  if (basis_types_names.isNull()){
    //returning all "bsplines"
    std::vector<std::string> basis_types_names_wrapped(number_of_covariates,std::string{FDAGWR_BASIS_TYPES::_bsplines_});
    return basis_types_names_wrapped;}

  //wrapper
  std::vector<std::string> basis_types_names_wrapped = Rcpp::as<std::vector<std::string>>(basis_types_names);
  // number of covariates in the passed input
  std::size_t q = basis_types_names_wrapped.size();
  //checking if correct dimension of the passed input
  check_dim_input<fdagwr_cov_t>(number_of_covariates,q,"vector with the basis types");

  //checking if the requested basis type is implemented, for all the covariates
  for(std::size_t i = 0; i < q; ++i){
    //throwing an exception if not
    if (std::find(FDAGWR_BASIS_TYPES::_implemented_basis_.cbegin(),FDAGWR_BASIS_TYPES::_implemented_basis_.cend(),basis_types_names_wrapped[i]) == FDAGWR_BASIS_TYPES::_implemented_basis_.cend()){
      std::string covariates_type = std::string{covariate_type<fdagwr_cov_t>()};
      std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
      std::string error_message = "For " + covariates_type + " covariates, basis type " + basis_types_names_wrapped[i] + " is not acceptable: basis types accepted: ";
      for(auto it = FDAGWR_BASIS_TYPES::_implemented_basis_.cbegin(); it != FDAGWR_BASIS_TYPES::_implemented_basis_.cend(); ++it){
          error_message += *it;
          if (next(it)!=FDAGWR_BASIS_TYPES::_implemented_basis_.cend()){
            error_message += ", ";}}
      throw std::invalid_argument(error_message);}
  }

  return basis_types_names_wrapped;
}



/*!
* @brief Columnize response basis expansion coefficients one below others, from a Lyxn matrix to a (n*Ly)x1 vector (n is the numer of statistical units, Ly the number if basis used for the basis expansion of the response)
* @param coeff_resp a Eigen matrix of double such that each row represents a specific base of the response (Ly), and each column a specific statistical units (n), containing basis expansion coefficients
* @return a Eigen matrix of double of dimension (n*Ly) x 1 containing the column of the input matrices one below the other
*/
inline
FDAGWR_TRAITS::Dense_Matrix
columnize_coeff_resp(const FDAGWR_TRAITS::Dense_Matrix& coeff_resp)
{
  //Eigen::Map presumes that data are stored col-major (ok)
  FDAGWR_TRAITS::Dense_Matrix c = Eigen::Map<const FDAGWR_TRAITS::Dense_Matrix>(coeff_resp.data(), coeff_resp.size(), 1);

  return c;
}


/*!
* @brief Wrapping the points over which the discrete evaluations of the functional object are available/knots for basis system.
* @param abscissas vector containing the domain points
* @param a left domain extreme
* @param b right domain extreme
* @return an vector containing the points.
* @note Checking consistency of domain extremes and passed points, eventualy throwing an error.
*/
inline
std::vector<double>
wrap_abscissas(Rcpp::NumericVector abscissas, 
               double a, 
               double b)    
{ 
  //check that domain extremes are consistent
  if(a>=b){
    //throwing an exception if not
    std::string error_message1 = "Left extreme of the domain has to be lower than the right one";
    throw std::invalid_argument(error_message1);}
  
  //sorting the abscissas values (security check)
  std::vector<double> abscissas_wrapped = Rcpp::as<std::vector<double>>(abscissas);
  std::sort(abscissas_wrapped.begin(),abscissas_wrapped.end());
  
  //checking that the passed points are inside the domain
  if(abscissas_wrapped[0] < a || abscissas_wrapped.back() > b){
    //throwing an exception if not
    std::string error_message2 = "The points in which there are the discrete evaluations of the curves have to be in the interval (" + std::to_string(a) + "," + std::to_string(b) + ")";
    throw std::invalid_argument(error_message2);}
    
  return abscissas_wrapped;
}



/*!
* @brief Wrapping the number of basis and the degree of the basis expansion for a functional datum, depending on the basis type
* @tparam fdagwr_cov_t enum indicating the type of the covariate
* @param basis_number integer indicating the number of basis of the functional datum basis expansion
* @param basis_degree integer indicating the degree of the functional datum basis expansion
* @param knots_size size of the knots used to perform the functional datum basis expansion
* @param basis_type string indicating the basis type
* @return a map containig two keys, "Basis number" and "Basis degree", associated respectively with the number of basis and their degree
* @note if the basis type is "constant", "Basis number" is 0 while "Basis degree" is 1, by default. For "bsplines", default degree is 3, 
*       the default basis number is computed as basis_number = basis_degree + knots_size - 1. An exception is thrown if input parameters that
*       do not satisfy this relationship are passed, or negative basis_degree or non-positive basis_number are passed, or if
*       basis_number < knots_size - 1 (would mean a negative degree)
*/
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
inline
std::map<std::string,std::size_t>
wrap_and_check_basis_number_and_degree(Rcpp::Nullable<int> basis_number, 
                                       Rcpp::Nullable<int> basis_degree, 
                                       std::size_t knots_size,
                                       std::string basis_type)
{
  //container to store the returning element
  std::map<std::string,std::size_t> returning_element;


  //"constant" basis type case
  if(basis_type == std::string{FDAGWR_BASIS_TYPES::_constant_})
  {
    //default (and only possible) case
    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::n_basis_string},constant_basis<FDAGWR_TRAITS::basis_geometry>::number_of_basis_constant_basis));    //one base
    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::degree_basis_string},constant_basis<FDAGWR_TRAITS::basis_geometry>::degree_constant_basis));        //degree zero

    return returning_element;
  }


  //"bsplines" basis type case

  //basis number unknown, order unknown: default values
  if(basis_number.isNull() && basis_degree.isNull())     
  {
    std::size_t degree = bsplines_basis<FDAGWR_TRAITS::basis_geometry>::bsplines_degree_default;       //default is a cubic B-spline (degree 3)
    std::size_t n_basis = degree + knots_size - static_cast<std::size_t>(1);                           //B-splines constraint for basis number

    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::n_basis_string},n_basis));
    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::degree_basis_string},degree));
  }

  //basis number unknown, degree known
  if (basis_number.isNull() && basis_degree.isNotNull())
  {
    //throwing an exception if a negative degree is passed
    if (Rcpp::as<int>(basis_degree) < 0){
      std::string error_message1 = "Basis degree for the response has to be a non-negative integer";
      throw std::invalid_argument(error_message1);}

    //computing the basis number
    std::size_t n_basis = static_cast<std::size_t>(Rcpp::as<int>(basis_degree)) + knots_size - static_cast<std::size_t>(1);

    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::n_basis_string},n_basis));
    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::degree_basis_string},static_cast<std::size_t>(Rcpp::as<int>(basis_degree))));
  }

  //basis number known, degree unknown
  if (basis_number.isNotNull() && basis_degree.isNull())
  {
    //throwing an exception if basis_number < knots_size - 1, that would inquire a negative degree
    if (static_cast<std::size_t>(Rcpp::as<int>(basis_number)) < knots_size - static_cast<std::size_t>(1)){
      std::string error_message2 = "The number of basis for the response has to be at least the number of knots (" + std::to_string(knots_size) + ") - 1";
      throw std::invalid_argument(error_message2);}

    //computing the degree
    std::size_t degree = static_cast<std::size_t>(Rcpp::as<int>(basis_number)) - knots_size + static_cast<std::size_t>(1);

    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::n_basis_string},static_cast<std::size_t>(Rcpp::as<int>(basis_number))));
    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::degree_basis_string},degree));
  }

  //both basis number and order known
  if (basis_number.isNotNull() && basis_degree.isNotNull())
  {
    //thrwoing an exception if a negative basis degree is passed
    if (Rcpp::as<int>(basis_degree) < 0){
      std::string error_message3 = "Basis degree for the response has to be a non-negative integer";
      throw std::invalid_argument(error_message3);}
    //thrwoing an exception if a non-positive basis number is passed
    if (Rcpp::as<int>(basis_number) <= 0){
      std::string error_message4 = "Basis number for the response has to be a positive integer";
      throw std::invalid_argument(error_message4);}
    //throwing an exception if basis_number != basis_degree + knots_size - 1
    if (static_cast<std::size_t>(Rcpp::as<int>(basis_number)) != static_cast<std::size_t>(Rcpp::as<int>(basis_degree)) + knots_size - static_cast<std::size_t>(1)){
      std::string error_message5 = "The number of basis for the response has to be the order of the basis (" + std::to_string(static_cast<std::size_t>(Rcpp::as<int>(basis_degree))) + ") + the number of knots (" + std::to_string(knots_size) + ") - 1";
      throw std::invalid_argument(error_message5);}

    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::n_basis_string},static_cast<std::size_t>(static_cast<std::size_t>(Rcpp::as<int>(basis_number)))));
    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::degree_basis_string},static_cast<std::size_t>(static_cast<std::size_t>(Rcpp::as<int>(basis_degree)))));
  }
  
  return returning_element;
}



/*!
* @brief Wrapping the number of basis and the degree of the basis expansion for a functional datum, depending on the basis type, for each one of the covariate of the intended type
* @tparam fdagwr_cov_t enum indicating the type of the covariate
* @param basis_numbers vector of integers, element i-th indicating the number of basis of the functional datum basis expansion for the i-th covariate of the intended type
* @param basis_degrees vector of integers, element i-th indicating the degree of the functional datum basis expansion for the i-th covariate of the intended type
* @param knots_size size of the knots used to perform the functional datum basis expansion
* @param number_of_covariates number of covariates of the intended type
* @param basis_types vector of strings, element i-th indicating the basis type of the i-th covariate of the intended type
* @return a map containig two keys, "Basis number" and "Basis degree", associated respectively with a vector containing basis number and basis degree, element i-th referred to the i-th covariate of the intended type
* @note if the basis type is "constant", "Basis number" is 0 while "Basis degree" is 1, by default. For "bsplines", default degree is 3, 
*       the default basis number is computed as basis_number = basis_degree + knots_size - 1. An exception is thrown if input parameters that
*       do not satisfy this relationship are passed, or negative basis_degree or non-positive basis_number are passed, or if
*       basis_number < knots_size - 1 (would mean a negative degree). An exception is also thrown if the dimension of the input differs from the covariates number. This applies for each covariate
*/
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
inline
std::map<std::string,std::vector<std::size_t>>
wrap_and_check_basis_number_and_degree(Rcpp::Nullable<Rcpp::IntegerVector> basis_numbers, 
                                       Rcpp::Nullable<Rcpp::IntegerVector> basis_degrees, 
                                       std::size_t knots_size, 
                                       std::size_t number_of_covariates,
                                       const std::vector<std::string> & basis_types)
{
  //container to store the returning element
  std::map<std::string,std::vector<std::size_t>> returning_element;

  //basis number unknown, order unknown: default values
  if (basis_numbers.isNull() && basis_degrees.isNull())
  {
    std::vector<std::size_t> degrees(number_of_covariates,bsplines_basis<FDAGWR_TRAITS::basis_geometry>::bsplines_degree_default);    //default is a cubic B-spline (degree 3) for all the covariates
    std::vector<std::size_t> ns_basis(number_of_covariates,bsplines_basis<FDAGWR_TRAITS::basis_geometry>::bsplines_degree_default + knots_size - static_cast<std::size_t>(1));

    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::n_basis_string},ns_basis));
    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::degree_basis_string},degrees));
  }

  //basis number unknown, order known
  if (basis_numbers.isNull() && basis_degrees.isNotNull())
  {
    //wraping the R input
    auto basis_degrees_w = Rcpp::as<std::vector<int>>(basis_degrees);

    //check the correct dimension of the input (number of covariates, indeed), if not throwing an exception
    if (basis_degrees_w.size() != number_of_covariates){
      std::string covariates_type = covariate_type<fdagwr_cov_t>();
      std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
      std::string error_message1 = "It is necessary to pass a vector with " + std::to_string(number_of_covariates) + " basis degrees for the " + covariates_type + " covariates";
      throw std::invalid_argument(error_message1);}

    //checking that the input is consistent (basis degrees non-negative for all the covariates), if not throwing an exception
    auto min_basis_degree = std::min_element(basis_degrees_w.cbegin(),basis_degrees_w.cend());
    if (*min_basis_degree < 0){
      std::string covariates_type = covariate_type<fdagwr_cov_t>();
      std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
      std::string error_message2 = "Basis degrees for all the " + covariates_type + " covariates have to be non-negative";
      throw std::invalid_argument(error_message2);}

    //converting into a vector of std::size_t (create another copy for the conversion, they are objects of small dimensions)
    std::vector<std::size_t> degrees;
    degrees.resize(number_of_covariates);
    std::transform(basis_degrees_w.cbegin(),
                   basis_degrees_w.cend(),
                   degrees.begin(),
                   [](auto el){return static_cast<std::size_t>(el);});

    //computing the number of basis for each covariate accordingly
    std::vector<std::size_t> ns_basis;
    ns_basis.resize(number_of_covariates);
    std::transform(degrees.cbegin(),
                   degrees.cend(),
                   ns_basis.begin(),
                   [knots_size](std::size_t const &el){ return (el + knots_size - static_cast<std::size_t>(1));});

    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::n_basis_string},ns_basis));
    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::degree_basis_string},degrees));
  }

  //basis number known, order unknown
  if (basis_numbers.isNotNull() && basis_degrees.isNull())
  {
    //wrapping the R input
    auto basis_numbers_w = Rcpp::as<std::vector<int>>(basis_numbers);

    //check the correct dimension of the input (number of covariates, indeed), if not throwing an exception
    if (basis_numbers_w.size() != number_of_covariates){
      std::string covariates_type = covariate_type<fdagwr_cov_t>();
      std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
      std::string error_message3 = "It is necessary to pass a vector with " + std::to_string(number_of_covariates) + " number of basis for the " + covariates_type + " covariates";
      throw std::invalid_argument(error_message3);}

    //checking that the input is consistent (basis number at least the number of knots - 1 for all the covariates for bsplines), if not throwing an exception
    for(std::size_t i = 0; i < number_of_covariates; ++i){
          if ((basis_numbers_w[i] < (knots_size - static_cast<std::size_t>(1))) && basis_types[i] == FDAGWR_BASIS_TYPES::_bsplines_){ //checking B-splines relationship for each covariate, where basis are bsplines
              std::string covariates_type = covariate_type<fdagwr_cov_t>();
              std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
              std::string error_message4 = "The number of basis, bspline basis, for all the " + covariates_type + " covariates has to be at least the number of knots (" + std::to_string(knots_size) + ") - 1";
              throw std::invalid_argument(error_message4);}}

    //converting into a vector of std::size_t (create another copy for the conversion, they are objects of small dimensions)
    std::vector<std::size_t> ns_basis;
    ns_basis.resize(number_of_covariates);
    std::transform(basis_numbers_w.cbegin(),
                   basis_numbers_w.cend(),
                   ns_basis.begin(),
                   [](auto el){return static_cast<std::size_t>(el);});

    //computing the order of the basis for each covariate
    std::vector<std::size_t> degrees;
    degrees.resize(number_of_covariates);
    std::transform(ns_basis.cbegin(),
                   ns_basis.cend(),
                   degrees.begin(),
                   [knots_size](std::size_t const &el){return (el - knots_size + static_cast<std::size_t>(1));});

    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::n_basis_string},ns_basis));
    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::degree_basis_string},degrees));
  }
  
  //both basis number and order known
  if (basis_numbers.isNotNull() && basis_degrees.isNotNull())
  {
    //wrapping the R input
    auto basis_degrees_w = Rcpp::as<std::vector<int>>(basis_degrees);
    auto basis_numbers_w = Rcpp::as<std::vector<int>>(basis_numbers);

    //check the correct dimension of the inputs (number of covariates, indeed), if not throwing an exception
    if (basis_degrees_w.size() != number_of_covariates){
      std::string covariates_type = covariate_type<fdagwr_cov_t>();
      std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
      std::string error_message5 = "It is necessary to pass a vector with " + std::to_string(number_of_covariates) + " basis degrees for the " + covariates_type + " covariates";
      throw std::invalid_argument(error_message5);}
    if (basis_numbers_w.size() != number_of_covariates){
      std::string covariates_type = covariate_type<fdagwr_cov_t>();
      std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
      std::string error_message6 = "It is necessary to pass a vector with " + std::to_string(number_of_covariates) + " number of basis for the " + covariates_type + " covariates";
      throw std::invalid_argument(error_message6);}

    //checking that the degrees of basis input are consistent (basis order non-negative for all the covariates), if not throwing an exception
    auto min_basis_degree = std::min_element(basis_degrees_w.cbegin(),basis_degrees_w.cend());
    if (*min_basis_degree < 0){
        std::string covariates_type = covariate_type<fdagwr_cov_t>();
        std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
        std::string error_message7 = "Basis degrees for all the " + covariates_type + " covariates have to be non-negative";
        throw std::invalid_argument(error_message7);}

    //converting into a vector of std::size_t (create another copy for the conversion, they are objects of small dimensions)
    //numbers of basis
    std::vector<std::size_t> ns_basis;
    ns_basis.resize(number_of_covariates);
    std::transform(basis_numbers_w.cbegin(),
                   basis_numbers_w.cend(),
                   ns_basis.begin(),
                   [](auto el){return static_cast<std::size_t>(el);});
    //degrees of basis
    std::vector<std::size_t> degrees;
    degrees.resize(number_of_covariates);
    std::transform(basis_degrees_w.cbegin(),
                   basis_degrees_w.cend(),
                   degrees.begin(),
                   [](auto el){return static_cast<std::size_t>(el);});

    //check that the number of basis input is consistent, if not throwing an exception
    for(std::size_t i = 0; i < number_of_covariates; ++i){
          //checking B-splines relationship for each covariate, where basis are bsplines
          if ((ns_basis[i] - (degrees[i] + knots_size - static_cast<std::size_t>(1)) != 0) && basis_types[i] == FDAGWR_BASIS_TYPES::_bsplines_){  
            std::string covariates_type = covariate_type<fdagwr_cov_t>();
            std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
            std::string error_message8 = "The number of basis, bspline basis, for the " + covariates_type + " covariates has to be the degree of the basis + the number of knots - 1";
            throw std::invalid_argument(error_message8);}}

    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::n_basis_string},ns_basis));
    returning_element.insert(std::make_pair(std::string{FDAGWR_FEATS::degree_basis_string},degrees));
  }


  //looking for any constant basis, putting coherent values for those covariates
  for(std::size_t i = 0; i < basis_types.size(); ++i)
  {
    if (basis_types[i] == FDAGWR_BASIS_TYPES::_constant_)
    {
      returning_element[std::string{FDAGWR_FEATS::n_basis_string}][i] = constant_basis<FDAGWR_TRAITS::basis_geometry>::number_of_basis_constant_basis;
      returning_element[std::string{FDAGWR_FEATS::degree_basis_string}][i] = constant_basis<FDAGWR_TRAITS::basis_geometry>::degree_constant_basis;
    }
  }

  return returning_element;
}



/*!
* @brief Wrapping the penalizations passed, for the intended type of covariates
* @tparam fdagwr_cov_t enum indicating the type of the covariate
* @param lambdas vector of double containing, in the element i-th, the penalization for the i-th covariate of the intended type
* @param number_of_covariates number of covariates of the intended type
* @return a vector with the penalization, in the element i-th, the penalization for the i-th covariate of the intended type
* @note checking if the size of the passed vector coincides with the number of covariates and if all the elements of the passed input are non-negative, throwing an exception if not
*/
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
inline
std::vector<double>
wrap_and_check_penalizations(Rcpp::NumericVector lambdas,
                             std::size_t number_of_covariates)
{
  //checking input dimension
  check_dim_input<fdagwr_cov_t>(number_of_covariates,lambdas.size(),"penalizations vector");

  //wrapping the input
  std::vector<double> lambdas_wrapped = Rcpp::as<std::vector<double>>(lambdas);

  //checking that all the elements are positive
  auto min_lambda = std::min_element(lambdas_wrapped.begin(), lambdas_wrapped.end());
  //throwing an exception if not
  if (*min_lambda < 0){
    // type of the covariates for which the penalization is used
    std::string covariates_type = covariate_type<fdagwr_cov_t>();
    std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
    std::string error_message = "Penalization terms for " + covariates_type + " covariates have to be non-negative";
    throw std::invalid_argument(error_message);}

  return lambdas_wrapped;  
}



/*!
* @brief Wrapping the kernel bandwith
* @tparam fdagwr_cov_t enum indicating the type of the covariate
* @param bandwith kernel bandwith
* @return return the kernel bandwith
* @note checking that the bandwith is a positive number, throwing an exception if not
*/
template < FDAGWR_COVARIATES_TYPES fdagwr_cov_t >
inline
double
wrap_and_check_kernel_bandwith(double bandwith)
{
  //checking that the bandwith is positive, throwing an exception if not
  if (bandwith <= 0){
    // type of the covariates for which the kernel bandwith is used
    std::string covariates_type = covariate_type<fdagwr_cov_t>();
    std::transform(covariates_type.begin(),covariates_type.end(),covariates_type.begin(),[](unsigned char c) { return std::tolower(c);});
    std::string error_message = "Kernel bandwith for " + covariates_type + " covariates has to be positive";
    throw std::invalid_argument(error_message);}

  return bandwith;  
}




/*!
* @brief Wrapping the number of knots used for smoothing the response without the non-stationary components
* @param n_knots number of knots used for smoothing the response without the non-stationary components
* @return return the number of knots used for smoothing the response without the non-stationary components
* @note checking that the number of knots used for smoothing the response without the non-stationary components is a positive number, throwing an exception if not
*/
inline
int
wrap_and_check_n_knots_smoothing(int n_knots)
{
  //checking that the number of intervals is positive, throwing an exception if not
  if (n_knots <= 0){
    std::string error_message = "The number of knots used for smoothing the response without the non-stationary components has to be positive";
    throw std::invalid_argument(error_message);}

  return n_knots;  
}




/*!
* @brief Wrapping the number of intervals for the integration via midpoint quadrature rule
* @param n_intervals number of intervals used for integrating via midpoint quadrature rule
* @return return the number of intervals used for integrating via midpoint quadrature rule
* @note checking that the number of intervals used for integrating via midpoint quadrature rule is a positive number, throwing an exception if not
*/
inline
int
wrap_and_check_n_intervals_quadrature(int n_intervals)
{
  //checking that the number of intervals is positive, throwing an exception if not
  if (n_intervals <= 0){
    std::string error_message = "The number of intervals used for integrating via midpoint quadrature rule has to be positive";
    throw std::invalid_argument(error_message);}

  return n_intervals;  
}





/*!
* @brief Wrapping the number of threads for OMP
* @param num_threads indicates how many threads to be used by multi-threading directives.
* @return the number of threads
* @details if omp is not included: will return 1. If not, a number going from 1 up to the maximum cores available by the machine used (default, or if the input is smaller than 1 or bigger than the maximum number of available cores)
* @note omp requested
*/
inline
int
wrap_num_thread(Rcpp::Nullable<int> num_threads)
{
#ifndef _OPENMP

  return 1;
#else

  //getting maximum number of cores in the machine
  int max_n_t = omp_get_num_procs();
  
  if(num_threads.isNull())
  {
    return max_n_t;
  }
  else
  {
    int n_t = Rcpp::as<int>(num_threads);
    if(n_t < 1 || n_t > max_n_t){  return max_n_t;}
    
    return n_t;
  }
#endif
}




////////////////////////////////////////////////////////////////////////////////
///// FUNCTIONS to check input consistency for the predict function input  /////
////////////////////////////////////////////////////////////////////////////////
template< FDAGWR_ALGO fdagwr_algo >
void
wrap_predict_input(Rcpp::List pred_input)
{
  //FMSGWR_ESC
  if constexpr( fdagwr_algo == FDAGWR_ALGO::_FMSGWR_ESC_ )
  {
    //check input list
    if (pred_input.size() != 9){ throw std::invalid_argument("Lenght of input list model_fitted has to be 8");}

    //check that derives from the right algorithm
    if( as<std::string>(pred_input[std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name}]) != std::string{algo_type<fdagwr_algo>()}){ throw std::invalid_argument("It is not a fitted FMSGWR_ESC");}
  }

  //FGWR_FMS_SEC
  if constexpr( fdagwr_algo == FDAGWR_ALGO::_FMSGWR_SEC_ )
  {
    //check input list
    if (pred_input.size() != 9){ throw std::invalid_argument("Lenght of input list model_fitted has to be 8");}

    //check that derives from the right algorithm
    if( as<std::string>(pred_input[std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name}]) != std::string{algo_type<fdagwr_algo>()}){ throw std::invalid_argument("It is not a fitted FMSGWR_SEC");}
  }

  //FMGWR
  if constexpr( fdagwr_algo == FDAGWR_ALGO::_FMGWR_  )
  {
    //check input list
    if (pred_input.size() != 7){ throw std::invalid_argument("Lenght of input list model_fitted has to be 6");}

    //check that derives from the right algorithm
    if( as<std::string>(pred_input[std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name}]) != std::string{algo_type<fdagwr_algo>()}){ throw std::invalid_argument("It is not a fitted FMGWR");}
  }

  //FGWR
  if constexpr( fdagwr_algo == FDAGWR_ALGO::_FGWR_  )
  {
    //check input list
    if (pred_input.size() != 4){ throw std::invalid_argument("Lenght of input list model_fitted has to be 4");}

    //check that derives from the right algorithm
    if( as<std::string>(pred_input[std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name}]) != std::string{algo_type<fdagwr_algo>()}){ throw std::invalid_argument("It is not a fitted FGWR");}
  }

  //FWR 
  if constexpr( fdagwr_algo == FDAGWR_ALGO::_FWR_ )
  {
    //check input list
    if (pred_input.size() != 4){ throw std::invalid_argument("Lenght of input list model_fitted has to be 4");}

    //check that derives from the right algorithm
    if( as<std::string>(pred_input[std::string{FDAGWR_HELPERS_for_PRED_NAMES::model_name}]) != std::string{algo_type<fdagwr_algo>()}){ throw std::invalid_argument("It is not a fitted FWR");}
  }
}

#endif  /*FDAGWR_WRAP_PARAMS_HPP*/