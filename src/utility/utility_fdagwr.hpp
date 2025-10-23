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


#ifndef FDAGWR_UTILITIES_HPP
#define FDAGWR_UTILITIES_HPP


#include "traits_fdagwr.hpp"
#include <RcppEigen.h>




/*!
* @file utility_fdagwr.hpp
* @brief Contains utilities for fdagwr, especially for wrapping the output
* @author Andrea Enrico Franzoni
*/


/*!
* @struct extract_template
* @tparam T template template parameter
* @brief utility to extract a template template parameter as a template parameter
*/
// Primaria: vuota
template <typename T>
struct extract_template;

/*!
* @struct extract_template
* @tparam TT template template parameter
* @tparam Args variadic template parameters
* @brief Specialization for template with only parameters
*/
template <template <typename...> class TT, typename... Args>
struct extract_template<TT<Args...>> {
    // Alias to reproduce the starting template param
    template <typename... Ts>
    using template_type = TT<Ts...>;
};

// Helper per normalizzare il tipo (toglie const/ref)
/*!
* @tparam T template template parameter
* @brief Specialization for template with only parameters, removing const ref
* @example template <typename Domain>
           struct basis {};

           template <template <typename> class BasisTemplate>
           struct wrapper {};

           int main() 
           {
                std::unique_ptr<basis<int>> ptr;

                using PointeeType   = typename decltype(ptr)::element_type; // basis<int>
                using Extracted     = extract_template_t<PointeeType>;
                using BasisTemplate = Extracted::template_type;             // alias template for a template template param

                wrapper<BasisTemplate> w;                                   // using the alias as a template template param
            }
*/
template <typename T>
using extract_template_t = extract_template<
    std::remove_cv_t<std::remove_reference_t<T>>
>;



/*!
* @brief Function to return the name of the estimation techinque (brute force or exact)
* @param bf_estimation bool: if true, brute force estimation is used. If not, exact
* @return a string with the estimation technique name
*/
std::string
estimation_iter(bool bf_estimation)
{
    return bf_estimation ? "BruteForceEstimation" : "ExactEstimation";
}



/*!
* @brief Converts an std::vector of Eigen::MatrixXd to an R list with matrices of double, representing the stationary b (basis expansion coefficient of the betas)
* @param b std::vector (one element for each stationary covariate) of Eigen::MatrixXd with all the bC (for covariate j-th: Lc_jx1)
* @param add_unit_number if true, every elements of the list is named "unit_" + the number of the element. If false, not
* @return a list with the matrices representing the bC (for covariate j-th: Lc_jx1)
*/
Rcpp::List 
toRList(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
        bool add_unit_number = false) 
{
    //WRAP B STATIONARY
    Rcpp::List out(b.size());
    Rcpp::CharacterVector names(b.size());
    for (std::size_t i = 0; i < b.size(); i++) {
        out[i] = Rcpp::wrap(b[i]);  // RcppEigen converts Eigen::MatrixXd
        names[i] = std::string("unit_") + std::to_string(i+1);
    }

    if (add_unit_number){   out.names() = names;}

    return out;
}



/*!
* @brief Helper for adding extra infos to the Rcpp::List that wraps stationary b (basis expansion coefficient of the betas)
* @param b Eigen::MatrixXd with the bC (for covariate j-th: Lc_jx1)
* @param basis_type type of basis of the betas
* @param basis_number number of basis for the betas
* @param basis_knots vector containing the knots for the betas basis expansion
* @return a list with the matrices representing the bC (Lc_jx1) and additional information
*/
Rcpp::List 
enrichB(const FDAGWR_TRAITS::Dense_Matrix& b,
        const std::string& basis_type,
        std::size_t basis_number,
        const std::vector<FDAGWR_TRAITS::fd_obj_x_type>& basis_knots) 
{
    //WRAP B STATIONARY
    return Rcpp::List::create(Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis) = Rcpp::wrap(b),
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::basis_t)     = basis_type,
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::n_basis)     = basis_number,
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::basis_knots) = Rcpp::NumericVector(basis_knots.cbegin(), basis_knots.cend()));
}



/*!
* @brief Converts all the stationary b (basis expansion coefficient of the betas) into an Rcpp::List with additional information
* @param b std::vector (one element for each stationary covariate) of Eigen::MatrixXd with all the bC (for covariate j-th: Lc_jx1)
* @param basis_type std::vector of string with the basis names for the betaC
* @param basis_number std::vector of integer with the basis numbers for the betaC
* @param basis_knots std::vector with std::vector of doubles with the knots for the betaC
* @return an Rcpp::List containing the information of all betaC basis expansions (the coefficients, for covariate j-th: Lc_jx1)
*/
Rcpp::List 
toRList(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
        const std::vector<std::string>& basis_type,
        const std::vector<std::size_t>& basis_number,
        const std::vector<FDAGWR_TRAITS::fd_obj_x_type>& basis_knots) 
{
    //WRAP B STATIONARY
    Rcpp::List out(b.size());
    for (size_t j = 0; j < b.size(); ++j) {
        out[j] = enrichB(b[j],
                         basis_type[j],
                         basis_number[j],
                         basis_knots);
    }
    return out;
}



/*!
* @brief Converts all the non-stationary b into an R list of matrices
* @param b vector (one element for each covariate) of vector (one element for each statistical unit) of Eigen::MatrixXd that represent non-stationary covariates regression coefficients basis expansion coefficients (for covariate j-th: L_jx1)
* @return an Rcpp::List (one for each non-stationary covariates) containing the Rcpp::List of the basis expansion coefficients for each unit
*/
Rcpp::List
toRList(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& b)
{
    // WRAP B NON-STATIONARY
    //for each non-stationary covariate
    Rcpp::List outerList(b.size());

    for(std::size_t i = 0; i < b.size(); ++i){
        Rcpp::List innerList(b[i].size());
        Rcpp::CharacterVector innerNames(b[i].size());
        //for each unit
        for(std::size_t j = 0; j < b[i].size(); ++j){
            innerList[j] = Rcpp::wrap(b[i][j]);
            innerNames[j] = std::string("unit_") + std::to_string(j+1);
        }

        innerList.names() = innerNames;
        outerList[i]=(innerList);
    }
    return outerList;
}



/*!
* @brief Converts a the non-stationary b into an Rcpp::List with additional information of the basis expansion
* @param b vector (one element for each statistical unit) of Eigen::MatrixXd that represent non-stationary covariates regression coefficients basis expansion coefficients (for covariate j-th: L_jx1)
* @param basis_type string with the basis type
* @param basis_number integer with the basis number
* @param basis_knots vector of double containing the knots over which the basis expansion is performed
* @return an Rcpp::List containing the basis expansion coefficients for each unit, the basis type, the basis number and the knots over which the basis expansion is performed
*/
Rcpp::List 
enrichB(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
        const std::string& basis_type,
        std::size_t basis_number,
        const std::vector<FDAGWR_TRAITS::fd_obj_x_type>& basis_knots) 
{
    // WRAP B NON-STATIONARY
    return Rcpp::List::create(Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis) = toRList(b,true),
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::basis_t)     = basis_type,
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::n_basis)     = basis_number,
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::basis_knots) = Rcpp::NumericVector(basis_knots.cbegin(), basis_knots.cend()));
}



/*!
* @brief Converts all the non-stationary b (basis expansion coefficient of the betas) into an Rcpp::List with additional information
* @param b std::vector (one element for each stationary covariate) of std::vector (one element for each unit) of Eigen::MatrixXd with all the bNC (for covariate j-th: Lnc_jx1)
* @param basis_type std::vector of string with the basis names for the betaNC
* @param basis_number std::vector of integer with the basis numbers for the betaNC
* @param basis_knots std::vector with std::vector of doubles with the knots for the betaNC
* @return an Rcpp::List containing the information of all betaNC basis expansions (the coefficients, for covariate j-th: Lnc_jx1, for each unit)
*/
Rcpp::List 
toRList(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& b,
        const std::vector<std::string>& basis_type,
        const std::vector<std::size_t>& basis_number,
        const std::vector<FDAGWR_TRAITS::fd_obj_x_type>& basis_knots) 
{
    // WRAP B NON-STATIONARY
    Rcpp::List out(b.size());
    for (size_t j = 0; j < b.size(); ++j) {
        out[j] = enrichB(b[j],
                         basis_type[j],
                         basis_number[j],
                         basis_knots);
    }
    return out;
}



/*!
* @brief Wrap the discretized evaluated stationary betas into an Rcpp::List
* @param betas a vector (one for each stationary covariate) of vector (evaluations of beta for each given abscissa) with the betas evaluation
* @return an Rcpp::List with the evaluations of each stationary beta
*/
Rcpp::List 
toRList(const std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>& betas) 
{
    // WRAP BETAS STATIONARY 
    Rcpp::List out(betas.size());
    for (size_t i = 0; i < betas.size(); ++i) {
        out[i] = Rcpp::NumericVector(betas[i].cbegin(), betas[i].cend());
    }
    return out;
}



/*!
* @brief Wrap the discretized evaluated stationary betas into an Rcpp::List, adding the abscissas
* @param betas vector (one element for each stationary covariate) with the vector with betas evaluation (dimension equal to the number of abscissa over which the evaluation is performed)
* @param abscissas vector with the abscissa over which the beta is evalauted
*/
Rcpp::List 
toRList(const std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>& betas,
        const std::vector< FDAGWR_TRAITS::fd_obj_x_type>& abscissas) 
{
    // WRAP BETAS STATIONARY 
    Rcpp::List out(betas.size());
    for (size_t j = 0; j < betas.size(); ++j) {
        Rcpp::List elem = Rcpp::List::create(Rcpp::Named("Beta_eval") = Rcpp::NumericVector(betas[j].cbegin(), betas[j].cend()),
                                             Rcpp::Named("Abscissa")  = Rcpp::NumericVector(abscissas.cbegin(), abscissas.cend()));

        out[j] = elem;}

    return out;
}



/*!
* @brief Wrap the discretized evaluated non-stationary betas into an Rcpp::List
* @param betas vector (one for each non-stationary covariate) of vector (one element for each unit) of the non-stationary betas
* @return an Rcpp::List, one element for each non-statinary covariate, of Rcpp::List, one for each unit, of the betas evalautions
*/
Rcpp::List
toRList(const std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>>& betas)
{
    // WRAP BETAS NON-STATIONARY
    Rcpp::List outerList(betas.size());

    for(std::size_t i = 0; i < betas.size(); ++i){
        Rcpp::List innerList(betas[i].size());
        Rcpp::CharacterVector innerNames(betas[i].size());

        for(std::size_t j = 0; j < betas[i].size(); ++j){
            innerList[j] = Rcpp::NumericVector(betas[i][j].cbegin(), betas[i][j].cend());
            innerNames[j] = std::string("unit_") + std::to_string(j+1);
        }

        innerList.names() = innerNames;
        outerList[i]=(innerList);
    }

    return outerList;
}



/*!
* @brief Wrap the discretized evaluated non-stationary betas into an Rcpp::List
* @param betas vector (one for each non-stationary covariate) of vector (one element for each unit) of the non-stationary betas (dimension equal to the number of abscissas)
* @param abscissas vector with the abscissa over which the betas are evaluated
* @return an Rcpp::List, one element for each non-statinary covariate, of Rcpp::List, one for each unit, of the betas evalautions
*/
Rcpp::List 
toRList(const std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>>& betas,
        const std::vector< FDAGWR_TRAITS::fd_obj_x_type>& abscissas) 
{
    // WRAP BETAS NON-STATIONARY
    Rcpp::List outerList(betas.size());

    for(std::size_t i = 0; i < betas.size(); ++i){
        Rcpp::List innerList(betas[i].size());
        Rcpp::CharacterVector innerNames(betas[i].size());

        for(std::size_t j = 0; j < betas[i].size(); ++j){
            innerList[j] = Rcpp::NumericVector(betas[i][j].cbegin(), betas[i][j].cend());
            innerNames[j] = std::string("unit_") + std::to_string(j+1);
        }

        innerList.names() = innerNames;

        //adding the abscissas
        Rcpp::List elem = Rcpp::List::create(
            Rcpp::Named("Beta_eval") = innerList,
            Rcpp::Named("Abscissa")  = Rcpp::NumericVector(abscissas.cbegin(), abscissas.cend()));

        outerList[i]=elem;
    }

    return outerList;
}




/*!
* @brief Wrapping the b (regression coefficients basis expansion coefficients) coming from a fwr fitted model
* @param b a variant, coming from fwr, containing a tuple with the different b
* @param names_bc vector of string with the names of stationary covariates
* @param basis_type_bc vector of string with the types of stationary betas basis
* @param basis_number_bc vector of integers with number of stationary betas basis
* @param knots_bc vector of vector of doubles with the knots of the basis expansion for stationary betas
* @param names_bnc vector of string with the names of non-stationary covariates
* @param basis_type_bnc vector of string with the types of non-stationary betas basis
* @param basis_number_bnc vector of integers with number of non-stationary betas basis
* @param knots_bnc vector of vector of doubles with the knots of the basis expansion for non-stationary betas
* @param names_be vector of string with the names of event-dependent covariates
* @param basis_type_be vector of string with the types of event-dependent betas basis
* @param basis_number_be vector of integers with number of event-dependent betas basis
* @param knots_be vector of vector of doubles with the knots of the basis expansion for event-dependent betas
* @param names_bs vector of string with the names of station-dependent covariates
* @param basis_type_bs vector of string with the types of station-dependent betas basis
* @param basis_number_bs vector of integers with number of station-dependent betas basis
* @param knots_bs vector of vector of doubles with the knots of the basis expansion for station-dependent betas
* @return an Rcpp::List containing the information of each beta basis expansion
*/
Rcpp::List 
wrap_b_to_R_list(const BTuple& b,
                 const std::vector<std::string>& names_bc                    = {},
                 const std::vector<std::string>& basis_type_bc               = {},
                 const std::vector<std::size_t>& basis_number_bc             = {},
                 const std::vector<FDAGWR_TRAITS::fd_obj_x_type> & knots_bc  = {},
                 const std::vector<std::string>& names_bnc                   = {}, 
                 const std::vector<std::string>& basis_type_bnc              = {},
                 const std::vector<std::size_t>& basis_number_bnc            = {},
                 const std::vector<FDAGWR_TRAITS::fd_obj_x_type> & knots_bnc = {},
                 const std::vector<std::string>& names_be                    = {},
                 const std::vector<std::string>& basis_type_be               = {},
                 const std::vector<std::size_t>& basis_number_be             = {},
                 const std::vector<FDAGWR_TRAITS::fd_obj_x_type> & knots_be  = {},
                 const std::vector<std::string>& names_bs                    = {},
                 const std::vector<std::string>& basis_type_bs               = {},
                 const std::vector<std::size_t>& basis_number_bs             = {},
                 const std::vector<FDAGWR_TRAITS::fd_obj_x_type> & knots_bs  = {}) 
{
    //using std::visit to wrap the betas basis expansion depending on the type of model fitted
    return std::visit([&](auto&& tup) -> Rcpp::List {
        using T = std::decay_t<decltype(tup)>;

        //FWR
        if constexpr (std::is_same_v<T, std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix >>>) 
        {
            //stationary b
            Rcpp::List bc = toRList(std::get<0>(tup),basis_type_bc,basis_number_bc,knots_bc);
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_B_NAMES::bc) = bc);
        }

        //FGWR
        if constexpr (std::is_same_v<T, std::tuple< std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>>>) 
        {
            //non stationary b
            Rcpp::List bnc = toRList(std::get<0>(tup),basis_type_bnc,basis_number_bnc,knots_bnc);
            if(!names_bnc.empty())
                bnc.names() = Rcpp::CharacterVector(names_bnc.cbegin(), names_bnc.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_B_NAMES::bnc) = bnc); 
        }

        //FMGWR
        else if constexpr (std::is_same_v<T, std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix >,
                                                         std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>>>) 
        {
            //stationary b
            Rcpp::List bc = toRList(std::get<0>(tup),basis_type_bc,basis_number_bc,knots_bc);
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());
            //non-stationary b                                            
            Rcpp::List bnc = toRList(std::get<1>(tup),basis_type_bnc,basis_number_bnc,knots_bnc);
            if (!names_bnc.empty())
                bnc.names() = Rcpp::CharacterVector(names_bnc.cbegin(), names_bnc.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_B_NAMES::bc)  = bc,
                                      Rcpp::Named(FDAGWR_B_NAMES::bnc) = bnc);
        }

        //FMSGWR
        else if constexpr (std::is_same_v<T, std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix >,
                                                         std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>,
                                                         std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>>>) 
        {
            //stationary b                                            
            Rcpp::List bc = toRList(std::get<0>(tup),basis_type_bc,basis_number_bc,knots_bc);
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());
            //event b                                            
            Rcpp::List be = toRList(std::get<1>(tup),basis_type_be,basis_number_be,knots_be);
            if (!names_be.empty())
                be.names() = Rcpp::CharacterVector(names_be.cbegin(), names_be.cend());
            //station b                                            
            Rcpp::List bs = toRList(std::get<2>(tup),basis_type_bs,basis_number_bs,knots_bs);
            if (!names_bs.empty())
                bs.names() = Rcpp::CharacterVector(names_bs.cbegin(), names_bs.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_B_NAMES::bc) = bc,
                                      Rcpp::Named(FDAGWR_B_NAMES::be) = be,
                                      Rcpp::Named(FDAGWR_B_NAMES::bs) = bs);
        }
    }, b);
}



/*!
* @brief Wrapping the betas (regression coefficients) discrete evaluations coming from a fwr fitted model
* @param betas a variant, coming from fwr, containing a tuple with the different betas evaluations
* @param abscissas vector with the abscissa over which the betas are evaluated
* @param names_beta_c vector of string with the names of stationary covariates
* @param names_beta_nc vector of string with the names of non-stationary covariates
* @param names_beta_e vector of string with the names of event-dependent covariates
* @param names_beta_s vector of string with the names of station-dependent covariates
* @return an Rcpp::List containing the information of each beta evaluation
*/
Rcpp::List 
wrap_beta_to_R_list(const BetasTuple& betas,
                    const std::vector<FDAGWR_TRAITS::fd_obj_x_type> & abscissas,
                    const std::vector<std::string>& names_beta_c  = {},
                    const std::vector<std::string>& names_beta_nc = {},
                    const std::vector<std::string>& names_beta_e  = {},
                    const std::vector<std::string>& names_beta_s  = {}) 
{
    //using std::visit to wrap the betas discrete evaluation depending on the type of model fitted
    return std::visit([&](auto&& tup) -> Rcpp::List {
        using T = std::decay_t<decltype(tup)>;

        //FWR
        if constexpr (std::is_same_v< T, std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type >>> >) 
        {
            //stationary beta
            Rcpp::List beta_c = toRList(std::get<0>(tup),abscissas);
            if (!names_beta_c.empty())
                beta_c.names() = Rcpp::CharacterVector(names_beta_c.cbegin(), names_beta_c.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_BETAS_NAMES::beta_c) = beta_c);
        }

        //FGWR
        if constexpr (std::is_same_v<T, std::tuple< std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type >>>>>) 
        {
            //non-stationary beta
            Rcpp::List beta_nc = toRList(std::get<0>(tup),abscissas);
            if(!names_beta_nc.empty())
                beta_nc.names() = Rcpp::CharacterVector(names_beta_nc.cbegin(), names_beta_nc.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_BETAS_NAMES::beta_nc) = beta_nc);
        }

        //FMGWR
        else if constexpr (std::is_same_v<T, std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type >>,
                                                         std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type >>>>> ) 
        {
            //stationary beta
            Rcpp::List beta_c = toRList(std::get<0>(tup),abscissas);
            if (!names_beta_c.empty())
                beta_c.names() = Rcpp::CharacterVector(names_beta_c.cbegin(), names_beta_c.cend());
            //non-stationary beta
            Rcpp::List beta_nc = toRList(std::get<1>(tup),abscissas);
            if (!names_beta_nc.empty())
                beta_nc.names() = Rcpp::CharacterVector(names_beta_nc.cbegin(), names_beta_nc.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_BETAS_NAMES::beta_c)  = beta_c,
                                      Rcpp::Named(FDAGWR_BETAS_NAMES::beta_nc) = beta_nc);
        }

        //FMSGWR
        else if constexpr (std::is_same_v<T, std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type >>,
                                                         std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type >>>,
                                                         std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type >>>>>) 
        {
            //stationary beta
            Rcpp::List beta_c = toRList(std::get<0>(tup),abscissas);
            if (!names_beta_c.empty())
                beta_c.names() = Rcpp::CharacterVector(names_beta_c.cbegin(), names_beta_c.cend());
            //event beta
            Rcpp::List beta_e = toRList(std::get<1>(tup),abscissas);
            if (!names_beta_e.empty())
                beta_e.names() = Rcpp::CharacterVector(names_beta_e.cbegin(), names_beta_e.cend());
            //station beta
            Rcpp::List beta_s = toRList(std::get<2>(tup),abscissas);
            if (!names_beta_s.empty())
                beta_s.names() = Rcpp::CharacterVector(names_beta_s.cbegin(), names_beta_s.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_BETAS_NAMES::beta_c) = beta_c,
                                      Rcpp::Named(FDAGWR_BETAS_NAMES::beta_e) = beta_e,
                                      Rcpp::Named(FDAGWR_BETAS_NAMES::beta_s) = beta_s);
        }
    }, betas);
}



/*!
* @brief Wrapping the elements needed to reconstruct the partial residuals of a fwr fitted model
* @param p_res a variant, coming from fwr, containing a tuple with the different elements needed to reconstruct the element needed to reconstruct the partial residuals
* @return an Rcpp::List containing the information for reconstructing the partial residuals
*/
Rcpp::List 
wrap_PRes_to_R_list(const PartialResidualTuple& p_res)
{
    //using std::visit to wrap the element needed to reconstruct the partial residuals of a specific fwr model
    return std::visit([&](auto&& tup) -> Rcpp::List {
        using T = std::decay_t<decltype(tup)>;

        //FMSGWR
        if constexpr (std::is_same_v<T, std::tuple< FDAGWR_TRAITS::Dense_Matrix, std::vector< FDAGWR_TRAITS::Dense_Matrix >, std::vector< FDAGWR_TRAITS::Dense_Matrix >>>) 
        {
            return Rcpp::List::create(Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat) = Rcpp::wrap(std::get<0>(tup)),
                                      Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::p_res_A__)         = toRList(std::get<1>(tup),false),
                                      Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::p_res_B__for_K)    = toRList(std::get<2>(tup),false));
        } 

        //FMGWR
        else if constexpr (std::is_same_v<T, std::tuple< FDAGWR_TRAITS::Dense_Matrix >>) 
        {
            return Rcpp::List::create(Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat) = Rcpp::wrap(std::get<0>(tup)));
        } 

        //FGWR and FWR
        else 
        {
            return Rcpp::List::create();    // no partial residuals here
        }
    }, p_res);
}



/*!
* @brief Wrapping the elements needed to wrap the discretized prediction of a fwr into an Rcpp::List
* @param pred vector (one element for each prediction) of vector (one element for each abscissa) with the discretized prediction
* @param abscissa containing the abscissa over which the prediction are evalauted
* @param pred_coeff the basis expansion coefficients for each prediction (num basis x number prediction)
* @param basis_t string with the basis type of the prediction
* @param n_basis integer with the number of basis of the prediction
* @param basis_deg integer with the degree of the basis of the prediction basis expansion
* @param basis_knots knots for the basis expansion of the prediction
* @return an Rcpp::List containing the wrapped prediction
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
Rcpp::List
wrap_prediction_to_R_list(const std::vector< std::vector<OUTPUT>> &pred,
                          const std::vector< INPUT> &abscissa,
                          const FDAGWR_TRAITS::Dense_Matrix &pred_coeff,
                          std::string basis_t,
                          std::size_t n_basis,
                          std::size_t basis_deg,
                          const FDAGWR_TRAITS::Dense_Matrix &basis_knots)
{
    Rcpp::List pred_w;

    //evaluation part
    std::size_t number_pred = pred.size();
    Rcpp::List predictions(number_pred);
    Rcpp::CharacterVector names(number_pred);

    for(std::size_t i = 0; i < number_pred; ++i){
        predictions[i] = Rcpp::wrap(pred[i]);
        names[i] = std::string("unit_pred_") + std::to_string(i+1);
    }

    predictions.names() = names;

    //pred evaluation
    Rcpp::List pred_ev;
    pred_ev[FDAGWR_HELPERS_for_PRED_NAMES::pred + "_ev"] = predictions;
    pred_ev[FDAGWR_HELPERS_for_PRED_NAMES::abscissa + "_ev"] = abscissa;
    //pred fd
    Rcpp::List pred_fd;
    pred_fd[FDAGWR_HELPERS_for_PRED_NAMES::pred + "_" + FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis] = Rcpp::wrap(pred_coeff);
    pred_fd[FDAGWR_HELPERS_for_PRED_NAMES::pred + "_" + FDAGWR_HELPERS_for_PRED_NAMES::basis_t]     = basis_t;
    pred_fd[FDAGWR_HELPERS_for_PRED_NAMES::pred + "_" + FDAGWR_HELPERS_for_PRED_NAMES::n_basis]     = n_basis;
    pred_fd[FDAGWR_HELPERS_for_PRED_NAMES::pred + "_" + FDAGWR_HELPERS_for_PRED_NAMES::basis_deg]   = basis_deg;
    pred_fd[FDAGWR_HELPERS_for_PRED_NAMES::pred + "_" + FDAGWR_HELPERS_for_PRED_NAMES::basis_knots] = Rcpp::wrap(basis_knots);

    //returning everything
    pred_w["evaluation"] = pred_ev;
    pred_w["fd"]         = pred_fd;

    return pred_w;
}

#endif  /*FDAGWR_UTILITIES_HPP*/