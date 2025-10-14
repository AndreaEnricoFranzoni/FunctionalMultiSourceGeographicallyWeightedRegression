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


#ifndef FDAGWR_UTILITIES_HPP
#define FDAGWR_UTILITIES_HPP


#include "traits_fdagwr.hpp"
#include <RcppEigen.h>


// Primaria: vuota
template <typename T>
struct extract_template;

// Specializzazione generica per template con soli parametri di tipo
template <template <typename...> class TT, typename... Args>
struct extract_template<TT<Args...>> {
    // Alias template che ricrea il template di partenza
    template <typename... Ts>
    using template_type = TT<Ts...>;
};

// Helper per normalizzare il tipo (toglie const/ref)
template <typename T>
using extract_template_t = extract_template<
    std::remove_cv_t<std::remove_reference_t<T>>
>;






/*
template <typename Domain>
struct basis {};

template <template <typename> class BasisTemplate>
struct wrapper {};

int main() {
    std::unique_ptr<basis<int>> ptr;

    using PointeeType   = typename decltype(ptr)::element_type; // basis<int>
    using Extracted     = extract_template_t<PointeeType>;
    using BasisTemplate = Extracted::template_type; // <— qui è un alias template

    wrapper<BasisTemplate> w; // funziona
}

*/










/////////////////////////
/// WRAP B STATIONARY ///
/////////////////////////
/*!
* @brief Converte un vettore di Eigen::MatrixXd in una R list con matrici di double (b stazionari)
*/
Rcpp::List 
toRList(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
        bool add_unit_number = false) 
{
    Rcpp::List out(b.size());
    Rcpp::CharacterVector names(b.size());
    for (std::size_t i = 0; i < b.size(); i++) {
        out[i] = Rcpp::wrap(b[i]); // RcppEigen converte Eigen::MatrixXd
        names[i] = std::string("unit_") + std::to_string(i+1);
    }

    if (add_unit_number){   out.names() = names;}

    return out;
}

/*!
* @brief Helper per aggiungere extra info ai b stazionari
*/
Rcpp::List 
enrichB(const FDAGWR_TRAITS::Dense_Matrix& b,
        const std::string& basis_type,
        std::size_t basis_number,
        const std::vector<FDAGWR_TRAITS::fd_obj_x_type>& basis_knots) 
{
    return Rcpp::List::create(Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis) = Rcpp::wrap(b),
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::basis_t)     = basis_type,
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::n_basis)     = basis_number,
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::basis_knots) = Rcpp::NumericVector(basis_knots.cbegin(), basis_knots.cend()));
}

/*!
* @brief Converte un vettore di Eigen::MatrixXd in una R list con matrici di double (b stazionari), con extra infos
*/
Rcpp::List 
toRList(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
        const std::vector<std::string>& basis_type,
        const std::vector<std::size_t>& basis_number,
        const std::vector<FDAGWR_TRAITS::fd_obj_x_type>& basis_knots) 
{
    Rcpp::List out(b.size());
    for (size_t j = 0; j < b.size(); ++j) {
        out[j] = enrichB(b[j],
                         basis_type[j],
                         basis_number[j],
                         basis_knots);
    }
    return out;
}




/////////////////////////////
/// WRAP B NON-STATIONARY ///
/////////////////////////////
/*!
* @brief Converte un vettore di vettori di Eigen::MatrixXd in una R list con liste di matrici di double (b non stazionari)
*/
Rcpp::List
toRList(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& b)
{
    Rcpp::List outerList(b.size());

    for(std::size_t i = 0; i < b.size(); ++i){
        Rcpp::List innerList(b[i].size());
        Rcpp::CharacterVector innerNames(b[i].size());

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
* @brief Helper per aggiungere extra info ai b non stazionari
*/
Rcpp::List 
enrichB(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
        const std::string& basis_type,
        std::size_t basis_number,
        const std::vector<FDAGWR_TRAITS::fd_obj_x_type>& basis_knots) 
{
    return Rcpp::List::create(Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::coeff_basis) = toRList(b,true),
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::basis_t)     = basis_type,
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::n_basis)     = basis_number,
                              Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::basis_knots) = Rcpp::NumericVector(basis_knots.cbegin(), basis_knots.cend()));
}


/*!
* @brief Converte un vettore di vettori di Eigen::MatrixXd in una R list con liste di matrici di double (b non stazionari), con extra info
*/
Rcpp::List 
toRList(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& b,
        const std::vector<std::string>& basis_type,
        const std::vector<std::size_t>& basis_number,
        const std::vector<FDAGWR_TRAITS::fd_obj_x_type>& basis_knots) 
{
    Rcpp::List out(b.size());
    for (size_t j = 0; j < b.size(); ++j) {
        out[j] = enrichB(b[j],
                         basis_type[j],
                         basis_number[j],
                         basis_knots);
    }
    return out;
}






/////////////////////////////
/// WRAP BETAS STATIONARY ///
/////////////////////////////
/*!
* @brief Converte un vettore di vettore di double in una R list di vettori di double (betas stazionari)
*/
Rcpp::List 
toRList(const std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>& betas) 
{
    Rcpp::List out(betas.size());
    for (size_t i = 0; i < betas.size(); ++i) {
        out[i] = Rcpp::NumericVector(betas[i].cbegin(), betas[i].cend());
    }
    return out;
}

/*!
* @brief Converte un std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>> in una R list con vettori di double (betas stazionari), con extra infos
*/
Rcpp::List 
toRList(const std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>& betas,
        const std::vector< FDAGWR_TRAITS::fd_obj_x_type>& abscissas) 
{
    Rcpp::List out(betas.size());
    for (size_t j = 0; j < betas.size(); ++j) {
        Rcpp::List elem = Rcpp::List::create(Rcpp::Named("Beta_eval") = Rcpp::NumericVector(betas[j].cbegin(), betas[j].cend()),
                                             Rcpp::Named("Abscissa")  = Rcpp::NumericVector(abscissas.cbegin(), abscissas.cend()));

        out[j] = elem;}

    return out;
}



/////////////////////////////////
/// WRAP BETAS NON-STATIONARY ///
/////////////////////////////////
/*!
* @brief Converte un vettore di vettori di double in una R list con liste di vettori di double (betas non stazionari)
*/
Rcpp::List
toRList(const std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>>& betas)
{
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
* @brief Converte un std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>> in una R list con vettori di double (betas non stazionari), con extra infos
*/
Rcpp::List 
toRList(const std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>>& betas,
        const std::vector< FDAGWR_TRAITS::fd_obj_x_type>& abscissas) 
{

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
* @brief Wrapping the b
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
    return std::visit([&](auto&& tup) -> Rcpp::List {
        using T = std::decay_t<decltype(tup)>;

        if constexpr (std::is_same_v<T, std::tuple<std::vector<FDAGWR_TRAITS::Dense_Matrix>>>) {
            Rcpp::List bc = toRList(std::get<0>(tup),basis_type_bc,basis_number_bc,knots_bc);
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_B_NAMES::bc) = bc);
        }
        else if constexpr (std::is_same_v<T, std::tuple<std::vector<FDAGWR_TRAITS::Dense_Matrix>,
                                                       std::vector<std::vector<FDAGWR_TRAITS::Dense_Matrix>> >>) {
            Rcpp::List bc = toRList(std::get<0>(tup),basis_type_bc,basis_number_bc,knots_bc);
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());

            Rcpp::List bnc = toRList(std::get<1>(tup),basis_type_bnc,basis_number_bnc,knots_bnc);
            if (!names_bnc.empty())
                bnc.names() = Rcpp::CharacterVector(names_bnc.cbegin(), names_bnc.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_B_NAMES::bc)  = bc,
                                      Rcpp::Named(FDAGWR_B_NAMES::bnc) = bnc);
        }
        else if constexpr (std::is_same_v<T, std::tuple<std::vector<FDAGWR_TRAITS::Dense_Matrix>,
                                                       std::vector<std::vector<FDAGWR_TRAITS::Dense_Matrix>>,
                                                       std::vector<std::vector<FDAGWR_TRAITS::Dense_Matrix>> >>) {

            Rcpp::List bc = toRList(std::get<0>(tup),basis_type_bc,basis_number_bc,knots_bc);
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());

            Rcpp::List be = toRList(std::get<1>(tup),basis_type_be,basis_number_be,knots_be);
            if (!names_be.empty())
                be.names() = Rcpp::CharacterVector(names_be.cbegin(), names_be.cend());

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
* @brief Wrapping the betas
*/
Rcpp::List 
wrap_beta_to_R_list(const BetasTuple& betas,
                    const std::vector<FDAGWR_TRAITS::fd_obj_x_type> & abscissas,
                    const std::vector<std::string>& names_beta_c  = {},
                    const std::vector<std::string>& names_beta_nc = {},
                    const std::vector<std::string>& names_beta_e  = {},
                    const std::vector<std::string>& names_beta_s  = {}) 
{
    return std::visit([&](auto&& tup) -> Rcpp::List {
        using T = std::decay_t<decltype(tup)>;

        if constexpr (std::is_same_v< T, std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>> >) {

            Rcpp::List beta_c = toRList(std::get<0>(tup),abscissas);
            if (!names_beta_c.empty())
                beta_c.names() = Rcpp::CharacterVector(names_beta_c.cbegin(), names_beta_c.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_BETAS_NAMES::beta_c) = beta_c);
        }
        else if constexpr (std::is_same_v<T, std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>,
                                                         std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>> > >) {

            Rcpp::List beta_c = toRList(std::get<0>(tup),abscissas);
            if (!names_beta_c.empty())
                beta_c.names() = Rcpp::CharacterVector(names_beta_c.cbegin(), names_beta_c.cend());

            Rcpp::List beta_nc = toRList(std::get<1>(tup),abscissas);
            if (!names_beta_nc.empty())
                beta_nc.names() = Rcpp::CharacterVector(names_beta_nc.cbegin(), names_beta_nc.cend());

            return Rcpp::List::create(Rcpp::Named(FDAGWR_BETAS_NAMES::beta_c)  = beta_c,
                                      Rcpp::Named(FDAGWR_BETAS_NAMES::beta_nc) = beta_nc);
        }
        else if constexpr (std::is_same_v<T, std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>,
                                                         std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>>,
                                                         std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>> >>) {

            Rcpp::List beta_c = toRList(std::get<0>(tup),abscissas);
            if (!names_beta_c.empty())
                beta_c.names() = Rcpp::CharacterVector(names_beta_c.cbegin(), names_beta_c.cend());

            Rcpp::List beta_e = toRList(std::get<1>(tup),abscissas);
            if (!names_beta_e.empty())
                beta_e.names() = Rcpp::CharacterVector(names_beta_e.cbegin(), names_beta_e.cend());

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
* @brief Wrapping the objects needed for reconstructing the partial residuals
*/
Rcpp::List 
wrap_PRes_to_R_list(const PartialResidualTuple& p_res)
{
  return std::visit([&](auto&& tup) -> Rcpp::List {
    using T = std::decay_t<decltype(tup)>;

    //two sources
    if constexpr (std::is_same_v<T, std::tuple< FDAGWR_TRAITS::Dense_Matrix, std::vector< FDAGWR_TRAITS::Dense_Matrix >, std::vector< FDAGWR_TRAITS::Dense_Matrix >>>) {
      return Rcpp::List::create(Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::p_res_c_tilde_hat) = Rcpp::wrap(std::get<0>(tup)),
                                Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::p_res_A__)         = toRList(std::get<1>(tup),false),
                                Rcpp::Named(FDAGWR_HELPERS_for_PRED_NAMES::p_res_B__for_K)    = toRList(std::get<2>(tup),false));
    } 
    //one source
    else if constexpr (std::is_same_v<T, std::tuple<int, int>>) {
      auto [a, b] = tup;
      return Rcpp::List::create(a, b);
    } 
    //stationary
    else {
      return Rcpp::List::create(); // lista vuota
    }
  }, p_res);
}




/*!
* @brief Wrapping the prediction
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
Rcpp::List
wrap_prediction_to_R_list(const std::vector< std::vector<OUTPUT>> & pred,
                          const std::vector< INPUT> &abscissa)
{
    Rcpp::List pred_w;
    std::size_t number_pred = pred.size();
    Rcpp::List predictions(number_pred);
    Rcpp::CharacterVector names(number_pred);

    for(std::size_t i = 0; i < number_pred; ++i){
        predictions[i] = Rcpp::wrap(pred[i]);
        names[i] = std::string("unit_pred_") + std::to_string(i+1);
    }

    predictions.names() = names;

    //returning it
    pred_w[FDAGWR_HELPERS_for_PRED_NAMES::pred + "_ev"] = predictions;
    //adding also the abscissa
    pred_w[FDAGWR_HELPERS_for_PRED_NAMES::abscissa] = abscissa;
    return pred_w;
}

#endif  /*FDAGWR_UTILITIES_HPP*/