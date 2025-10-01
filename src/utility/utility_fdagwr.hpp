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







/*!
* @brief Converte un vettore di Eigen::MatrixXd in una R list con matrici di double
*/
Rcpp::List 
toRList(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& mats) 
{
    Rcpp::List out(mats.size());
    for (std::size_t i = 0; i < mats.size(); i++) {
        out[i] = Rcpp::wrap(mats[i]); // RcppEigen converte Eigen::MatrixXd
    }
    return out;
}

/*!
* @brief Converte un vettore di vettori di Eigen::MatrixXd in una R list con liste di matrici di double
*/
Rcpp::List
toRList(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& mats )
{
    Rcpp::List outerList(mats.size());

    for(std::size_t i = 0; i < mats.size(); ++i){
        Rcpp::List innerList(mats[i].size());
        Rcpp::CharacterVector innerNames(mats[i].size());

        for(std::size_t j = 0; j < mats[i].size(); ++j){
            innerList[j] = Rcpp::wrap(mats[i][j]);
            innerNames[j] = std::string("Unit ") + std::to_string(j+1);
        }

        innerList.names() = innerNames;
        outerList[i]=(innerList);
    }

    return outerList;
}

/*!
* @brief Converte un vettore di vettore di double in una R list di vettori di double
*/
Rcpp::List 
toRList(const std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>& betas) {
    Rcpp::List out(betas.size());
    for (size_t i = 0; i < betas.size(); ++i) {
        out[i] = Rcpp::NumericVector(betas[i].cbegin(), betas[i].cend());
    }
    return out;
}

/*!
* @brief Converte un vettore di vettori di double in una R list con liste di vettori di double
*/
Rcpp::List
toRList(const std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>>& betas)
{
    Rcpp::List outerList(betas.size());

    for(std::size_t i = 0; i < betas.size(); ++i){
        Rcpp::List innerList(betas[i].size());
        Rcpp::CharacterVector innerNames(betas[i].size());

        for(std::size_t j = 0; j < betas[i].size(); ++j){
            innerList[j] = toRList(betas[i]);
            innerNames[j] = std::string("Unit ") + std::to_string(j+1);
        }

        innerList.names() = innerNames;
        outerList[i]=(innerList);
    }

    return outerList;
}







// funzione di conversione dei b coefficients
/*
Rcpp::List wrap_b_to_R_list(const BTuple& r) {
    return std::visit([](auto&& tup) -> Rcpp::List {
        using T = std::decay_t<decltype(tup)>;
        
        if constexpr (std::is_same_v< T,  std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix > > >) {
            return Rcpp::List::create(
                Rcpp::Named("bc") = toRList(std::get<0>(tup))
            );
        } 
        else if constexpr (std::is_same_v< T, std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix >, std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > > > >) {
            return Rcpp::List::create(
                Rcpp::Named("bc") = toRList(std::get<0>(tup)),
                Rcpp::Named("bnc") = toRList(std::get<1>(tup))
            );
        } 
        else if constexpr (std::is_same_v< T, std::tuple< std::vector< FDAGWR_TRAITS::Dense_Matrix >, std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > >, std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > > > >) {
            return Rcpp::List::create(
                Rcpp::Named("bc") = toRList(std::get<0>(tup)),
                Rcpp::Named("be") = toRList(std::get<1>(tup)),
                Rcpp::Named("bs") = toRList(std::get<2>(tup))
            );
        }
    }, r);
}

*/


Rcpp::List wrap_b_to_R_list(const BTuple& r,
                            const std::vector<std::string>& names_bc = {},
                            const std::vector<std::string>& names_bnc = {},
                            const std::vector<std::string>& names_be = {},
                            const std::vector<std::string>& names_bs = {}) {
    return std::visit([&](auto&& tup) -> Rcpp::List {
        using T = std::decay_t<decltype(tup)>;

        if constexpr (std::is_same_v<T, std::tuple<std::vector<FDAGWR_TRAITS::Dense_Matrix>>>) {
            Rcpp::List bc = toRList(std::get<0>(tup));
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());

            return Rcpp::List::create(Rcpp::Named("bc") = bc);
        }
        else if constexpr (std::is_same_v<T, std::tuple<std::vector<FDAGWR_TRAITS::Dense_Matrix>,
                                                       std::vector<std::vector<FDAGWR_TRAITS::Dense_Matrix>> >>) {
            Rcpp::List bc = toRList(std::get<0>(tup));
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());

            Rcpp::List bnc = toRList(std::get<1>(tup));
            if (!names_bnc.empty())
                bnc.names() = Rcpp::CharacterVector(names_bnc.cbegin(), names_bnc.cend());

            return Rcpp::List::create(Rcpp::Named("bc") = bc,
                                      Rcpp::Named("bnc") = bnc);
        }
        else if constexpr (std::is_same_v<T, std::tuple<std::vector<FDAGWR_TRAITS::Dense_Matrix>,
                                                       std::vector<std::vector<FDAGWR_TRAITS::Dense_Matrix>>,
                                                       std::vector<std::vector<FDAGWR_TRAITS::Dense_Matrix>> >>) {
            Rcpp::List bc = toRList(std::get<0>(tup));
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());

            Rcpp::List be = toRList(std::get<1>(tup));
            if (!names_be.empty())
                be.names() = Rcpp::CharacterVector(names_be.cbegin(), names_be.cend());

            Rcpp::List bs = toRList(std::get<2>(tup));
            if (!names_bs.empty())
                bs.names() = Rcpp::CharacterVector(names_bs.cbegin(), names_bs.cend());

            return Rcpp::List::create(Rcpp::Named("bc") = bc,
                                      Rcpp::Named("be") = be,
                                      Rcpp::Named("bs") = bs);
        }
    }, r);
}






Rcpp::List wrap_beta_to_R_list(const BetasTuple& r,
                               const std::vector<std::string>& names_bc = {},
                               const std::vector<std::string>& names_bnc = {},
                               const std::vector<std::string>& names_be = {},
                               const std::vector<std::string>& names_bs = {}) {
    return std::visit([&](auto&& tup) -> Rcpp::List {
        using T = std::decay_t<decltype(tup)>;

        if constexpr (std::is_same_v< T, std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>> >) {
            Rcpp::List bc = toRList(std::get<0>(tup));
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());

            return Rcpp::List::create(Rcpp::Named("beta_c") = bc);
        }
        else if constexpr (std::is_same_v<T, std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>,
                                                         std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>> > >) {
            Rcpp::List bc = toRList(std::get<0>(tup));
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());

            Rcpp::List bnc = toRList(std::get<1>(tup));
            if (!names_bnc.empty())
                bnc.names() = Rcpp::CharacterVector(names_bnc.cbegin(), names_bnc.cend());

            return Rcpp::List::create(Rcpp::Named("beta_c") = bc,
                                      Rcpp::Named("beta_nc") = bnc);
        }
        else if constexpr (std::is_same_v<T, std::tuple< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>,
                                                         std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>>,
                                                         std::vector< std::vector< std::vector< FDAGWR_TRAITS::fd_obj_y_type>>> >>) {
            Rcpp::List bc = toRList(std::get<0>(tup));
            if (!names_bc.empty())
                bc.names() = Rcpp::CharacterVector(names_bc.cbegin(), names_bc.cend());

            Rcpp::List be = toRList(std::get<1>(tup));
            if (!names_be.empty())
                be.names() = Rcpp::CharacterVector(names_be.cbegin(), names_be.cend());

            Rcpp::List bs = toRList(std::get<2>(tup));
            if (!names_bs.empty())
                bs.names() = Rcpp::CharacterVector(names_bs.cbegin(), names_bs.cend());

            return Rcpp::List::create(Rcpp::Named("beta_c") = bc,
                                      Rcpp::Named("beta_e") = be,
                                      Rcpp::Named("beta_s") = bs);
        }
    }, r);
}





#endif  /*FDAGWR_UTILITIES_HPP*/




/*
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List creaListaFinale(const BTuple& r) {
    // Richiamo wrap_b_to_R_list per ottenere i coefficienti
    Rcpp::List b_list = wrap_b_to_R_list(r);

    // --- primo elemento: FWGR
    std::string fwgr_str = "FWGR string"; // oppure passala come parametro

    // --- secondo elemento: C
    Rcpp::List bc = b_list["bc"];           // vettore di q elementi
    int q = bc.size();
    Rcpp::List C(q);
    for (int i = 0; i < q; ++i) {
        // ogni elemento di bc è già una lista o un NumericVector
        Rcpp::NumericVector coeff = bc[i];

        // esempi dummy per beta, type, number, knots (puoi sostituire con dati reali)
        Rcpp::NumericVector beta = Rcpp::NumericVector::create(0.1, 0.2, 0.3);
        std::string type = "type_example";
        int number = i + 1;
        Rcpp::NumericVector knots = Rcpp::NumericVector::create(0.5, 1.5);

        C[i] = Rcpp::List::create(
            Rcpp::Named("coefficiente") = coeff,
            Rcpp::Named("beta") = beta,
            Rcpp::Named("type") = type,
            Rcpp::Named("number") = number,
            Rcpp::Named("knots") = knots
        );
    }

    // --- terzo elemento: E
    Rcpp::List be = b_list.containsElementNamed("be") ? b_list["be"] : Rcpp::List();
    int k = be.size();
    Rcpp::List E(k);
    for (int i = 0; i < k; ++i) {
        Rcpp::List first_sublist  = Rcpp::List::create(be[i], be[i]);  // esempio, inserisci dati reali
        Rcpp::List second_sublist = Rcpp::List::create(be[i], be[i]);  // esempio
        E[i] = Rcpp::List::create(first_sublist, second_sublist);
    }

    // --- quarto elemento: S uguale a E
    Rcpp::List S = clone(E);

    // --- costruisco lista finale
    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("FWGR") = fwgr_str,
        Rcpp::Named("C") = C,
        Rcpp::Named("E") = E,
        Rcpp::Named("S") = S
    );

    return out;
}

*/

