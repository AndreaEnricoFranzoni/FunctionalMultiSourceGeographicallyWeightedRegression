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


#include "fgwr_predictor.hpp"






/*!
* @brief Dewrap b, for stationary covariates: me li incolonna tutti 
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
FDAGWR_TRAITS::Dense_Matrix 
fgwr_predictor<INPUT,OUTPUT>::dewrap_b(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
                                       const std::vector<std::size_t>& L_j) 
const
{
    //input coherency
    assert(b.size() == L_j.size());
    for(std::size_t i = 0; i < b.size(); ++i){  assert((b[i].cols()==1) && (b[i].rows()==L_j[i]));}

    FDAGWR_TRAITS::Dense_Matrix b_dewrapped(std::reduce(L_j.cbegin(),L_j.cend(),static_cast<std::size_t>(0)),1);

    for(std::size_t j = 0; j < L_j.size(); ++j)
    {
        std::size_t start_idx = std::reduce(L_j.cbegin(),std::next(L_j.cbegin(),j),static_cast<std::size_t>(0));
        b_dewrapped.block(start_idx,0,L_j[j],1) = b[j];
    }

    return b_dewrapped;
}



//per ogni unità
/*!
* @brief Per ogni unità, mi fa i b incolonnati
* @param b vettore esterno: le covariate: ogni elemento è un vettore che contiene, per quella covariate, i b non-stazionari in ogni unità
* @return un vettore coi b incolonnati per ogni unità
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fgwr_predictor<INPUT,OUTPUT>::dewrap_b(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& b,
                                       const std::vector<std::size_t>& L_j,
                                       std::size_t n) 
const
{
    //input coherency
    assert(b.size() == L_j.size());
    std::size_t q = L_j.size();
    for(std::size_t j = 0; j < q; ++j){ assert(b[j].size() == n);}

    std::vector< FDAGWR_TRAITS::Dense_Matrix > b_dewrapped;
    b_dewrapped.reserve(n);

    for(std::size_t i = 0; i < n; ++i){

        std::vector< FDAGWR_TRAITS::Dense_Matrix > b_i;
        b_i.reserve(q);
        for(std::size_t j = 0; j < q; ++j){     b_i.push_back(b[j][i]);}
        b_dewrapped.push_back(this->dewrap_b(b_i,L_j));
    }

    return b_dewrapped;
}


// EVAL BETAS
//stationary
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< std::vector<OUTPUT> >
fgwr_predictor<INPUT,OUTPUT>::eval_betas(const functional_matrix<INPUT,OUTPUT> &beta,
                                         std::size_t q,
                                         std::vector<INPUT> abscissa) 
const
{
    //input coherency
    assert((beta.rows() == q) && (beta.cols() == 1));
    //number of evaluations
    std::size_t n_abs = abscissa.size();

    //reserving
    std::vector< std::vector<OUTPUT>> beta_ev;    
    beta_ev.reserve(q);        

    for (std::size_t j = 0; j < q; ++j)
    {
        std::vector<OUTPUT> beta_j_ev;
        beta_j_ev.resize(n_abs);

#ifdef _OPENMP
#pragma omp parallel for shared(beta_j_ev,j,abscissa,n_abs) num_threads(this->number_threads())
#endif
        for(std::size_t i = 0; i < n_abs; ++i)
        {
            beta_j_ev[i] = beta(j,0)(abscissa[i]);
        }

        beta_ev.push_back(beta_j_ev);
    }

    return beta_ev;
}



//non-stationary: risultato: un vettore di len q, che contiene, per ogni n_pred, le valutazioni dei beta
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< std::vector< std::vector<OUTPUT>>>
fgwr_predictor<INPUT,OUTPUT>::eval_betas(const std::vector< functional_matrix<INPUT,OUTPUT>> &beta,
                                         std::size_t q,
                                         std::vector<INPUT> abscissa)
const
{
    std::size_t n_pred = beta.size();
    std::size_t n_abs  = abscissa.size();
    //input coherency
    for(std::size_t i = 0; i < n_pred; ++i){    assert((beta[i].rows() == q) && (beta[i].cols() == 1));}
    

    //reserving
    std::vector< std::vector< std::vector<OUTPUT>> > beta_ev;    
    beta_ev.reserve(q);  
    
    for(std::size_t j = 0; j < q; ++j)
    {
        std::vector< std::vector<OUTPUT>> beta_j_ev;
        beta_j_ev.reserve(n_pred);

        for(std::size_t i = 0; i < n_pred; ++i)
        {
            std::vector<OUTPUT> beta_j_i_ev;
            beta_j_i_ev.resize(n_abs);

#ifdef _OPENMP
#pragma omp parallel for shared(beta_j_i_ev,j,i,abscissa,n_abs) num_threads(this->number_threads())
#endif
            for(std::size_t i_ev = 0; i_ev < n_abs; ++i_ev)
            {
                beta_j_i_ev[i_ev] = beta[i](j,0)(abscissa[i_ev]);
            }

            beta_j_ev.push_back(beta_j_i_ev);
        }

        beta_ev.push_back(beta_j_ev);
    }

    return beta_ev;
}