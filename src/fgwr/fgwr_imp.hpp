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


#include "fgwr.hpp"


/*
////////////////////////
/////  PENALTY    /////
////////////////////////


template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> >
fgwr<INPUT,OUTPUT>::compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                                    const functional_matrix<INPUT,OUTPUT> &X_t,
                                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                    const functional_matrix<INPUT,OUTPUT> &X,
                                    const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                    const FDAGWR_TRAITS::Sparse_Matrix &R)
const
{
    //the vector contains factorization of the matrix
    std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > penalty;
    penalty.resize(W.size());

    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);   //necessary to compute the sum later

#ifdef _OPENMP
#pragma omp parallel for shared(penalty,base,base_t,X,X_t,W,_R_,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {
        //dimension: L x L, where L is the number of basis
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(fm_prod(base_t,X_t),W[i],m_number_threads),X,m_number_threads),base);

        //performing integration and factorization
        penalty[i] = Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >( this->fm_integration(integrand) + _R_ );    
        // penalty[i].solve(M) equivale a fare elemento penalty[i], che è una matrice inversa, times M
    }
    
    return penalty;
}




template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > >
fgwr<INPUT,OUTPUT>::compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                    const functional_matrix<INPUT,OUTPUT> &X_crossed,
                                    const FDAGWR_TRAITS::Sparse_Matrix &R) 
const
{
    //the vector contains factorization of the matrix
    std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > penalty;
    penalty.resize(W.size());

    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);   //necessary to compute the sum later

#ifdef _OPENMP
#pragma omp parallel for shared(penalty,X_crossed_t,X_crossed,W,_R_,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {
        //dimension: L x L, where L is the number of basis
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_crossed_t,W[i],m_number_threads),X_crossed,m_number_threads);

        //performing integration and factorization
        penalty[i] = Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >( this->fm_integration(integrand) + _R_ );    
        // penalty[i].solve(M) equivale a fare elemento penalty[i], che è una matrice inversa, times M
    }
    
    return penalty;
}



template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                                    const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                                    const functional_matrix<INPUT,OUTPUT> &X_crossed,
                                    const FDAGWR_TRAITS::Sparse_Matrix &R) 
const
{
    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);
    functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_crossed_t,W,m_number_threads),X_crossed,m_number_threads);

    //performing integration and factorization
    return Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >( this->fm_integration(integrand) + _R_ ); 
}



template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                                    const functional_matrix<INPUT,OUTPUT> &X_t,
                                    const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                                    const functional_matrix<INPUT,OUTPUT> &X,
                                    const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                    const FDAGWR_TRAITS::Sparse_Matrix &R)
const
{   
    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);
    functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(fm_prod(base_t,X_t),W,m_number_threads),X,m_number_threads),base);

    //performing integration and factorization
    return Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >( this->fm_integration(integrand) + _R_ ); 
}



/////////////////////
///// OPERATORS /////
/////////////////////

template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > operator_;
    operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,base_lhs,X_lhs,X_rhs,base_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(fm_prod(base_lhs,X_lhs),W[i],m_number_threads),X_rhs,m_number_threads),base_rhs);

        //performing integration and multiplication with the penalty (inverse factorized)
        operator_[i] = penalty[i].solve(this->fm_integration(integrand)); 
    }

    return operator_;
}



template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > operator_;
    operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,base_lhs,X_lhs,base_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(base_lhs,X_lhs),W[i],m_number_threads),base_rhs);

        //performing integration and multiplication with the penalty (inverse factorized)
        operator_[i] = penalty[i].solve(this->fm_integration(integrand)); 
    }

    return operator_;
}



template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > operator_;
    operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,X_lhs,X_rhs,base_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(X_lhs,W[i],m_number_threads),X_rhs,m_number_threads),base_rhs);

        //performing integration and multiplication with the penalty (inverse factorized)
        operator_[i] = penalty[i].solve(this->fm_integration(integrand)); 
    }

    return operator_;
}




template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                     const functional_matrix_sparse<INPUT,OUTPUT> &base_rhs,
                                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > operator_;
    operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,X_lhs,base_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_lhs,W[i],m_number_threads),base_rhs);

        //performing integration and multiplication with the penalty (inverse factorized)
        operator_[i] = penalty[i].solve(this->fm_integration(integrand)); 
    }

    return operator_;
}



template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > operator_;
    operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,X_lhs,X_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_lhs,W[i],m_number_threads),X_rhs,m_number_threads);

        //performing integration and multiplication with the penalty (inverse factorized)
        operator_[i] = penalty[i].solve(this->fm_integration(integrand)); 
    }

    return operator_;
}


template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) 
const
{
    //the vector contains factorization of the matrix
    std::vector< FDAGWR_TRAITS::Dense_Matrix > operator_;
    operator_.resize(W.size());

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,base_lhs,X_lhs,X_rhs,W,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < W.size(); ++i)
    {       
        //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(base_lhs,X_lhs),W[i],m_number_threads),X_rhs,m_number_threads);

        //performing integration and multiplication with the penalty (inverse factorized)
        operator_[i] = penalty[i].solve(this->fm_integration(integrand)); 
    }

    return operator_;
}


template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
FDAGWR_TRAITS::Dense_Matrix
fgwr<INPUT,OUTPUT>::compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                     const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                     const Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > &penalty) 
const
{
    //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
    functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_lhs,W,m_number_threads),X_rhs,m_number_threads);

    //performing integration and multiplication with the penalty (inverse factorized)
    return penalty.solve(this->fm_integration(integrand)); 
}


template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
FDAGWR_TRAITS::Dense_Matrix
fgwr<INPUT,OUTPUT>::compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                                     const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                                     const Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > &penalty) 
const
{
    //dimension: L_lhs x L_rhs, where L is the number of basis (the left basis is transpost)
    functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(base_lhs,X_lhs),W,m_number_threads),X_rhs,m_number_threads);

    //performing integration and multiplication with the penalty (inverse factorized)
    return penalty.solve(this->fm_integration(integrand)); 
}


/////////////////////////////
//// FUNCTIONAL OPERATOR ////
/////////////////////////////

template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
functional_matrix<INPUT,OUTPUT> 
fgwr<INPUT,OUTPUT>::compute_functional_operator(const functional_matrix<INPUT,OUTPUT> &X,
                                                const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                                const std::vector< FDAGWR_TRAITS::Dense_Matrix > &operator_) 
const
{
    //result
    functional_matrix<INPUT,OUTPUT> func_oper(m_n,operator_[0].cols());

    functional_matrix<INPUT,OUTPUT> x_times_base = fm_prod(X,base);

#ifdef _OPENMP
#pragma omp parallel for shared(func_oper,x_times_base,operator_,m_n,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < m_n; ++i){
        //trnasforming the scalar matrix into a functional one, with constant functions
        std::vector< FUNC_OBJ<INPUT,OUTPUT> > row_i_v(x_times_base.row(i).cbegin(),x_times_base.row(i).cend());
        functional_matrix<INPUT,OUTPUT> row_i(row_i_v,1,base.cols());
        functional_matrix<INPUT,OUTPUT> row_i_prod = fm_prod(row_i,operator_[i],m_number_threads);
        func_oper.row_sub(row_i_prod.as_vector(),i);
    }

    return func_oper;
}
*/



/*
//////////////////////
/////// WRAP B ///////
//////////////////////

//for stationary covariate
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::wrap_b(const FDAGWR_TRAITS::Dense_Matrix& b,
                           const std::vector<std::size_t>& L_j,
                           std::size_t q)
const
{
    //input coherency
    assert((L_j.size() == q) && (b.cols() == 1) && (b.rows() == std::reduce(L_j.cbegin(),L_j.cend(),static_cast<std::size_t>(0))));
    //container
    std::vector< FDAGWR_TRAITS::Dense_Matrix > B;
    B.reserve(q);
    for(std::size_t j = 0; j < q; ++j)
    {
        //for each stationary covariates
        std::size_t start_idx = std::reduce(L_j.cbegin(),std::next(L_j.cbegin(),j),static_cast<std::size_t>(0));
        //taking the right coefficients of the basis expansion
        FDAGWR_TRAITS::Dense_Matrix B_j = b.block(start_idx,0,L_j[j],1);
        B.push_back(B_j);
    }

    return B;
}


//for non stationary covariates
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > >
fgwr<INPUT,OUTPUT>::wrap_b(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
                           const std::vector<std::size_t>& L_j,
                           std::size_t q) 
const
{
    //input coherency
    assert((b.size() == this->n()) && (L_j.size() == q));
    for(std::size_t i = 0; i < this->n(); ++i){     assert((b[i].cols() == 1) && (b[i].rows() == std::reduce(L_j.cbegin(),L_j.cend(),static_cast<std::size_t>(0))));}
    //container
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> B;    
    B.reserve(q);
    for(std::size_t j = 0; j < q; ++j)
    {
        //for each event-dependent covariates
        std::size_t start_idx = std::reduce(L_j.cbegin(),std::next(L_j.cbegin(),j),static_cast<std::size_t>(0));
        std::vector< FDAGWR_TRAITS::Dense_Matrix > B_j;
        B_j.reserve(this->n());
        //for all the units
        for(std::size_t i = 0; i < this->n(); ++i){
            //taking the right coefficients of the basis expansion
            FDAGWR_TRAITS::Dense_Matrix B_j_i = b[i].block(start_idx,0,L_j[j],1);
            B_j.push_back(B_j_i);}
        B.push_back(B_j);
    }

    return B;
}
*/




//////////////////////
///// EVAL BETAS /////
//////////////////////

//stationary betas
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< std::vector< OUTPUT >>
fgwr<INPUT,OUTPUT>::eval_betas(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& B,
                               const functional_matrix_sparse<INPUT,OUTPUT>& basis_B,
                               const std::vector<std::size_t>& L_j,
                               std::size_t q,
                               const std::vector< INPUT >& abscissas) 
const
{
    
    //input coherency
    assert((B.size() == q) && (L_j.size() == q) && (basis_B.rows() == q) && (basis_B.cols() == std::reduce(L_j.cbegin(),L_j.cend(),static_cast<std::size_t>(0))));
    for(std::size_t j = 0; j < q; ++j){     assert((B[j].rows() == L_j[j]) && (B[j].cols() == 1));}
    //container
    std::vector< std::vector< OUTPUT >> beta;
    beta.reserve(q);
    //aliases
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    for(std::size_t j = 0; j < q; ++j)
    {
        //retrieving the basis
        std::vector< F_OBJ > basis_j_v;
        basis_j_v.reserve(B[j].rows());
        std::size_t start_idx = std::reduce(L_j.cbegin(),std::next(L_j.cbegin(),j),static_cast<std::size_t>(0));
        std::size_t end_idx = start_idx + L_j[j];
        for(std::size_t k = start_idx; k < end_idx; ++k){   basis_j_v.push_back(basis_B(j,k));}
        functional_matrix<INPUT,OUTPUT> basis_j(basis_j_v,1,B[j].rows());

        //compute the beta
        FUNC_OBJ<INPUT,OUTPUT> beta_j = fm_prod<INPUT,OUTPUT>(basis_j,B[j],this->number_threads())(0,0);
        //eval the beta
        std::vector< OUTPUT > beta_j_ev; 
        beta_j_ev.resize(abscissas.size());
        std::transform(abscissas.cbegin(),abscissas.cend(),beta_j_ev.begin(),[&beta_j](F_OBJ_INPUT x){return beta_j(x);});
        beta.push_back(beta_j_ev);
    }

    return beta;
}


//non-stationary betas
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< std::vector< std::vector< OUTPUT >>>
fgwr<INPUT,OUTPUT>::eval_betas(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& B,
                               const functional_matrix_sparse<INPUT,OUTPUT>& basis_B,
                               const std::vector<std::size_t>& L_j,
                               std::size_t q,
                               const std::vector< INPUT >& abscissas)
const
{
    //input coherency
    assert((B.size() == q) && (L_j.size() == q) && (basis_B.rows() == q) && (basis_B.cols() == std::reduce(L_j.cbegin(),L_j.cend(),static_cast<std::size_t>(0))));
    for(std::size_t j = 0; j < q; ++j){  
        assert(B[j].size() == this->n());   
        for(std::size_t i = 0; i < this->n(); ++i){     assert((B[j][i].rows() == L_j[j]) && (B[j][i].cols() == 1));}}
        
    //container
    std::vector< std::vector< std::vector< OUTPUT >>> beta;
    beta.reserve(q);
    //aliases
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;
    

    for(std::size_t j = 0; j < q; ++j)
    {
        //retrieving the basis
        std::vector< F_OBJ > basis_j_v;
        basis_j_v.reserve(L_j[j]);
        std::size_t start_idx = std::reduce(L_j.cbegin(),std::next(L_j.cbegin(),j),static_cast<std::size_t>(0));
        std::size_t end_idx = start_idx + L_j[j];
        for(std::size_t k = start_idx; k < end_idx; ++k){   basis_j_v.push_back(basis_B(j,k));}
        functional_matrix<INPUT,OUTPUT> basis_j(basis_j_v,1,L_j[j]);

        //evaluating the betas in every unit
        std::vector< std::vector<OUTPUT> > beta_j_ev;
        beta_j_ev.reserve(this->n());
        for(std::size_t i = 0; i < this->n(); ++i)
        {
            //compute the beta j-th for unit i-th
            FUNC_OBJ<INPUT,OUTPUT> beta_j_i = fm_prod<INPUT,OUTPUT>(basis_j,B[j][i],this->number_threads())(0,0);
            //eval the beta
            std::vector< OUTPUT > beta_j_i_ev; 
            beta_j_i_ev.resize(abscissas.size());
            std::transform(abscissas.cbegin(),abscissas.cend(),beta_j_i_ev.begin(),[&beta_j_i](F_OBJ_INPUT x){return beta_j_i(x);});
            beta_j_ev.push_back(beta_j_i_ev);
        }

        beta.push_back(beta_j_ev);
    }

    return beta;
}