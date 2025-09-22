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


////////////////////////
/////  PENALTY    /////
////////////////////////

/*!
* @brief Compute all the [J_2_tilde_i + R]^(-1): 
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> >
fgwr<INPUT,OUTPUT>::compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                    const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                                    const functional_matrix<INPUT,OUTPUT> &X,
                                    const functional_matrix<INPUT,OUTPUT> &X_t,
                                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                    const FDAGWR_TRAITS::Sparse_Matrix &R)
const
{
    //the vector contains factorization of the matrix
    std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > penalty;
    penalty.resize(m_n);

    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);   //necessary to compute the sum later

#ifdef _OPENMP
#pragma omp parallel for shared(penalty,base,base_t,X,X_t,W,_R_,m_n,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < m_n; ++i)
    {
        //dimension: L x L, where L is the number of basis
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(fm_prod(base_t,X_t),W[i],m_number_threads),X,m_number_threads),base);

        //performing integration and factorization
        penalty[i] = Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >( this->fm_integration(integrand) + _R_ );    
        // penalty[i].solve(M) equivale a fare elemento penalty[i], che è una matrice inversa, times M
    }
    
    return penalty;
}



/*!
* @brief Compute [J_tilde_i + R]^(-1)
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > >
fgwr<INPUT,OUTPUT>::compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed,
                                    const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                    const FDAGWR_TRAITS::Sparse_Matrix &R) 
const
{
    //the vector contains factorization of the matrix
    std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > penalty;
    penalty.resize(m_n);

    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);   //necessary to compute the sum later

#ifdef _OPENMP
#pragma omp parallel for shared(penalty,X_crossed_t,X_crossed,W,_R_,m_n,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < m_n; ++i)
    {
        //dimension: L x L, where L is the number of basis
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_crossed_t,W[i],m_number_threads),X_crossed,m_number_threads);

        //performing integration and factorization
        penalty[i] = Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >( this->fm_integration(integrand) + _R_ );    
        // penalty[i].solve(M) equivale a fare elemento penalty[i], che è una matrice inversa, times M
    }
    
    return penalty;
}


/*!
* @brief Compute [J + Rc]^(-1)
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed,
                                    const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                                    const functional_matrix_diagonal<INPUT,OUTPUT> &W,
                                    const FDAGWR_TRAITS::Sparse_Matrix &R) 
const
{
    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);
    functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(X_crossed_t,W,m_number_threads),X_crossed,m_number_threads);

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
    operator_.resize(m_n);

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,base_lhs,X_lhs,X_rhs,base_rhs,W,m_n,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < m_n; ++i)
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
    operator_.resize(m_n);

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,base_lhs,X_lhs,base_rhs,W,m_n,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < m_n; ++i)
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
    operator_.resize(m_n);

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,X_lhs,X_rhs,base_rhs,W,m_n,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < m_n; ++i)
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
    operator_.resize(m_n);

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,X_lhs,base_rhs,W,m_n,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < m_n; ++i)
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
    operator_.resize(m_n);

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,X_lhs,X_rhs,W,m_n,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < m_n; ++i)
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
    operator_.resize(m_n);

#ifdef _OPENMP
#pragma omp parallel for shared(operator_,penalty,base_lhs,X_lhs,X_rhs,W,m_n,m_number_threads) num_threads(m_number_threads)
#endif
    for(std::size_t i = 0; i < m_n; ++i)
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
        std::vector< F_OBJ<INPUT,OUTPUT> > row_i_v(x_times_base.row(i).cbegin(),x_times_base.row(i).cend());
        functional_matrix<INPUT,OUTPUT> row_i(row_i_v,1,base.cols());
        functional_matrix<INPUT,OUTPUT> row_i_prod = fm_prod(row_i,operator_[i],m_number_threads);
        func_oper.row_sub(row_i_prod.as_vector(),i);
    }

    return func_oper;
}