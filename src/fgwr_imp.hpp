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


template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> >
fgwr<INPUT,OUTPUT>::compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                    const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                                    const functional_matrix<INPUT,OUTPUT> &X,
                                    const functional_matrix<INPUT,OUTPUT> &X_t,
                                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                    const FDAGWR_TRAITS::Dense_Matrix &R)
const
{
    //the vector contains factorization of the matrix
    std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > penalty;
    penalty.resize(m_n);

    FDAGWR_TRAITS::Dense_Matrix _R_ = FDAGWR_TRAITS::Dense_Matrix(R);   //necessary to compute the sum later

#ifdef _OPENMP
#pragma omp parallel for shared(penalty,base,base_t,X,X_t,W,_R_,m_n,m_number_threads) num_threads(m_number_threads)
    for(std::size_t i = 0; i < m_n; ++i)
    {
        //dimension: L x L, where L is the number of basis
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(fm_prod(base_t,X_t),W[i],m_number_threads),X,m_number_threads),base);

        //performing integration and factorization
        FDAGWR_TRAITS::Dense_Matrix _j_tilde_tilde_i_ = this->fm_integration(integrand).eval();
        FDAGWR_TRAITS::Dense_Matrix inverse_penalty = _j_tilde_tilde_i_ + _R_;
        penalty[i] = Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix>(inverse_penalty);    //.eval() is needed to evaluate the lazy expression of ETs  
        // penalty[i].solve(M) equivale a fare elemento penalty[i], che è una matrice inversa, times M
    }
#endif
    
    return penalty;
}




