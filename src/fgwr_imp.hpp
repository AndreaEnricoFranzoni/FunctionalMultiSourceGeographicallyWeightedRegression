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
std::vector< FDAGWR_TRAITS::Dense_Matrix >
fgwr<INPUT,OUTPUT>::compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                    const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                                    const functional_matrix<INPUT,OUTPUT> &X,
                                    const functional_matrix<INPUT,OUTPUT> &X_t,
                                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                                    const FDAGWR_TRAITS::Dense_Matrix &R)
const
{
    std::vector< FDAGWR_TRAITS::Dense_Matrix > penalty;
    penalty.resize(m_n);

    //qua da usare OMP
    for(std::size_t i = 0; i < m_n; ++i)
    {
        //dimension: L x L, where L is the number of basis
        functional_matrix<INPUT,OUTPUT> integrand = fm_prod(fm_prod(fm_prod(fm_prod(base_t,X_t),W[i]),X),base);
        std::vector<OUTPUT> result_integrand;
        result_integrand.resize(integrand.size());
        std::transform(cbegin(integrand),
                       cend(integrand),
                       result_integrand.begin(),
                       [this](const FUNC_OBJ<INPUT,OUTPUT> &f){this->m_integrating.integrate(f);});

        penalty[i] = (Eigen::Map< FDAGWR_TRAITS::Dense_Matrix >(result_integrand.data(),integrand.rows(),integrand.cols()) + R).inverse();
        // = inv_pen.inverse();
    }
    
    return penalty;
}