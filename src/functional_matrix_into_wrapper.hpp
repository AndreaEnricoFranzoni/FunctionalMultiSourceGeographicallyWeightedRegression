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


#ifndef FUNCTIONAL_MATRIX_INTO_WRAPPER_HPP
#define FUNCTIONAL_MATRIX_INTO_WRAPPER_HPP

#include "functional_matrix_storing_type.hpp"
#include "functional_matrix.hpp"
#include "traits_fdagwr.hpp"


/*!
* @brief Function to wrap a functional data object functional_data into a column (row) vector intended as a functional matrix object functional_matrix
*/
template< typename INPUT = double, typename OUTPUT = double, class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>) && fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
inline
functional_matrix<INPUT,OUTPUT>
wrap_into_fm(const functional_data<domain_type,basis_type> &fd,
             int number_threads,
             bool as_column = true)
{
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    std::vector< F_OBJ > f_i;
    f_i.resize(fd.n());

#ifdef _OPENMP
#pragma omp parallel for shared(fd) num_threads(number_threads)
    for(std::size_t i = 0; i < fd.n(); ++i)
    {
        f_i[i] = [i,&fd](F_OBJ_INPUT x){return fd.eval(x,i);};
    }
#endif

    if(as_column == true)
    {
        functional_matrix<INPUT,OUTPUT> fm(std::move(f_i),fd.n(),1);
        return fm;
    }
    else
    {
        functional_matrix<INPUT,OUTPUT> fm(std::move(f_i),1,fd.n());
        return fm;   
    }
}

#endif  /*FUNCTIONAL_MATRIX_INTO_WRAPPER_HPP*/