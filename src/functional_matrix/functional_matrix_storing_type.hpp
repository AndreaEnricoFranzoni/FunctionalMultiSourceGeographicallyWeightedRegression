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


#ifndef FUNCTIONAL_MATRIX_STORING_TYPE_HPP
#define FUNCTIONAL_MATRIX_STORING_TYPE_HPP


#include <functional>
#include <type_traits>
#include <utility>
#include <concepts>



/*!
* @file functional_matrix_storing_type.hpp
* @brief Contains the specific univariate 1D domain std::function object contained in a functional matrix declaration
* @author Andrea Enrico Franzoni
*/


/*!
* @brief univariate 1D domain  std::function object stored in a functional_matrix
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
*/
template< typename INPUT, typename OUTPUT >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using FUNC_OBJ = std::function< OUTPUT (INPUT const &) >;

#endif  /*FUNCTIONAL_MATRIX_STORING_TYPE_HPP*/