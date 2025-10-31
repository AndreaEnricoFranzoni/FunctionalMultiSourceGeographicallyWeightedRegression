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

#ifndef EXAMPLES_SRC_QUADRATURERULE_BASEVERSION_QUADRATURERULETRAITS_HPP_
#define EXAMPLES_SRC_QUADRATURERULE_BASEVERSION_QUADRATURERULETRAITS_HPP_

#include "CloningUtilities.hpp"
#include "../functional_matrix/functional_matrix_storing_type.hpp"
#include "../utility/traits_fdagwr.hpp"
#include <functional>
#include <memory>

/*!
* @file QuadratureRuleTraits.hpp
* @brief Contains the traits for the integration. 
* @author Luca Formaggia
* @note Taken from pacs-examples, folder of repository PACS Course (https://github.com/pacs-course), Advanced Programming for Scientific Computing, Politecnico di Milano
*/


/*!
* @namespace apsc::NumericalIntegration
* @brief Contains the integration features
*/
namespace apsc::NumericalIntegration
{
/*! 
* @brief The type the integrand 
*/
using FunPoint = FUNC_OBJ<double,double>;

} // namespace apsc::NumericalIntegration

#endif /* EXAMPLES_SRC_QUADRATURERULE_BASEVERSION_QUADRATURERULETRAITS_HPP_ */
