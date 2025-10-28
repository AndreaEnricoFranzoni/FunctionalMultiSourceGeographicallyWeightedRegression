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


#ifndef FDAGWR_BASIS_INCLUDE_HPP
#define FDAGWR_BASIS_INCLUDE_HPP


#include "basis_bspline.hpp"
#include "basis_constant.hpp"


/*!
* @file basis_include.hpp
* @brief Contains the includes for the basis
* @author Andrea Enrico Franzoni
*/


/*!
* @struct FDAGWR_BASIS_TYPES
* @brief Struct to decipt which are the basis types implemented
* @note It matches the includes above
*/
struct FDAGWR_BASIS_TYPES
{
  inline static constexpr std::size_t _number_implemented_basis_types_ = static_cast<std::size_t>(2);      ///< Number of concrete basis objects implemented

  inline static constexpr std::string_view _bsplines_ = "bsplines";                                             ///< Name of bspline basis

  inline static constexpr std::string_view _constant_ = "constant";                                             ///< Name of constant basis

  inline static constexpr std::array<std::string_view,FDAGWR_BASIS_TYPES::_number_implemented_basis_types_> _implemented_basis_{FDAGWR_BASIS_TYPES::_bsplines_,
                                                                                                                                FDAGWR_BASIS_TYPES::_constant_};  ///< Array with the concrete basis implemented
};

#endif  /*FDAGWR_BASIS_INCLUDE_HPP*/