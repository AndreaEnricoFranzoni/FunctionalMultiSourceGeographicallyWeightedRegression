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

#ifndef FDAGWR_BASIS_FACTORY_PROXY_HPP
#define FDAGWR_BASIS_FACTORY_PROXY_HPP

#include "basis_factory.hpp"


/*!
* @file basis_factory_proxy.hpp
* @brief Contains the proxy for the factory for the basis
* @author Andrea Enrico Franzoni
*/

namespace {
  using basis_factory::basisProxy;

  basisProxy<bsplines_basis<FDAGWR_TRAITS::basis_geometry>> basisBSPLINES(FDAGWR_BASIS_TYPES::_bsplines_);  ///< Registration of bspline basis
  basisProxy<constant_basis<FDAGWR_TRAITS::basis_geometry>> basisCONSTANT(FDAGWR_BASIS_TYPES::_constant_);  ///< Registration of constant basis

}

#endif  /*FDAGWR_BASIS_FACTORY_PROXY_HPP*/