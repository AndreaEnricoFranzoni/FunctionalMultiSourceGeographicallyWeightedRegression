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


#ifndef FDAGWR_BASIS_FACTORY_HPP
#define FDAGWR_BASIS_FACTORY_HPP

#include "traits_fdagwr.hpp"
#include "concepts_fdagwr.hpp"
#include "basis_include.hpp"


template<typename domain_type, class... Args>
    requires fdagwr_concepts::as_interval<domain_type>
std::unique_ptr<basis_base_class> baseFactory(const std::string& basis_id, Args&&... args)
{
switch(basis_id)
case FDAGWR_BASIS_TYPES::_bsplines_ :   return std::make_unique<bsplines_basis<domain_type>>(std::forward<Args>(args)...);
case FDAGWR_BASIS_TYPES::_constant_ :   return std::make_unique<constant_basis<domain_type>>(std::forward<Args>(args)...); 
default                             :   return std::unique_ptr<basis_base_class>();
}

#endif  /*FDAGWR_BASIS_FACTORY_HPP*/