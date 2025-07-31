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


#ifndef FDAGWR_PENALTIES_COMP_HPP
#define FDAGWR_PENALTIES_COMP_HPP

#include "traits_fdagwr.hpp"
#include "basis_bspline_systems.hpp"
#include "penalization_matrix_penalties_policies.hpp"


template <class PENALTY_ORDER_policy> 
class penalty_computation
{
public:
  
  FDAGWR_TRAITS::Sparse_Matrix 
  operator()
  (const basis_systems< FDAGWR_TRAITS::basis_geometry, bsplines_basis > &bs, std::size_t system_number) 
  const 
  {   
    return penalty_computing(bs, system_number);}
  
private:
  /*!Policy to correctly compute the penalization*/
  PENALTY_ORDER_policy penalty_computing;
};

#endif  /*FDAGWR_PENALTIES_COMP_HPP*/