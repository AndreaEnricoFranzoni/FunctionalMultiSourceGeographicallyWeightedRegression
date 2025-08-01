// Copyright (c) 2024 Andrea Enrico Franzoni (andreaenrico.franzoni@gmail.com)
//
// This file is part of PPCKO
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of PPCKO and associated documentation files (the PPCKO software), to deal
// PPCKO without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of PPCKO, and to permit persons to whom PPCKO is
// furnished to do so, subject to the following conditions:
//
// PPCKO IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH PPCKO OR THE USE OR OTHER DEALINGS IN
// PPCKO.

#include "meshGenerators.hpp"
#include <algorithm>
#include <stdexcept>

/*!
* @file meshGenerators.cpp
* @brief Contains implementation of the class for generating an unidimensional mesh. Little modification: retained only the part for an uniform mesh.
* @author Luca Formaggia
* @note Taken from pacs-examples, folder of repository PACS Course (https://github.com/pacs-course), Advanced Programming for Scientific Computing, Politecnico di Milano
*/

namespace Geometry
{
MeshNodes
Uniform::operator()() const
{
  auto const &n = this->M_num_elements;
  auto const &a = this->M_domain.left();
  auto const &b = this->M_domain.right();
  if(n == 0)
    throw std::runtime_error("At least two elements");
  MeshNodes    mesh(n + 1);
  double const h = (b - a) / static_cast<double>(n);
#pragma omp parallel for
  for(auto i = 0u; i < n; ++i)
    mesh[i] = a + h * i;
  mesh[n] = b;
  return mesh;
  
}

} // namespace Geometry
