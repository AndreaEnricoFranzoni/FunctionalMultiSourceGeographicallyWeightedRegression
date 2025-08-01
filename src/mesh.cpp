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

#include "mesh.hpp"
#include <algorithm>
#include <numeric>

/*!
* @file mesh.cpp
* @brief Contains the implementation of the class for an unidimensioanl mesh. 
* @author Luca Formaggia
* @note Taken from pacs-examples, folder of repository PACS Course (https://github.com/pacs-course), Advanced Programming for Scientific Computing, Politecnico di Milano
*/

namespace Geometry
{
Mesh1D::Mesh1D(Domain1D const &d, unsigned int const &n) : myDomain(d)
{
  Uniform g(d, n);
  myNodes = g();
};

double
Mesh1D::hmax() const
{
  std::vector<double> tmp(myNodes.size());
  std::adjacent_difference(myNodes.begin(), myNodes.end(), tmp.begin());
  return *std::max_element(++tmp.begin(), tmp.end());
}

double
Mesh1D::hmin() const
{
  std::vector<double> tmp(myNodes.size());
  std::adjacent_difference(myNodes.begin(), myNodes.end(), tmp.begin());
  return *std::min_element(++tmp.begin(), tmp.end());
}

void
Mesh1D::reset(OneDMeshGenerator const &mg)
{
  myDomain = mg.getDomain();
  myNodes = mg();
}

} // namespace Geometry