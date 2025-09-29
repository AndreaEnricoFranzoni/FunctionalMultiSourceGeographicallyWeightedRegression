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

#ifndef HH_GENERATOR_HH
#define HH_GENERATOR_HH
#include "domain.hpp"
#include <functional>
#include <stdexcept>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif


/*!
* @file meshGenerators.hpp
* @brief Contains the class for generating an unidimensional mesh. Little modification: retained only the part for an uniform mesh.
* @author Luca Formaggia
* @note Taken from pacs-examples, folder of repository PACS Course (https://github.com/pacs-course), Advanced Programming for Scientific Computing, Politecnico di Milano
*/

namespace Geometry
{
using MeshNodes = std::vector<double>;
//! General interface
class OneDMeshGenerator
{
public:
  OneDMeshGenerator(Geometry::Domain1D const &d) : M_domain{d} {}
  virtual MeshNodes operator()() const = 0;
  Domain1D
  getDomain() const
  {
    return M_domain;
  }
  virtual ~OneDMeshGenerator() = default;

protected:
  Geometry::Domain1D M_domain;
};
/*! \defgroup meshers Functors which generates a 1D mesh.
  @{ */
//! Uniform mesh
class Uniform : public OneDMeshGenerator
{
public:
  /*! constructor
@param domain A 1D domain
@param b num_elements Number of elements
  */
  Uniform(Geometry::Domain1D const &domain, unsigned int num_elements)
    : OneDMeshGenerator(domain), M_num_elements(num_elements)
  {}
  //! Call operator
  /*!
    @param meshNodes a mesh of nodes
  */
  MeshNodes operator()() const override
  {
    auto const &n = this->M_num_elements;
    auto const &a = this->M_domain.left();
    auto const &b = this->M_domain.right();
    if(n == 0)
      throw std::runtime_error("At least two elements");
    MeshNodes    mesh(n + 1);
    double const h = (b - a) / static_cast<double>(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(auto i = 0u; i < n; ++i)
      mesh[i] = a + h * i;
    
    mesh[n] = b;
    return mesh;
  }

private:
  std::size_t M_num_elements;
};
/*! @}*/
} // namespace Geometry
#endif
