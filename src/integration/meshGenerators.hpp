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

/*!
* @namespace Geometry
* @brief Contains domain geometry definition
*/
namespace Geometry
{
/*!
* @brief Container for the nodes
*/
using MeshNodes = std::vector<double>;
//! General interface

/*!
* @class OneDMeshGenerator
* @brief Contains the generator for a 1D mesh
*/
class OneDMeshGenerator
{
public:
  /*!
  * @brief Constructor
  * @param d a 1D domain
  */
  OneDMeshGenerator(Geometry::Domain1D const &d) : M_domain{d} {}

  /*!
  * @brief Call operator
  * @return a mesh
  * @note virtual method
  */
  virtual MeshNodes operator()() const = 0;

  /*!
  * @brief Getter for the domain
  * @return the 1D domain
  */
  Domain1D
  getDomain() const
  {
    return M_domain;
  }

  /*!
  * @brief Virtual destructor
  */
  virtual ~OneDMeshGenerator() = default;

protected:
  /*!1D domain*/
  Geometry::Domain1D M_domain;
};


/*!
* @class Uniform
* @brief Uniform mesh
*/
class Uniform : public OneDMeshGenerator
{
public:
  /*! 
  * @brief Constructor
  * @param domain A 1D domain
  * @param num_elements Number of elements
  */
  Uniform(Geometry::Domain1D const &domain, unsigned int num_elements)
    : OneDMeshGenerator(domain), M_num_elements(num_elements)
  {}

  /*!
  * @brief Call operator
  * @return a mesh of equally spaced nodes
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
  /*!Number of elements in the mesh*/
  std::size_t M_num_elements;
};

} // namespace Geometry

#endif  /*HH_GENERATOR_HH*/
