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

#ifndef _HH_MESH_HH
#define _HH_MESH_HH
#include "domain.hpp"
#include "meshGenerators.hpp"
#include <functional>
#include <vector>
#include <algorithm>
#include <numeric>

/*!
* @file mesh.hpp
* @brief Contains the class for an unidimensioanl mesh. 
* @author Luca Formaggia
* @note Taken from pacs-examples, folder of repository PACS Course (https://github.com/pacs-course), Advanced Programming for Scientific Computing, Politecnico di Milano
*/

namespace Geometry
{
class Mesh1D
{
public:
  //! Default constructor is defaulted.
  Mesh1D() = default;
  //! Constructor for an equaly spaced mesh
  /*!
    \param d  A domain
    \param n  Number of intervals (not nodes!)
  */
  Mesh1D(Domain1D const &d, unsigned int const &n) : myDomain(d)
  {
    Uniform g(d, n);
    myNodes = g();
  }
  //! Constructor for an variably spaced mesh
  /*!
    \param gf the policy for generating mesh
  */
  Mesh1D(Geometry::OneDMeshGenerator const &gf)
    : myDomain{gf.getDomain()}, myNodes{gf()} {};
  //! Generate mesh (it will destroy old mesh)
  /*!
    @param mg a mesh generator
   */
  void reset(OneDMeshGenerator const &mg)
  {
    myDomain = mg.getDomain();
    myNodes = mg();
  }

  //! Number of nodes.
  unsigned int
  numNodes() const
  {
    return myNodes.size();
  }
  //! The i-th node.
  double
  operator[](int i) const
  {
    return myNodes[i];
  }
  //! The nodes.
  std::vector<double>
  nodes() const
    {
      return myNodes;
    }
  //! To use the mesh in range based for loop I need begin()
  std::vector<double>::iterator
  begin()
  {
    return myNodes.begin();
  }
  std::vector<double>::const_iterator
  cbegin() const
  {
    return myNodes.cbegin();
  }
  //! To use the mesh in range based for loop I need end()
  std::vector<double>::iterator
  end()
  {
    return myNodes.end();
  }
  std::vector<double>::const_iterator
  cend() const
  {
    return myNodes.cend();
  }
  //! I return a copy of the DOmain1D.
  /*!
    In case it is needed.
  */
  Domain1D
  domain() const
  {
    return myDomain;
  }
  //! The max mesh size.
  double hmin() const
  {
    std::vector<double> tmp(myNodes.size());
    std::adjacent_difference(myNodes.begin(), myNodes.end(), tmp.begin());
    return *std::max_element(++tmp.begin(), tmp.end());
  }
  //! The max mesh size.
  double hmax() const
  {
    std::vector<double> tmp(myNodes.size());
    std::adjacent_difference(myNodes.begin(), myNodes.end(), tmp.begin());
    return *std::min_element(++tmp.begin(), tmp.end());
  }

private:
  Domain1D            myDomain;
  std::vector<double> myNodes;
};

} // namespace Geometry
#endif