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


/*!
* @namespace Geometry
* @brief Contains domain geometry definition
*/
namespace Geometry
{

/*!
* @class Mesh1D
* @brief Conat Defines a 1D domain.
*/
class Mesh1D
{
public:

  /*!
  * @brief Default constructor
  */
  Mesh1D() = default;

  /*!
  * @brief Constructor for an equaly spaced mesh
  * @param d 1D domain
  * @param Number of intervals (not nodes!)
  */
  Mesh1D(Domain1D const &d, unsigned int const &n) : myDomain(d)
  {
    Uniform g(d, n);
    myNodes = g();
  }

  /*!
  * @brief Constructor for an variably spaced mesh
  * @param gf the policy for generating mesh
  */
  Mesh1D(Geometry::OneDMeshGenerator const &gf)
    : myDomain{gf.getDomain()}, myNodes{gf()} {};

  /*!
  * @brief Generate mesh (it will destroy old mesh)
  * @param mg a mesh generator
  */
  void reset(OneDMeshGenerator const &mg)
  {
    myDomain = mg.getDomain();
    myNodes = mg();
  }

  /*!
  * @brief Number of nodes
  * @return the number of nodes of the mesh
  */
  unsigned int
  numNodes() const
  {
    return myNodes.size();
  }

  /*!
  * @brief The i-th node.
  * @param i which node
  * @return the i-th node
  */
  double
  operator[](int i) const
  {
    return myNodes[i];
  }

  /*!
  * @brief The nodes
  * @return the nodes
  */
  std::vector<double>
  nodes() const
  {
    return myNodes;
  }

  /*!
  * @brief .begin() iterator for the mesh
  * @return the begin() iterator of the mesh
  * @note To use the mesh in range based for loop, non-constant version
  */
  std::vector<double>::iterator
  begin()
  {
    return myNodes.begin();
  }

  /*!
  * @brief .cbegin() iterator for the mesh
  * @return the cbegin() iterator of the mesh
  * @note To use the mesh in range based for loop, constant version
  */
  std::vector<double>::const_iterator
  cbegin() const
  {
    return myNodes.cbegin();
  }

  /*!
  * @brief .end() iterator for the mesh
  * @return the end() iterator of the mesh
  * @note To use the mesh in range based for loop, non-constant version
  */
  std::vector<double>::iterator
  end()
  {
    return myNodes.end();
  }

  /*!
  * @brief .cend() iterator for the mesh
  * @return the cend() iterator of the mesh
  * @note To use the mesh in range based for loop, constant version
  */
  std::vector<double>::const_iterator
  cend() const
  {
    return myNodes.cend();
  }

  /*! 
  * @brief Getter for the domain
  * @return the domain of the mesh
  */
  Domain1D
  domain() const
  {
    return myDomain;
  }

  /*!
  * @brief The minimum length of the intervals in the mesh
  * @return the minimum mesh size
  */
  double hmin() const
  {
    std::vector<double> tmp(myNodes.size());
    std::adjacent_difference(myNodes.begin(), myNodes.end(), tmp.begin());
    return *std::max_element(++tmp.begin(), tmp.end());
  }

  /*!
  * @brief The maximum length of the intervals in the mesh
  * @return the maximum mesh size
  */
  double hmax() const
  {
    std::vector<double> tmp(myNodes.size());
    std::adjacent_difference(myNodes.begin(), myNodes.end(), tmp.begin());
    return *std::min_element(++tmp.begin(), tmp.end());
  }

private:
  /*!Domain*/
  Domain1D            myDomain;
  /*!Nodes*/
  std::vector<double> myNodes;
};

} // namespace Geometry
#endif