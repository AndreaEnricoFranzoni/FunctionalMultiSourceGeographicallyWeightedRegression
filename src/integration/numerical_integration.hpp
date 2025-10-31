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

#ifndef NUMERICAL_INTEGRATION_HPP
#define NUMERICAL_INTEGRATION_HPP

#include "mesh.hpp"
#include "QuadratureRuleBase.hpp"
#ifdef PALALLELCPP
#include <execution>
#endif


/*!
* @file numerical_integration.hpp
* @brief Contains the class for composite integration. 
* @author Luca Formaggia
* @note Taken from pacs-examples, folder of repository PACS Course (https://github.com/pacs-course), Advanced Programming for Scientific Computing, Politecnico di Milano
*/


/*!
* @namespace apsc::NumericalIntegration
* @brief Contains the integration features
*/
namespace apsc::NumericalIntegration
{
/*!
* @brief Uplodaing the namespace for the domain geometry
*/
using namespace Geometry;

/*!
* @brief Quadrature
* @brief Class for composite integration.
*        It is implemented using the following design:
*        - Composition with a QuadraturRule is implemented via a
*        unique_ptr<>. A Quadrature object thus owns a polymorphic object
*        of type QuadratureRule. This allows to assign the quadrature rule run-time.
*        - A 1D mesh is simply aggregated as an object. So a copy is
*        made. We however support move semantic (introduced in C++11) so that
*        if the mesh class implements a move constructor we can move the
*        mesh into the Quadrature, saving memory.
* @note   It is an example of the Bridge Design pattern: part of the implementation is delegated to a polymorphic object (the QuadratureRule).
*/
class Quadrature
{
public:
  /*!
  * @brief The type of the function to be integrated
  */
  typedef apsc::NumericalIntegration::FunPoint FunPoint;

  /*!
  * @brief Constructor
  * @param rule A unique_ptr storing the rule.
  * @param mesh The 1D mesh (passed via universal reference)
  */
  template <typename QuadHandler, typename MESH>
  Quadrature(QuadHandler &&rule, MESH &&mesh)
    : rule_(std::forward<QuadHandler>(rule)), mesh_(std::forward<MESH>(mesh))
  {}

  //! A second constructor
  /*!  
  * @brief Constructor
  * @param rule The rule.
  * @param mesh The 1D mesh
  * @note In this case the object is passed, with the
  *       QuadratureRule classes Clonable (that is they contain a
  *       clone() method). So, it is possible to safely pass also a reference to a
  *       QuadratureRule base class.  If QuadratureRule where not clonable
  *       it would not work.
  */
  template <typename MESH>
  Quadrature(const QuadratureRuleBase &rule, MESH &&mesh)
    : rule_(rule.clone()), mesh_(std::forward<MESH>(mesh))
  {}

  /*!
  * @brief Copy constructor
  * @todo I could have used the Wrapper class in cloningUtilities.hpp and save
  *       the need of building copy/move and assignement operator. I could have
  *       used the synthetic ones. Since instead I am storing a unique_ptr,
  *       if I want to make the class copiable/movable with a deep copy, I need
  *       to write the operators myself, exploiting clone().
  */
  Quadrature(Quadrature const &rhs) = default;

  /*!
  * @brief Move constructor.
  * @param rhs The input quadrature
  */
  Quadrature(Quadrature &&rhs) = default;
  
  /*!
  * @brief Copy assignment
  */
  Quadrature &operator=(Quadrature const &) = default;
  
  /*!
  * @brief Move assignment
  */
  Quadrature &operator=(Quadrature &&) = default;

  /*!
  * @brief Calculates the integal on the passed integrand function
  * @param f integrand
  * @return the integral over the domain of the function
  */
  double apply(FunPoint const &f) const
  {
    double result(0);
#ifndef PARALLELCPP
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : result) shared(f)
#endif
    for(unsigned int i = 0u; i < mesh_.numNodes() - 1; ++i)
    {
      double const a = mesh_[i];
      double const b = mesh_[i + 1];
      result += rule_->apply(f, a, b);
    }
#else
    result = std::transform_reduce(std::execution::par_unseq, mesh_.begin(),
                                   mesh_.end() - 1, 0.0, std::plus<double>(),
                                   [this, &f](double const a, double const b) {
                                     return this->rule_->apply(f, a, b);
                                   });
#endif
    return result;
  }

  /*!
  * @brief Getter for the rule
  * @return a const reference to the quadrature rule
  */
  QuadratureRuleBase const &
  myRule() const
  {
    return *(rule_.get());
  }

protected:
  /*!Rule*/
  QuadratureRuleHandler rule_;
  /*!Mesh over which integrating*/
  Mesh1D                mesh_;
};

} // namespace apsc::NumericalIntegration

#endif  /*NUMERICAL_INTEGRATION_HPP*/
