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

#ifndef QUADRATURE_RULE_HPP
#define QUADRATURE_RULE_HPP


#include <array>
#include <utility>
#include "QuadratureRuleBase.hpp"


/*!
* @file StandardQuadratureRule.hpp
* @brief Contains the class for standard integration rules. 
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
* @class StandardQuadratureRule
* @brief A base class for standard quadrature rules.
*        This class provides the common interface to  a quadrature rule in
*        the interval \f$[-1,1]\f$ computed by the formula
*        \f$ \int_{-1}^{1} f(y) dy \simeq  \sum_{i=0}^{m-1}w_i f(y_i)\f$
*        It also provides the containers for nodes \f$y_i\in[-1,1]\f$ and
*        corresponding weights \f$w_i\f$.  This is the base class, the
*        derived class constructs the appropriate points and weights.
* @tparam NKNOTS the number of knots
* @note Constructors and assignement operators are not defined since the synthetic ones are sufficient.
*       This version implements the virtual constuctor paradigm through the method clone();
*/
template <unsigned int NKNOTS>
class StandardQuadratureRule : public QuadratureRuleBase
{
public:

  /*!
  * @brief Constructo
  * @param weight Weights of the rules
  * @param nodes  Nodes (knots) of the rule
  * @param order Order of the quadrature (exactness +1). Required only if adaptive rule is used
  * @note All quantities in the interval [-1,1]
  */
  StandardQuadratureRule(std::array<double, NKNOTS> const & weight,
                         std::array<double, NKNOTS> const & nodes, double order = 0)
    : w_(weight), n_(nodes), my_order(order)
  {}

  /*!
  * @brief Default constructor
  */
  StandardQuadratureRule() = default;

  /*!
  * @brief Number of nodes used by the rule
  * @return the number of roles used by the rule
  */
  constexpr 
  unsigned int
  num_nodes() 
  const
  {
    return NKNOTS;
  }

  /*!
  * @brief The i-th node
  * @param i the node wanted
  * @return the i-th node
  */
  double
  node(const unsigned int i) 
  const
  {
    return n_[i];
  }

  /*!
  * @brief The i-th weight
  * @param i the weight wanted
  * @return the i-th weight
  */
  double
  weight(const unsigned int i) 
  const
  {
    return w_[i];
  }

  /*!
  * @brief The order of convergence
  * @return the order
  * @note Used for error estimation
  */
  unsigned int
  order() const
  {
    return my_order;
  }

  /*!
  * @brief The general type of quadrature rule
  * @return the general type of quadrature rule
  */
  std::string
  name() const override
    {
      return "Standard Quadrature Rule";
    };

  /*!
  * @brief Having a clonable class makes it possible to write copy constructors
  *        and assignment operators for classes that aggregate object of the
  *        QuadratureRule hierarchy by composition.
  * @return a unique pointer convertible to the base class
  */
  virtual std::unique_ptr<QuadratureRuleBase> clone() const override = 0;

  /*!
  * @brief Applying the quadrature rule to the integrand in (a,b)
  * @param f integrand
  * @param a left integration domain extreme
  * @param b right integration domain extreme
  * @return the integral computed
  */
  double apply(FunPoint const &f, double const &a,
               double const &b) const override;

  /*!
  * @brief Default constructor
  */
  virtual ~StandardQuadratureRule() = default;

protected:
  /*!Weights*/
  std::array<double, NKNOTS> w_;
  /*!Nodes*/
  std::array<double, NKNOTS> n_;
  /*!Order*/
  unsigned int               my_order = 0;

private:
};

/*!
* @brief Applying the quadrature rule to the integrand in (a,b)
* @param f integrand
* @param a left integration domain extreme
* @param b right integration domain extreme
* @return the integral computed
*/
template <unsigned int N>
double
apsc::NumericalIntegration::StandardQuadratureRule<N>::apply(
  FunPoint const &f, double const &a, double const &b) const
{
  double h2=(b - a) * 0.5; // half length
  double xm=(a + b) * 0.5; // midpoint
  // scale functions
  auto   fscaled = [&h2,&xm,&f](double x) { return f(x * h2 + xm); };
  double tmp=0.0;
  auto   np = n_.begin();
  //    for (auto wp=w_.begin();wp<w_.end();++wp,++np)
  for(auto weight : w_)
      {
        tmp += fscaled(*(np)) * weight;
        ++np;
      }

  return h2 * tmp;
}

} // namespace apsc::NumericalIntegration

#endif  /*QUADRATURE_RULE_HPP*/
