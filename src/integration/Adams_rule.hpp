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

#ifndef __NUMERICAL_RULE_HPP
#define __NUMERICAL_RULE_HPP


#include "StandardQuadratureRule.hpp"


/*!
* @file Adams_rule.hpp
* @brief Contains the definition of the concrete quadrature rules
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
* @class Simpson
* @brief Simpson quadrature rule
*/
class Simpson final : public StandardQuadratureRule<3>
{
public:
  /*!
  * @brief Constructor 
  */
  Simpson(): StandardQuadratureRule<3>{{1. / 3, 4. / 3, 1. / 3}, {-1.0, 0.0, 1.0}, 4} {}

  /*!
  * @brief Method to clone 
  * @return an unique pointer to the derived class itself, implicitily derived to the base one
  */
  std::unique_ptr<QuadratureRuleBase> clone() const override {return std::make_unique<Simpson>(*this);}
};

/*!
* @class MidPoint
* @brief Rectangle quadrature rule
*/
class MidPoint final : public StandardQuadratureRule<1>
{
public:
  /*!
  * @brief Constructor 
  */
  MidPoint() : StandardQuadratureRule<1>{{2.0}, {0.0}, 2} {}

  /*!
  * @brief Method to clone 
  * @return an unique pointer to the derived class itself, implicitily derived to the base one
  */
  std::unique_ptr<QuadratureRuleBase> clone() const override {return std::make_unique<MidPoint>(*this);}
};


/*!
* @class Trapezoidal
* @brief Trapezoidal quadrature rule
*/
class Trapezoidal final : public StandardQuadratureRule<2>
{
public:
  /*!
  * @brief Constructor 
  */
  Trapezoidal(): StandardQuadratureRule<2>{{1., 1.}, {-1.0, 1.0}, 2}  {}

  /*!
  * @brief Method to clone 
  * @return an unique pointer to the derived class itself, implicitily derived to the base one
  */
  std::unique_ptr<QuadratureRuleBase> clone() const override{return std::make_unique<Trapezoidal>(*this);}
};

} // namespace apsc::NumericalIntegration

#endif  /*__NUMERICAL_RULE_HPP*/
