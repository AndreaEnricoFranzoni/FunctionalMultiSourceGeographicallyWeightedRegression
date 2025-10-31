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

#ifndef HH_QUADRATURERULEBASE_HH
#define HH_QUADRATURERULEBASE_HH


#include "QuadratureRuleTraits.hpp"
#include <functional>
#include <memory>
#include <string>


/*!
* @file QuadratureRuleBase.hpp
* @brief Contains the class for basic integration rules. 
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
* @class QuadratureRuleBase
* @brief Basis class for all the basic integration rules
*        This basis class is the common class for all numerical integration
*        formulae that approximate the integral \f$ \int_{-1}^{1} f(y) dy\f$.
*        It's a very light class (no variable members) that provides
*        the common interface to all integration rules.
* @note This class is an interface (no data members). It is done on purpose.
*       Constructors and assignment operators are not defined since the synthetic ones are sufficient.
*/
class QuadratureRuleBase
{
public:

  /*!
  * @brief The class is clonable. Having a clonable class makes it possible to write copy constructors
  *        and assignment operators for classes that aggregate object of the QuadratureRule hierarchy by composition.
  * @return a unique pointer to this base class
  */
  virtual std::unique_ptr<QuadratureRuleBase> clone() const = 0;

  /*!
  * @brief Integrates in the interval (a,b)
  * @param f integrand
  * @param a left extreme integration domain
  * @param b right extreme integration domain
  * @return the integral evaluated
  */
  virtual double apply(FunPoint const &f, double const &a,
                       double const &b) const = 0;

  /*!
  * @brief Virtual destructor
  */                     
  virtual ~QuadratureRuleBase() = default;

  /*!
  * @brief Sets the target error of an adaptive rule
  * @note Needed only for the adaptive quadrature. Enrich the class, so cannot be abstract
  */
  virtual void
  setTargetError(double const)
  {}

  /*!
  * @brief Sets the maximal level of refinement in an adaptive rule
  * @note Needed only for the adaptive quadrature. Enrich the class, so cannot be abstract
  */
  virtual void
  setMaxIter(unsigned int)
  {}

  /*!
  * @brief Name of the quadrature rule type
  * @return the name of the quadrature rule type
  */
  virtual std::string name() const = 0;
};

/*!
* @brief type of the object holding the quadrature rule.
* @note using the PoiterWrapper for clonable classes defined in CloningUtilities.hpp
*/
using QuadratureRuleHandler = apsc::PointerWrapper<QuadratureRuleBase>;

} // namespace apsc::NumericalIntegration

#endif  /*!HH_QUADRATURERULEBASE_HH*/
