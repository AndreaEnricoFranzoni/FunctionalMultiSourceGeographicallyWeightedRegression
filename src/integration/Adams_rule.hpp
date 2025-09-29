#ifndef __NUMERICAL_RULE_HPP
#define __NUMERICAL_RULE_HPP
#include "StandardQuadratureRule.hpp"
namespace apsc::NumericalIntegration
{
/*!
  \file numerical_rule.hpp
  \brief Some quadrature rules.

  Some standard quadrature rules.

 */
//! Simpson rule
class Simpson final : public StandardQuadratureRule<3>
{
public:
  Simpson(): StandardQuadratureRule<3>{{1. / 3, 4. / 3, 1. / 3}, {-1.0, 0.0, 1.0}, 4} {}
  std::unique_ptr<QuadratureRuleBase> clone() const override {return std::make_unique<Simpson>(*this);}
};

//! Midpoint rule
class MidPoint final : public StandardQuadratureRule<1>
{
public:
  MidPoint() : StandardQuadratureRule<1>{{2.0}, {0.0}, 2} {}
  std::unique_ptr<QuadratureRuleBase> clone() const override {return std::make_unique<MidPoint>(*this);}
};

//! Trapezoidal rule
class Trapezoidal final : public StandardQuadratureRule<2>
{
public:
  Trapezoidal(): StandardQuadratureRule<2>{{1., 1.}, {-1.0, 1.0}, 2}  {}
  std::unique_ptr<QuadratureRuleBase> clone() const override{return std::make_unique<Trapezoidal>(*this);}
};

} // namespace apsc::NumericalIntegration
#endif
