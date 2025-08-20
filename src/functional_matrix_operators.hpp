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
// OUT OF OR IN CONNECTION WITH PPCKO OR THE USE OR OTHER DEALINGS IN
// fdagwr.


#ifndef FUNCTIONAL_MATRIX_OPERATORS_HPP
#define FUNCTIONAL_MATRIX_OPERATORS_HPP


#include "functional_matrix_expression_wrapper.hpp"


// Operators can be defined throus static or non static call operators.
//  CLASSES THAT ENCAPSULATE OPERATIONS
//! Binary operator expression.
template <class LO, class RO, class OP>
class BinaryOperator : public Expr<BinaryOperator<LO, RO, OP> >
{
public:
  BinaryOperator(LO const &l, RO const &r) : M_lo(l), M_ro(r){};

  /*!
  * @brief Applies operation on operands
  */
  FDAGWR_TRAITS::f_type
  operator()(std::size_t i, std::size_t j) const
  {
    return OP()(M_lo(i,j), M_ro(i,j));
  }

  /*!
  * @brief Rows size
  */
  std::size_t
  rows() 
  const
  {
    // disabled when NDEBUG is set. Checks if both operands have the same size
    assert(M_lo.rows() == M_ro.rows());
    return M_lo.rows();
  }

  /*!
  * @brief Cols size
  */
  std::size_t
  cols() 
  const
  {
    // disabled when NDEBUG is set. Checks if both operands have the same size
    assert(M_lo.cols() == M_ro.cols());
    return M_lo.cols();
  }

  /*!
  * @brief Data size
  */
  std::size_t
  size() 
  const
  {
    // disabled when NDEBUG is set. Checks if both operands have the same size
    assert(M_lo.size() == M_ro.size());
    return M_lo.size();
  }

private:
  LO const &M_lo;
  RO const &M_ro;
};





//! Unary operator expression.
template <class RO, class OP>
class UnaryOperator : public Expr<UnaryOperator<RO, OP> >
{
public:
  UnaryOperator(RO const &r) : M_ro(r){};

  /*!
  * @brief Applies operation on operands
  */
  FDAGWR_TRAITS::f_type
  operator()(std::size_t i, std::size_t j) const
  {
    return OP()(M_ro(i,j));
  }

  /*!
  * @brief Rows size
  */
  std::size_t
  rows() 
  const
  {
    return M_ro.rows();
  }

  /*!
  * @brief Cols size
  */
  std::size_t
  cols() 
  const
  {
    return M_ro.cols();
  }

  /*!
  * @brief Data size
  */
  std::size_t
  size() 
  const
  {
    return M_ro.size();
  }

private:
  RO const &M_ro;
};

//! Specialization for operation by a scalar
template <class RO, class OP>
class BinaryOperator<double, RO, OP>
  : public Expr<BinaryOperator<double, RO, OP> >
{
public:
  using LO = double;
  BinaryOperator(LO const &l, RO const &r) : M_lo(l), M_ro(r){};

  FDAGWR_TRAITS::f_type
  operator()(std::size_t i, std::size_t j) const
  {
    return OP()(M_lo, M_ro(i,j));
  }

  std::size_t
  rows() const
  {
    return M_ro.rows();
  }

  std::size_t
  cols() const
  {
    return M_ro.cols();
  }

  /*!
  * @brief Data size
  */
  std::size_t
  size() 
  const
  {
    return M_ro.size();
  }

private:
  LO const  M_lo;
  RO const &M_ro;
};

//! Specialization for operation by a scalar
template <class LO, class OP>
class BinaryOperator<LO, double, OP>
  : public Expr<BinaryOperator<LO, double, OP> >
{
public:
  using RO = double;
  BinaryOperator(LO const &l, RO const &r) : M_lo(l), M_ro(r){};

  FDAGWR_TRAITS::f_type
  operator()(std::size_t i, std::size_t j) const
  {
    return OP()(M_lo(i,j), M_ro);
  }

  std::size_t
  rows() const
  {
    return M_lo.rows();
  }

  std::size_t
  cols() const
  {
    return M_lo.cols();
  }

  /*!
  * @brief Data size
  */
  std::size_t
  size() 
  const
  {
    return M_lo.size();
  }

private:
  LO const &M_lo;
  RO const  M_ro;
};

//  THE BASIC OPERATIONS AT ELEMENT LEVEL
//! The basic Addition
/*!
  Note that we can use directly the functors
  provided by the standard library!
  /code
  using Add = std::add<double>;
  /endcode
*/
struct Add
{
  FDAGWR_TRAITS::f_type
  operator()(FDAGWR_TRAITS::f_type f1, FDAGWR_TRAITS::f_type f2) const
  {
    return [f1,f2](const double& x){return f1(x) + f2(x);};
  }
};
//! The basic Multiplication
struct Multiply
{
  FDAGWR_TRAITS::f_type
  operator()(FDAGWR_TRAITS::f_type f1, FDAGWR_TRAITS::f_type f2) const
  {
    return [f1,f2](const double& x){return f1(x) * f2(x);};
  }

  FDAGWR_TRAITS::f_type
  operator()(double a, FDAGWR_TRAITS::f_type f) const
  {
    return [a,f](const double& x){return a * f(x);};
  }

  FDAGWR_TRAITS::f_type
  operator()(FDAGWR_TRAITS::f_type f, double a) const
  {
    return [a,f](const double& x){return a * f(x);};
  }
};

//! The basic Subtraction
struct Subtract
{
  FDAGWR_TRAITS::f_type
  operator()(FDAGWR_TRAITS::f_type f1, FDAGWR_TRAITS::f_type f2) const
  {
    return [f1,f2](const double& x){return f1(x) - f2(x);};
  }
};

//! Minus operator
struct Minus
{
  FDAGWR_TRAITS::f_type
  operator()(FDAGWR_TRAITS::f_type f) const
  {
    return [f](const double& x){return -f(x);};
  }
};

// Some fancier operators
//! Exponential
struct ExpOP
{
  FDAGWR_TRAITS::f_type
  operator()(FDAGWR_TRAITS::f_type f) const
  {
    return [f](const double& x){return std::exp(f(x));};
  }
};

//! Logarithm
struct LogOP
{
  FDAGWR_TRAITS::f_type
  operator()(FDAGWR_TRAITS::f_type f) const
  {
    return [f](const double& x){return std::log(f(x));};
  }
};

// WRAPPING THE BASE OPERATIONS INTO THE OPERATION CLASSES: ARE JUST TYPEDEFS
template <class LO, class RO> using AddExpr = BinaryOperator<LO, RO, Add>;

template <class LO, class RO> using MultExpr = BinaryOperator<LO, RO, Multiply>;

template <class LO, class RO> using SubExpr = BinaryOperator<LO, RO, Subtract>;

template <class RO> using MinusExpr = UnaryOperator<RO, Minus>;

template <class RO> using ExpExpr = UnaryOperator<RO, ExpOP>;

template <class RO> using LogExpr = UnaryOperator<RO, LogOP>;

//  USER LEVEL OPERATORS: THESE ARE THE ONLY ONES THE USER WILL ADOPT

//! Addition of  expression
template <class LO, class RO>
inline AddExpr<LO, RO>
operator+(LO const &l, RO const &r)
{
  return AddExpr<LO, RO>(l, r);
}

//! Multiplication of expressions
template <class LO, class RO>
inline MultExpr<LO, RO>
operator*(LO const &l, RO const &r)
{
  return MultExpr<LO, RO>(l, r);
}

template <class LO, class RO>
inline SubExpr<LO, RO>
operator-(LO const &l, RO const &r)
{
  return SubExpr<LO, RO>(l, r);
}

template <class RO>
inline MinusExpr<RO>
operator-(RO const &r)
{
  return MinusExpr<RO>(r);
}

//! Exponential
template <class RO>
inline ExpExpr<RO>
exp(RO const &r)
{
  return ExpExpr<RO>(r);
}

//! Logarithm
template <class RO>
inline LogExpr<RO>
log(RO const &r)
{
  return LogExpr<RO>(r);
}

#endif  /*FUNCTIONAL_MATRIX_OPERATORS_HPP*/