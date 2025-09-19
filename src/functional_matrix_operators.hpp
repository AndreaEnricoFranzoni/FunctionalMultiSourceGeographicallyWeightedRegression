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
#include "functional_matrix_utils.hpp"
#include <cmath>


// Operators can be defined throus static or non static call operators.
//  CLASSES THAT ENCAPSULATE OPERATIONS
//  TO AVOID EIGEN OVERLOADING
//! Binary operator expression.
template <class LO, class RO, class OP, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<LO>  &&  fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class BinaryOperator : public Expr<BinaryOperator<LO, RO, OP, INPUT, OUTPUT>, INPUT, OUTPUT>
{
public:
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>; 

  BinaryOperator(LO const &l, RO const &r) : M_lo(l), M_ro(r){};

  /*!
  * @brief Applies operation on operands
  */
  F_OBJ
  operator()(std::size_t i, std::size_t j) 
  const
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
template <class RO, class OP, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class UnaryOperator : public Expr<UnaryOperator<RO, OP, INPUT, OUTPUT>, INPUT, OUTPUT>
{
public:
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;

  UnaryOperator(RO const &r) : M_ro(r){};

  /*!
  * @brief Applies operation on operands
  */
  F_OBJ
  operator()(std::size_t i, std::size_t j) 
  const
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
template <class RO, class OP, typename INPUT, typename OUTPUT>
    requires fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class BinaryOperator<double, RO, OP, INPUT, OUTPUT>
  : public Expr<BinaryOperator<double, RO, OP, INPUT, OUTPUT>, INPUT, OUTPUT>
{
public:
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  using LO = double;
  BinaryOperator(LO const &l, RO const &r) : M_lo(l), M_ro(r){};

  F_OBJ
  operator()(std::size_t i, std::size_t j) 
  const
  {
    return OP()(M_lo, M_ro(i,j));
  }

  std::size_t
  rows() 
  const
  {
    return M_ro.rows();
  }

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
  LO const  M_lo;
  RO const &M_ro;
};

//! Specialization for operation by a scalar
template <class LO, class OP, typename INPUT, typename OUTPUT>
    requires fm_utils::not_eigen<LO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class BinaryOperator<LO, double, OP, INPUT, OUTPUT>
  : public Expr<BinaryOperator<LO, double, OP, INPUT, OUTPUT>, INPUT, OUTPUT >
{
public:
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  using RO = double;
  BinaryOperator(LO const &l, RO const &r) : M_lo(l), M_ro(r){};

  F_OBJ
  operator()(std::size_t i, std::size_t j) 
  const
  {
    return OP()(M_lo(i,j), M_ro);
  }

  std::size_t
  rows() 
  const
  {
    return M_lo.rows();
  }

  std::size_t
  cols() 
  const
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
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct Add
{
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>; //type of the input(including eventual const ref)

  F_OBJ
  operator()(F_OBJ f1, F_OBJ f2) 
  const
  {
    return [f1,f2](F_OBJ_INPUT x){return f1(x) + f2(x);};
  }
};

//! The basic Multiplication
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct Multiply
{
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>; //type of the input(including eventual const ref)

  F_OBJ
  operator()(F_OBJ f1, F_OBJ f2) 
  const
  {
    return [f1,f2](F_OBJ_INPUT x){return f1(x) * f2(x);};
  }

  F_OBJ
  operator()(double a, F_OBJ f) 
  const
  {
    return [a,f](F_OBJ_INPUT x){return a * f(x);};
  }

  F_OBJ
  operator()(F_OBJ f, double a) 
  const
  {
    return [a,f](F_OBJ_INPUT x){return a * f(x);};
  }
};

//! The basic Subtraction
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct Subtract
{
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>; //type of the input(including eventual const ref)

  F_OBJ
  operator()(F_OBJ f1, F_OBJ f2) 
  const
  {
    return [f1,f2](F_OBJ_INPUT x){return f1(x) - f2(x);};
  }
};

//! Minus operator
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct Minus
{
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>; //type of the input(including eventual const ref)

  F_OBJ
  operator()(F_OBJ f) 
  const
  {
    return [f](F_OBJ_INPUT x){return -f(x);};
  }
};

// Some fancier operators
//! Exponential
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct ExpOP
{
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>; //type of the input(including eventual const ref)

  F_OBJ
  operator()(F_OBJ f) 
  const
  {
    return [f](F_OBJ_INPUT x){return std::exp(f(x));};
  }
};

//! Logarithm
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct LogOP
{
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>; //type of the input(including eventual const ref)

  F_OBJ
  operator()(F_OBJ f) 
  const
  {
    return [f](F_OBJ_INPUT x){return std::log(f(x));};
  }
};

// WRAPPING THE BASE OPERATIONS INTO THE OPERATION CLASSES: ARE JUST TYPEDEFS
template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<LO>  &&  fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using AddExpr = BinaryOperator<LO, RO, Add<INPUT,OUTPUT>, INPUT, OUTPUT>;

template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<LO>  &&  fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>) 
using MultExpr = BinaryOperator<LO, RO, Multiply<INPUT,OUTPUT>, INPUT, OUTPUT>;

template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<LO>  &&  fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using SubExpr = BinaryOperator<LO, RO, Subtract<INPUT,OUTPUT>, INPUT, OUTPUT>;

template <class RO, typename INPUT = double, typename OUTPUT = double> 
    requires fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using MinusExpr = UnaryOperator<RO, Minus<INPUT,OUTPUT>, INPUT, OUTPUT>;

template <class RO, typename INPUT = double, typename OUTPUT = double> 
    requires fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using ExpExpr = UnaryOperator<RO, ExpOP<INPUT,OUTPUT>, INPUT, OUTPUT>;

template <class RO, typename INPUT = double, typename OUTPUT = double> 
    requires fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using LogExpr = UnaryOperator<RO, LogOP<INPUT,OUTPUT>, INPUT, OUTPUT>;


//  USER LEVEL OPERATORS: THESE ARE THE ONLY ONES THE USER WILL ADOPT
//! Addition of  expression
template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<LO>  &&  fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
AddExpr<LO, RO, INPUT, OUTPUT>
operator+(LO const &l, RO const &r)
{
  return AddExpr<LO, RO, INPUT, OUTPUT>(l, r);
}

//! Multiplication of expressions
template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<LO>  &&  fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
MultExpr<LO, RO, INPUT, OUTPUT>
operator*(LO const &l, RO const &r)
{
  return MultExpr<LO, RO, INPUT, OUTPUT>(l, r);
}

template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<LO>  &&  fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
SubExpr<LO, RO, INPUT, OUTPUT>
operator-(LO const &l, RO const &r)
{
  return SubExpr<LO, RO, INPUT, OUTPUT>(l, r);
}

template <class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
MinusExpr<RO, INPUT, OUTPUT>
operator-(RO const &r)
{
  return MinusExpr<RO, INPUT, OUTPUT>(r);
}

//! Exponential
template <class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
ExpExpr<RO, INPUT, OUTPUT>
exp(RO const &r)
{
  return ExpExpr<RO, INPUT, OUTPUT>(r);
}

//! Logarithm
template <class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::not_eigen<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
LogExpr<RO, INPUT, OUTPUT>
log(RO const &r)
{
  return LogExpr<RO, INPUT, OUTPUT>(r);
}

#endif  /*FUNCTIONAL_MATRIX_OPERATORS_HPP*/