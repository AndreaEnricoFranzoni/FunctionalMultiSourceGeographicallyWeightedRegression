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


#ifndef FUNCTIONAL_MATRIX_OPERATORS_HPP
#define FUNCTIONAL_MATRIX_OPERATORS_HPP


#include "functional_matrix_expression_wrapper.hpp"
#include "functional_matrix_utils.hpp"
#include <cmath>


/*!
* @file functional_matrix_operators.hpp
* @brief Contains the definition of the classes for computing element-wise operation within functional matrices
* @author Andrea Enrico Franzoni
*/



/*!
* @class BinaryOperator
* @tparam LO left side operand
* @tparam RO right side operand
* @tparam OP operation 
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Binary element-wise operation within two matrices containing std::function. It encapsulate operations
* @note avoid Eigen overloading
* @details exploiting Expression Templates design pattern
*/
template <class LO, class RO, class OP, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<LO>  &&  fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class BinaryOperator : public Expr<BinaryOperator<LO, RO, OP, INPUT, OUTPUT>, INPUT, OUTPUT>
{
public:
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;

  /*!
  * @brief Constructor
  * @param l left side operand
  * @param r right side operand
  * @note to exploit ETs, operands are saved into the class as const references
  */
  BinaryOperator(LO const &l, RO const &r) : M_lo(l), M_ro(r){};

  /*!
  * @brief Applies operation on operands, element-wise
  * @param i row index of the element of the functional matrix to which apply the operation
  * @param j column index of the element of the functional matrix to which apply the operation
  * @return the resulting std::function
  */
  F_OBJ
  operator()(std::size_t i, std::size_t j) 
  const
  {
    return OP()(M_lo(i,j), M_ro(i,j));
  }

  /*!
  * @brief Rows size
  * @return the number of rows of the operands
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
  * @return the number of columns of the operands
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
  * @return the number of element in the operands
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
  /*!Left side operand*/
  LO const &M_lo;
  /*!Right side operand*/
  RO const &M_ro;
};



/*!
* @class UnaryOperator
* @tparam RO operand
* @tparam OP operation 
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Unary element-wise operation to a matrix containing std::function. It encapsulate operations
* @note avoid Eigen overloading
* @details exploiting Expression Templates design pattern
*/
template <class RO, class OP, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class UnaryOperator : public Expr<UnaryOperator<RO, OP, INPUT, OUTPUT>, INPUT, OUTPUT>
{
public:
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;

  /*!
  * @brief Constructor
  * @param r operand
  * @note to exploit ETs, operand is saved into the class as const references
  */
  UnaryOperator(RO const &r) : M_ro(r){};

  /*!
  * @brief Applies operation to the operand, element-wise
  * @param i row index of the element of the functional matrix to which apply the operation
  * @param j column index of the element of the functional matrix to which apply the operation
  * @return the resulting std::function
  */
  F_OBJ
  operator()(std::size_t i, std::size_t j) 
  const
  {
    return OP()(M_ro(i,j));
  }

  /*!
  * @brief Rows size
  * @return the number of rows of the operand
  */
  std::size_t
  rows() 
  const
  {
    return M_ro.rows();
  }

  /*!
  * @brief Cols size
  * @return the number of columns of the operand
  */
  std::size_t
  cols() 
  const
  {
    return M_ro.cols();
  }

  /*!
  * @brief Data size
  * @return the number of element in the operand
  */
  std::size_t
  size() 
  const
  {
    return M_ro.size();
  }

private:
  /*!Operand*/
  RO const &M_ro;
};



/*!
* @class BinaryOperator, specialization for operation by a scalar
* @tparam RO right side operand
* @tparam OP operation
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Binary element-wise operation within a scalar and a matrix containing std::function. It encapsulate operations
* @note avoid Eigen overloading
* @details exploiting Expression Templates design pattern
*/
template <class RO, class OP, typename INPUT, typename OUTPUT>
    requires fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class BinaryOperator<double, RO, OP, INPUT, OUTPUT>
  : public Expr<BinaryOperator<double, RO, OP, INPUT, OUTPUT>, INPUT, OUTPUT>
{
public:
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  /*!Operation with a scalar*/
  using LO = double;

  /*!
  * @brief Constructor
  * @param l left side operand (scalar)
  * @param r right side operand
  * @note to exploit ETs, right side operator (the matrix) is saved as const reference
  */
  BinaryOperator(LO const &l, RO const &r) : M_lo(l), M_ro(r){};

  /*!
  * @brief Applies operation on operands, element-wise
  * @param i row index of the element of the functional matrix to which apply the operation
  * @param j column index of the element of the functional matrix to which apply the operation
  * @return the resulting std::function
  */
  F_OBJ
  operator()(std::size_t i, std::size_t j) 
  const
  {
    return OP()(M_lo, M_ro(i,j));
  }

  /*!
  * @brief Rows size
  * @return the number of rows of the matrix operand
  */
  std::size_t
  rows() 
  const
  {
    return M_ro.rows();
  }

  /*!
  * @brief Cols size
  * @return the number of columns of the matrix operand
  */
  std::size_t
  cols() 
  const
  {
    return M_ro.cols();
  }

  /*!
  * @brief Data size
  * @return the number of element in the matrix operand
  */
  std::size_t
  size() 
  const
  {
    return M_ro.size();
  }

private:
  /*!Left side operand (scalar)*/
  LO const  M_lo;
  /*!Right side operand*/
  RO const &M_ro;
};



/*!
* @class BinaryOperator, specialization for operation by a scalar
* @tparam LO left side operand
* @tparam OP operation
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Binary element-wise operation within a matrix containing std::function and a scalar. It encapsulate operations
* @note avoid Eigen overloading
* @details exploiting Expression Templates design pattern
*/
template <class LO, class OP, typename INPUT, typename OUTPUT>
    requires fm_utils::functional_matrix_like<LO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class BinaryOperator<LO, double, OP, INPUT, OUTPUT>
  : public Expr<BinaryOperator<LO, double, OP, INPUT, OUTPUT>, INPUT, OUTPUT >
{
public:
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  /*!Operation with a scalar*/
  using RO = double;

  /*!
  * @brief Constructor
  * @param l left side operand
  * @param r right side operand (scalar)
  * @note to exploit ETs, left side operator (the matrix) is saved as const reference
  */
  BinaryOperator(LO const &l, RO const &r) : M_lo(l), M_ro(r){};

  /*!
  * @brief Applies operation on operands, element-wise
  * @param i row index of the element of the functional matrix to which apply the operation
  * @param j column index of the element of the functional matrix to which apply the operation
  * @return the resulting std::function
  */
  F_OBJ
  operator()(std::size_t i, std::size_t j) 
  const
  {
    return OP()(M_lo(i,j), M_ro);
  }

  /*!
  * @brief Rows size
  * @return the number of rows of the matrix operand
  */
  std::size_t
  rows() 
  const
  {
    return M_lo.rows();
  }

  /*!
  * @brief Cols size
  * @return the number of columns of the matrix operand
  */
  std::size_t
  cols() 
  const
  {
    return M_lo.cols();
  }

  /*!
  * @brief Data size
  * @return the number of element in the matrix operand
  */
  std::size_t
  size() 
  const
  {
    return M_lo.size();
  }

private:
  /*!Left side operand*/
  LO const &M_lo;
  /*!Right side operand (scalar)*/
  RO const  M_ro;
};



/*!
* @struct Add
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Basic element-wise addition within two functional matrices
* @note Stored elements need to have the same signature
*/
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct Add
{
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  /*!std::function input type*/
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

  /*!
  * @brief Return the sum of two std::function with the same signature
  * @param f1 first std::function
  * @param f2 second std::function
  * @return the functions sum
  */
  F_OBJ
  operator()(F_OBJ f1, F_OBJ f2) 
  const
  {
    return [f1,f2](F_OBJ_INPUT x){return f1(x) + f2(x);};
  }
};

/*!
* @struct Multiply
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Basic element-wise multiplication within two functional matrices
* @note Stored elements need to have the same signature
*/
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct Multiply
{
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  /*!std::function input type*/
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

  /*!
  * @brief Return the multiplication of two std::function with the same signature
  * @param f1 first std::function
  * @param f2 second std::function
  * @return the functions multiplication
  */
  F_OBJ
  operator()(F_OBJ f1, F_OBJ f2) 
  const
  {
    return [f1,f2](F_OBJ_INPUT x){return f1(x) * f2(x);};
  }

  /*!
  * @brief Return the multiplication of a double and a std::function
  * @param a scalar
  * @param f std::function
  * @return the functions multiplication
  */
  F_OBJ
  operator()(double a, F_OBJ f) 
  const
  {
    return [a,f](F_OBJ_INPUT x){return a * f(x);};
  }

  /*!
  * @brief Return the multiplication of a std::function and a double
  * @param f std::function
  * @param a scalar
  * @return the functions multiplication
  */
  F_OBJ
  operator()(F_OBJ f, double a) 
  const
  {
    return [a,f](F_OBJ_INPUT x){return a * f(x);};
  }
};

/*!
* @struct Subtract
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Basic element-wise difference within two functional matrices
* @note Stored elements need to have the same signature
*/
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct Subtract
{
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  /*!std::function input type*/
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

  /*!
  * @brief Return the difference of two std::function with the same signature
  * @param f1 first std::function
  * @param f2 second std::function
  * @return the functions difference
  */
  F_OBJ
  operator()(F_OBJ f1, F_OBJ f2) 
  const
  {
    return [f1,f2](F_OBJ_INPUT x){return f1(x) - f2(x);};
  }
};

/*!
* @struct Minus
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Basic element-wise opposite of a functional matrix
*/
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct Minus
{
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  /*!std::function input type*/
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

  /*!
  * @brief Return the opposite of a std::function 
  * @param f std::function
  * @return the opposite function
  */
  F_OBJ
  operator()(F_OBJ f) 
  const
  {
    return [f](F_OBJ_INPUT x){return -f(x);};
  }
};

/*!
* @struct ExpOP
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Basic element-wise exponential of a functional matrix
*/
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct ExpOP
{
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  /*!std::function input type*/
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

  /*!
  * @brief Return the exponential of a std::function 
  * @param f std::function
  * @return the exponential of a function
  */
  F_OBJ
  operator()(F_OBJ f) 
  const
  {
    return [f](F_OBJ_INPUT x){return std::exp(f(x));};
  }
};

/*!
* @struct LogOP
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Basic element-wise logarithm of a functional matrix
*/
template <typename INPUT = double, typename OUTPUT = double>
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct LogOP
{
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
  /*!std::function input type*/
  using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

  /*!
  * @brief Return the logarithm of a std::function 
  * @param f std::function
  * @return the logarithm of a function
  */
  F_OBJ
  operator()(F_OBJ f) 
  const
  {
    return [f](F_OBJ_INPUT x){return std::log(f(x));};
  }
};

/*!
* @tparam LO left side operand
* @tparam RO right side operand
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Typedef for the element-wise sum of two functional matrices
*/
template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<LO>  &&  fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using AddExpr = BinaryOperator<LO, RO, Add<INPUT,OUTPUT>, INPUT, OUTPUT>;

/*!
* @tparam LO left side operand
* @tparam RO right side operand
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Typedef for the element-wise multiplication of two functional matrices
*/
template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<LO>  &&  fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>) 
using MultExpr = BinaryOperator<LO, RO, Multiply<INPUT,OUTPUT>, INPUT, OUTPUT>;

/*!
* @tparam LO left side operand
* @tparam RO right side operand
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Typedef for the element-wise difference of two functional matrices
*/
template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<LO>  &&  fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using SubExpr = BinaryOperator<LO, RO, Subtract<INPUT,OUTPUT>, INPUT, OUTPUT>;

/*!
* @tparam RO operand
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Typedef for the element-wise opposite of a functional matrix
*/
template <class RO, typename INPUT = double, typename OUTPUT = double> 
    requires fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using MinusExpr = UnaryOperator<RO, Minus<INPUT,OUTPUT>, INPUT, OUTPUT>;

/*!
* @tparam RO operand
* @tparam OP operation 
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Typedef for the element-wise exponential of a functional matrix
*/
template <class RO, typename INPUT = double, typename OUTPUT = double> 
    requires fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using ExpExpr = UnaryOperator<RO, ExpOP<INPUT,OUTPUT>, INPUT, OUTPUT>;

/*!
* @tparam RO operand
* @tparam OP operation 
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Typedef for the element-wise logarithm of a functional matrix
*/
template <class RO, typename INPUT = double, typename OUTPUT = double> 
    requires fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
using LogExpr = UnaryOperator<RO, LogOP<INPUT,OUTPUT>, INPUT, OUTPUT>;


/*!
* @tparam LO left side operand
* @tparam RO right side operand
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Operator definition for the element-wise sum of two functional matrices
* @param l left side operand
* @param r right side operand
* @return the element-wise sum of the two operands
*/
template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<LO>  &&  fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
AddExpr<LO, RO, INPUT, OUTPUT>
operator+(LO const &l, RO const &r)
{
  return AddExpr<LO, RO, INPUT, OUTPUT>(l, r);
}

/*!
* @tparam LO left side operand
* @tparam RO right side operand
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Operator definition for the element-wise multiplication of two functional matrices
* @param l left side operand
* @param r right side operand
* @return the element-wise multiplication of the two operands
*/
template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<LO>  &&  fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
MultExpr<LO, RO, INPUT, OUTPUT>
operator*(LO const &l, RO const &r)
{
  return MultExpr<LO, RO, INPUT, OUTPUT>(l, r);
}

/*!
* @tparam LO left side operand
* @tparam RO right side operand
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Operator definition for the element-wise difference of two functional matrices
* @param l left side operand
* @param r right side operand
* @return the element-wise difference of the two operands
*/
template <class LO, class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<LO>  &&  fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
SubExpr<LO, RO, INPUT, OUTPUT>
operator-(LO const &l, RO const &r)
{
  return SubExpr<LO, RO, INPUT, OUTPUT>(l, r);
}

/*!
* @tparam RO operand
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Operator definition for the element-wise opposite of a functional matrices
* @param r operand
* @return the element-wise opposite of the operand
*/
template <class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
MinusExpr<RO, INPUT, OUTPUT>
operator-(RO const &r)
{
  return MinusExpr<RO, INPUT, OUTPUT>(r);
}

/*!
* @tparam RO operand
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Operator definition for the element-wise exponential of a functional matrices
* @param r operand
* @return the element-wise exponential of the operand
*/
template <class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
ExpExpr<RO, INPUT, OUTPUT>
exp(RO const &r)
{
  return ExpExpr<RO, INPUT, OUTPUT>(r);
}

/*!
* @tparam RO operand
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Operator definition for the element-wise logarithm of a functional matrices
* @param r operand
* @return the element-wise logarithm of the operand
*/
template <class RO, typename INPUT = double, typename OUTPUT = double>
    requires fm_utils::functional_matrix_like<RO>  &&  (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
LogExpr<RO, INPUT, OUTPUT>
log(RO const &r)
{
  return LogExpr<RO, INPUT, OUTPUT>(r);
}

#endif  /*FUNCTIONAL_MATRIX_OPERATORS_HPP*/