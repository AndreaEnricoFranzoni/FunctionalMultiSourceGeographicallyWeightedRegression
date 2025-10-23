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


#ifndef FUNCTIONAL_MATRIX_EXPRESSION_WRAPPER_HPP
#define FUNCTIONAL_MATRIX_EXPRESSION_WRAPPER_HPP

#include <functional>
#include <type_traits>
#include <utility>
#include <concepts>
#include <iterator>

#include "functional_matrix_storing_type.hpp"



/*!
* @file functional_matrix_expression_wrapper.hpp
* @brief Contains the definition of a wrapper for expression for matrices (of functions)
* @author Andrea Enrico Franzoni
*/




/*!
* @struct Expr
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief A wrapper for expression of matrices (of functions)
* @details The class is used for CRTP
* @code
* class AnyExpression: public Expr<AnyExpression>
* {...};
* @endcode
* @note the derived class need the call operator .(i,j) and the methods .rows(), .cols() and .size()
*/
template <class E, typename INPUT = double, typename OUTPUT = double> 
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct Expr
{
  /*!std::function object stored*/
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;


  /*!
  * @brief Cast operator to the derived class, const version
  * @return a const reference to E, the derived class
  * @note This class is meant to be used with CRTP, so E is derived from Expr<E>, never instantiated alone. 
  *       "this" is always a pointer to an object of the derived class (E): *this is convertible to a reference to E
  */
  operator const E &() const { return static_cast<const E &>(*this); }

  /*!
  * @brief Cast operator to the derived class, non-const version
  * @return a reference to E, the derived class
  */
  operator E &() { return static_cast<E &>(*this); }

  /*!
  * @brief Cast to the derived class, const version
  * @return a const reference to E, the derived class
  * @note an alternative to the corresponding cast operator defined above
  */
  const 
  E &
  asDerived() 
  const
  {
    return static_cast<const E &>(*this);
  }

  /*!
  * @brief Cast to the derived class, non-const version
  * @return a reference to E, the derived class
  * @note an alternative to the corresponding cast operator defined above
  */
  E &
  asDerived()
  {
    return static_cast<E &>(*this);
  }

  /*!
  * @brief Interrogates the number of rows of the wrapped expression, using the .rows() method of the derived class
  * @return the number of rows of the derived class object
  */
  std::size_t
  rows() 
  const
  {
    return asDerived().rows();
  }

  /*!
  * @brief Interrogates the number of columns of the wrapped expression, using the .cols() method of the derived class
  * @return the number of columns of the derived class object
  */
  std::size_t
  cols() 
  const
  {
    return asDerived().cols();
  }

  /*!
  * @brief Interrogates the size of the wrapped expression, using the .size() method of the derived class
  * @return the size of the derived class object
  */
  std::size_t
  size()
  const
  {
    return asDerived().size();
  }

  /*!
  * @brief Delegates to the wrapped expression the calling operator, const version
  * @param i row index
  * @param j col index
  * @return a const reference to the element (i,j) of the wrapped expression
  */
  F_OBJ
  operator()
  (std::size_t i, std::size_t j) 
  const
  {
    return asDerived().operator()(i,j);
  }

  /*!
  * @brief Delegates to the wrapped expression the calling operator, non-const version
  * @param i row index
  * @param j col index
  * @return a reference to the element (i,j) of the wrapped expression
  */
  F_OBJ &
  operator()
  (std::size_t i, std::size_t j)
  {
    return asDerived().operator()(i,j);
  }
};

#endif  /*FUNCTIONAL_MATRIX_EXPRESSION_WRAPPER_HPP*/