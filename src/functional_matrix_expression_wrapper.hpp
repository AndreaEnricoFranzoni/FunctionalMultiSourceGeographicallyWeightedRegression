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


#ifndef FUNCTIONAL_MATRIX_EXPRESSION_WRAPPER_HPP
#define FUNCTIONAL_MATRIX_EXPRESSION_WRAPPER_HPP

#include <functional>
#include <type_traits>
#include <utility>
#include <concepts>

#include "functional_matrix_storing_type.hpp"



//! A wrapper for expressions
/*!
   This class is an example of use of CRTP (curiosly recursive
   template pattern). Indeed any class that encapsulates an
   expression should derive from it using

   \code
   class AnyExpression: public Expr<AnyExpression>
   {...};
   \endcode

   The derived class should contain the address operator [] and the
   size() method


   The cast operator (also called conversion operator) enables the
   implementation of static polymorphism in a easier way. However, it is not
   strictly necessary: an alternative is to use a method.

 */
template <class E, typename INPUT = double, typename OUTPUT = double> 
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
struct Expr
{
  //type of the function stored
  using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;


  //! Cast operator to derived class (const version)
  /*!
    Remember that this class is meant to be used with CRTP technique.
    So E is derived from Expr<E>!
    It is never instantiated alone.
    So "this" is always a pointer to an object of the derived class (which is
    E!)
    -> *this is convertible to a reference to E!.
  */
  operator const E &() const { return static_cast<const E &>(*this); }
  //! Cast operator to derived class (non const version)
  operator E &() { return static_cast<E &>(*this); }
  /*! The alternative with a method instead of a cast operator.
     It is less flexible however, but maybe clearer!
  */
  const 
  E &
  asDerived() 
  const
  {
    return static_cast<const E &>(*this);
  }

  E &
  asDerived()
  {
    return static_cast<E &>(*this);
  }

  //! Interrogates the size of the wrapped expression
  std::size_t
  rows() 
  const
  {
    return asDerived().rows();
  }

  std::size_t
  cols() 
  const
  {
    return asDerived().cols();
  }

  std::size_t
  size()
  const
  {
    return asDerived().rows() * asDerived().cols();
  }

  //! Delegates to the wrapped expression the addressing operator
  F_OBJ
  operator()
  (std::size_t i, std::size_t j) 
  const
  {
    return asDerived().operator()(i,j);
  }

  //! Delegates to the wrapped expression the addressing operator (non const)
  F_OBJ &
  operator()
  (std::size_t i, std::size_t j)
  {
    return asDerived().operator()(i,j);
  }
};

#endif  /*FUNCTIONAL_MATRIX_EXPRESSION_WRAPPER_HPP*/