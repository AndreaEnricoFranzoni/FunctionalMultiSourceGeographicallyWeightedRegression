// Copyright (c) 2024 Andrea Enrico Franzoni (andreaenrico.franzoni@gmail.com)
//
// This file is part of PPCKO
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of PPCKO and associated documentation files (the PPCKO software), to deal
// PPCKO without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of PPCKO, and to permit persons to whom PPCKO is
// furnished to do so, subject to the following conditions:
//
// PPCKO IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH PPCKO OR THE USE OR OTHER DEALINGS IN
// PPCKO.

#ifndef HH_DOMAIN_HH
#define HH_DOMAIN_HH

#include <cmath>

/*!
* @file domain.hpp
* @brief Contains the class for an unidimensioanl domain. 
* @author Luca Formaggia
* @note Taken from pacs-examples, folder of repository PACS Course (https://github.com/pacs-course), Advanced Programming for Scientific Computing, Politecnico di Milano
*/


/*!
  Defines a 1D domain.
 */
namespace Geometry
{
class Domain1D
{
public:
  //! Constructor. Default creates (0,1)
  explicit Domain1D(double const &a = 0., double const &b = 1.) : M_a(a), M_b(b) {}
  /*! \defgroup Accessor Accessing elements
    @{ */
  double
  left() const
  {
    return M_a;
  }
  double
  right() const
  {
    return M_b;
  }
  double &
  left()
  {
    return M_a;
  }
  double &
  right()
  {
    return M_b;
  }
  double length() const
  {
    return std::abs(M_b - M_a);
  }
  /*! @}*/
private:
  double M_a;
  double M_b;
};
} // namespace Geometry
#endif