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
* @namespace Geometry
* @brief Contains domain geometry definition
*/
namespace Geometry
{

/*!
* @class Domain1D
* @brief Defines a 1D domain.
*/
class Domain1D
{
public:
  /*!
  * @brief Constructor
  * @param a domain left extreme
  * @param b domain right extrem
  * @note Default creates (0,1)
  */
  explicit Domain1D(double const &a = 0., double const &b = 1.) : M_a(a), M_b(b) {}

  /*!
  * @brief Getting domain left extreme
  * @return the private M_a
  */
  double
  left() const
  {
    return M_a;
  }

  /*!
  * @brief Getting domain right extreme
  * @return the private M_b
  */
  double
  right() const
  {
    return M_b;
  }

  /*!
  * @brief Setting domain left extreme
  * @return a reference to the private M_a
  */
  double &
  left()
  {
    return M_a;
  }

  /*!
  * @brief Setting domain right extreme
  * @return a reference to the private M_b
  */
  double &
  right()
  {
    return M_b;
  }

  /*!
  * @brief Domain length
  * @return the measure of the domain
  */
  double length() const
  {
    return std::abs(M_b - M_a);
  }
  
private:
  /*!Domain left extreme*/
  double M_a;
  /*!Domain right extreme*/
  double M_b;
};

} //end namespace Geometry

#endif  /*HH_DOMAIN_HH*/