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

#ifndef FDAGWR_REMOVE_NAN_HPP
#define FDAGWR_REMOVE_NAN_HPP

#include <Eigen/Dense>

#include "traits_fdagwr.hpp"


#include <stdexcept>



/*!
* @file removing_nan.hpp
* @brief Contains the class to remove (dummy and not dummy) NaNs
* @author Andrea Enrico Franzoni
*/



/*!
* Doing tag dispatching for the correct way of removing non-dummy NaNs.
* @tparam MA_t: template parameter for the non-dummy NaNs removal type to be mapped by std::integral_constant
*/
template <REM_NAN MA_t>
using MAT = std::integral_constant<REM_NAN, MA_t>;


/*!
* @class removing_nan
* @brief Template class for removing NaNs from an Eigen::Matrix.
* @tparam T is the type stored
* @tparam MA_t is the way of removing non-dummy NaNs: 'MR': replacing with the average of the row. 'ZR': replacing with 0s
*/
template<typename T,REM_NAN MA_t>
class removing_nan
{
  
private:
  /*!Matrix containing data from which NaNs have to be removed*/
  fdagwr_traits::Dense_Matrix m_data;
  /*!Number of rows of the matrix*/
  std::size_t m_m;
  /*!Number of columns of the matrix*/
  std::size_t m_n;

  /*!
  * @brief Substituting non-dummy NaNs with row mean.
  */
  void row_removal(Eigen::Block<Eigen::Matrix<T,-1,-1>,1>& row, MAT<REM_NAN::MR>);
  /*!
  * @brief Substituting non-dummy NaNs with 0s.
  */
  void row_removal(Eigen::Block<Eigen::Matrix<T,-1,-1>,1>& row, MAT<REM_NAN::ZR>);
  
public:
  
  /*!
  * @brief Constructor taking the matrix from which NaNs have to be removed
  * @param data matrix from which removing NaNs
  * @details Universal constructor: move semantic used to optimazing handling big size objects
  */
  template<typename STOR_OBJ>
  removing_nan(STOR_OBJ&& data)
    :
    m_data{std::forward<STOR_OBJ>(data)}
    { 
      m_m = m_data.rows();
      m_n = m_data.cols();
    }
  
  /*!
  * @brief Getter for the data matrix
  * @return the private m_data
  */
  inline fdagwr_traits::Dense_Matrix data() const {return m_data;};
  
  /*!
  * @brief Substituting non-dummy NaNs. Tag-dispacther.
  * @param row row from which removing non-dummy NaNs
  * @return tag dispatching to the correct function to substitute non-dummy NaNs
  */
  void row_removal(Eigen::Block<Eigen::Matrix<T,-1,-1>,1>& row) { return row_removal(row, MAT<MA_t>{});};
  
  /*!
  * @brief Function to remove the row (dummy NaNs)
  */
  inline void remove_nan(){   for(auto row : m_data.rowwise()){   row_removal(row);} };
};


#include "removing_nan_imp.hpp"
#include "removing_nan_cleaner_imp.hpp"

#endif /*FDAGWR_REMOVE_NAN_HPP*/