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


#ifndef FUNCTIONAL_MATRIX_DIAGONAL_HPP
#define FUNCTIONAL_MATRIX_DIAGONAL_HPP


#include "functional_matrix_expression_wrapper.hpp"
#include "functional_matrix_utils.hpp"
#include <utility>
#include <vector>
#include <cassert>


/*!
* @file functional_matrix_diagonal.hpp
* @brief Contains the definition of a sparse matrix containing univariate 1D domain std::function objects
* @author Andrea Enrico Franzoni
*/


/*!
* @class functional_matrix_diagonal
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Class for diagonal matrices storing univariate 1D domain std::function objects
* @details Static polymorphism: deriving from a expression for expression templates
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class functional_matrix_diagonal : public Expr< functional_matrix_diagonal<INPUT,OUTPUT>, INPUT, OUTPUT >
{
    /*!std::function object stored*/
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    /*!std::function input type*/
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    /*!null function (const version)*/
    inline static const F_OBJ m_null_function = [](F_OBJ_INPUT x){ return static_cast<OUTPUT>(0);};
    /*!null function (non-const verion)*/
    inline static F_OBJ m_null_function_non_const = [](F_OBJ_INPUT x){ return static_cast<OUTPUT>(0);};

private:
    /*!Number of rows*/
    std::size_t m_rows;
    /*!Number of cols*/
    std::size_t m_cols;
    /*!Container for the std::function. The storage order is the diagonal, in order*/
    std::vector< F_OBJ > m_data;

public:

    /*!
    * @brief Default constructor
    */
    functional_matrix_diagonal() = default;

    /*!
    * @brief Constructor
    * @param fm vector of std::function objects
    * @param n_rows number of rows
    * @param n_cols number of columns
    */    
    functional_matrix_diagonal(std::vector< F_OBJ > const &fm,
                               std::size_t n)
                :   m_rows(n), m_cols(n), m_data{fm} 
                {}

    /*!
    * @brief Constructor with move semantic
    * @param fm vector of std::function objects
    * @param n_rows number of rows
    * @param n_cols number of columns
    */            
    functional_matrix_diagonal(std::vector< F_OBJ > &&fm,
                               std::size_t n)
                :   m_rows(n), m_cols(n), m_data{std::move(fm)} 
                {}

    /*!
    * @brief Constructor that initializes all the matrix elements with the same std::function object
    * @param n_rows number of rows
    * @param n_cols number of columns
    * @param f value used to initialize all the matrices elements (default is unit function)
    */            
    functional_matrix_diagonal(std::size_t n, F_OBJ f = [](F_OBJ_INPUT){return static_cast<OUTPUT>(1);})    : m_rows(n), m_cols(n), m_data(n,f)   {};

    /*!
    * @brief Copy constructor
    */    
    functional_matrix_diagonal(functional_matrix_diagonal const &) = default;

    /*!
    * @brief Move constructor
    */    
    functional_matrix_diagonal(functional_matrix_diagonal &&) = default;

    /*!
    * @brief Copy assignment
    */    
    functional_matrix_diagonal &operator=(functional_matrix_diagonal const &) = default;

    /*!
    * @brief Move assignment
    */    
    functional_matrix_diagonal &operator=(functional_matrix_diagonal &&) = default;

    /*!
    * @brief Constructor that builds a functional_matrix_diagonal from an Expr 
    * @tparam T template param indicating the derived type from expression from which the cast is done
    * @param e expression from which constructing a functional_matrix_diagonal
    * @details necessary for ETs design
    */    
    template <class T> 
    functional_matrix_diagonal(const Expr<T,INPUT,OUTPUT> &e)
        :   m_data()
    {
        //casting
        const T &et(e); 
        m_rows = et.cols(); 
        m_cols = et.cols(); 
        m_data.reserve(et.cols());
        for(std::size_t i = 0; i < et.cols(); ++i){     m_data.emplace_back(et(i,i));}
    }

    /*!
    * @brief Copy assignment from an Expr
    * @tparam T template param indicating the derived type from expression from which the cast is done
    * @param e expression from which constructing a functional_matrix_diagonal
    * @details necessary for ETs design
    */
    template <class T>
    functional_matrix_diagonal &
    operator=(const Expr<T,INPUT,OUTPUT> &e)
    {
        //casting
        const T &et(e); 
        m_rows = et.cols(); 
        m_cols = et.cols();
        m_data.resize(et.cols());
        for(std::size_t i = 0; i < et.cols(); ++i){   m_data[i] = et(i,i);}

        return *this;
    }

    /*!
    * @brief Returns element (i,j) of the matrix, non-const version
    * @param i row index
    * @param j column index
    * @return a refence to std::function in position (i,j)
    */
    F_OBJ &
    operator()
    (std::size_t i, std::size_t j)
    {
        return i==j ? m_data[i] : this->m_null_function_non_const;
    }

    /*!
    * @brief Returns element (i,j) of the matrix, const version
    * @param i row index
    * @param j column index
    * @return the std::function in position (i,j), read-only
    */
    F_OBJ
    operator()
    (std::size_t i, std::size_t j)
    const
    {
        return i==j ? m_data[i] : this->m_null_function;
    }

    /*!
    * @brief Rows size
    * @return the number of rows
    */
    std::size_t
    rows() 
    const
    {
        return m_rows;
    }

    /*!
    * @brief Columns size
    * @return the number of columns
    */
    std::size_t
    cols() 
    const
    {
        return m_cols;
    }

    /*!
    * @brief Number of elements
    * @return the total number of stored elements
    */
    std::size_t
    size() 
    const
    {
        return m_cols;
    }

    /*!
    * @brief Casting operator to a std::vector &, const version
    * @return a const reference to a std::vector of std::function, containing the function stored into the matrix
    * @code
    * functional_matrix_diagonal<INPUT,OUTPUT>  fm;
    * std::vector< FUNC_OBJ<INPUT,OUTPUT> > & fm_v(fm); 
    * FUNC_OBJ<INPUT,OUTPUT> f = [](fm_utils::input_param_t<FUNC_OBJ<INPUT,OUTPUT>> x){return static_cast<OUTPUT>(10.0);};
    * fm_v.emplace_back(f);
    * @endcode
    */
    operator std::vector< F_OBJ > const &() const { return m_data; }

    /*!
    * @brief Casting operator to a std::vector &, non-const version
    * @return a reference to a std::vector of std::function, containing the function stored into the matrix
    */    
    operator std::vector< F_OBJ > &() { return m_data; }
  
    /*!
    * @brief Casting to a std::vector &, const version
    * @return a const reference to a std::vector of std::function, containing the function stored into the matrix
    * @code
    * functional_matrix_diagonal<INPUT,OUTPUT>  fm;
    * std::vector< FUNC_OBJ<INPUT,OUTPUT> > & fm_v{fm.sa_vector()}; 
    * FUNC_OBJ<INPUT,OUTPUT> f = [](fm_utils::input_param_t<FUNC_OBJ<INPUT,OUTPUT>> x){return static_cast<OUTPUT>(10.0);};
    * fm_v.emplace_back(f);
    * @endcode
    */
    std::vector< F_OBJ > const &
    as_vector() 
    const
    {
        return m_data;
    }

    /*!
    * @brief Casting to a std::vector &, non-const version
    * @return a const reference to a std::vector of std::function, containing the function stored into the matrix
    */    
    std::vector< F_OBJ > &
    as_vector()
    {
        return m_data;
    }
};


/*!
* @brief Function to use range for loops over the stored functions, begin iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a reference to a functional_matrix_diagonal object
* @return an iterator to the begin of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
begin(functional_matrix_diagonal<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().begin())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).begin();
}

/*!
* @brief Function to use range for loops over the stored functions, end iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a reference to a functional_matrix_diagonal object
* @return an iterator to the end of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
end(functional_matrix_diagonal<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().end())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).end();
}

/*!
* @brief Function to use range for loops over the stored functions, const begin iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a const reference to a functional_matrix_diagonal object
* @return a constant iterator to the begin of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cbegin(functional_matrix_diagonal<INPUT,OUTPUT> const &fm)
  -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cbegin())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>const &
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cbegin();
}

/*!
* @brief Function to use range for loops over the stored functions, const end iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a const reference to a functional_matrix_diagonal object
* @return a constant iterator to the end of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cend(functional_matrix_diagonal<INPUT,OUTPUT> const &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cend())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>const &
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cend();
}

#endif  /*FUNCTIONAL_MATRIX_DIAGONAL_HPP*/