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


#ifndef FUNCTIONAL_MATRIX_DIAGONAL_HPP
#define FUNCTIONAL_MATRIX_DIAGONAL_HPP


#include "functional_matrix_expression_wrapper.hpp"
#include "functional_matrix_utils.hpp"
#include <utility>
#include <vector>
#include <cassert>


//! A class for diagonal matrices of functions
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class functional_matrix_diagonal : public Expr< functional_matrix_diagonal<INPUT,OUTPUT>, INPUT, OUTPUT >
{
//type of the function stored
using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    //null function
    static constexpr auto m_null_function = [](F_OBJ_INPUT x){ return static_cast<OUTPUT>(0);};

private:
    /*!Number of rows*/
    std::size_t m_rows;
    /*!Number of cols*/
    std::size_t m_cols;
    /*!Matrix of functions*/
    std::vector< F_OBJ > m_data;

public:
    //! default constructor
    functional_matrix_diagonal() = default;
    //constructor
    functional_matrix_diagonal(std::vector< F_OBJ > const &fm,
                               std::size_t n)
                :   m_rows(n), m_cols(n), m_data{fm} 
                {}
    //constuctor with move semantic
    functional_matrix_diagonal(std::vector< F_OBJ > &&fm,
                               std::size_t n)
                :   m_rows(n), m_cols(n), m_data{std::move(fm)} 
                {}
    //! Construct a Vector of n elements initialised by value   //DA METTERE IL DEFAULT f=[](const INPUT &){return static_cast<OUTPUT>(1.0);} 
    functional_matrix_diagonal(std::size_t n, F_OBJ f = [](F_OBJ_INPUT){return static_cast<OUTPUT>(1);})    : m_rows(n), m_cols(n), m_data(n,f)   {};
    //! Copy constructor
    functional_matrix_diagonal(functional_matrix_diagonal const &) = default;
    //! Move constructor
    functional_matrix_diagonal(functional_matrix_diagonal &&) = default;
    //! Copy assign
    functional_matrix_diagonal &operator=(functional_matrix_diagonal const &) = default;
    //! Move assign
    functional_matrix_diagonal &operator=(functional_matrix_diagonal &&) = default;

    template <class T> 
    functional_matrix_diagonal(const Expr<T,INPUT,OUTPUT> &e)
        :   m_data()
    {
        const T &et(e); // casting!
        m_rows = et.cols(); 
        m_cols = et.cols(); 
        m_data.reserve(et.cols());
        for(std::size_t i = 0; i < et.cols(); ++i){     m_data.emplace_back(et(i,i));}
    }

    //! Assigning an expression
    /*!
        This method is fundamental for expression template technique.
    */
    template <class T>
    functional_matrix_diagonal &
    operator=(const Expr<T,INPUT,OUTPUT> &e)
    {
        const T &et(e); // casting!
        m_rows = et.cols(); 
        m_cols = et.cols();
        m_data.resize(et.cols());
        for(std::size_t i = 0; i < et.cols(); ++i){   m_data[i] = et(i,i);}

        return *this;
    }

    /*!
    * @brief Returns element (i,j)
    */
    F_OBJ &
    operator()
    (std::size_t i, std::size_t j)
    {
        return i==j ? m_data[i] : this->m_null_function;
    }

    /*!
    * @brief Returns element (i,j) (const version)
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
    */
    std::size_t
    rows() 
    const
    {
        return m_rows;
    }

    /*!
    * @brief Cols size
    */
    std::size_t
    cols() 
    const
    {
        return m_cols;
    }

    /*!
    * @brief Number of elements (elements only on the diagonal)
    */
    std::size_t
    size() 
    const
    {
        return m_cols;
    }

    //! May be cast to a std::vector &
    /*!
    This way I can use all the methods of a std::vector!
    @code
    Vector a;
    std::vector<double> & av(a); // you cannot use {a} here!
    av.emplace_back(10.0);
    @endcode
    */
    operator std::vector< F_OBJ > const &() const { return m_data; }
    //! Non const version of casting operator
    operator std::vector< F_OBJ > &() { return m_data; }
  
    //! This does the same as casting but with a method
    /*!
    Only to show that you do not need casting operators, if you find them
    confusing (or if the compiler gets confused!)
    @code
    Vector a;
    std::vector<double> & av{a.as_vector()}; // Here you may use {}! (but ()
    works as well!) av.emplace_back(10.0);
    @endcode
    */
    std::vector< F_OBJ > const &
    as_vector() 
    const
    {
        return m_data;
    }
    //! Non const version of casting operator
    std::vector< F_OBJ > &
    as_vector()
    {
        return m_data;
    }
};


//! I want to use range for loops with Vector objects.
/*!
  Note the use of declval. I do not need to istantiate a vector to interrogate
  the type returned by begin!
 */
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
begin(functional_matrix_diagonal<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().begin())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).begin();
  // If you prefer
  // return fm.as_vector().begin();
}

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
end(functional_matrix_diagonal<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().end())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).end();
}

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cbegin(functional_matrix_diagonal<INPUT,OUTPUT> const &fm)
  -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cbegin())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>
  // const &
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cbegin();
}

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cend(functional_matrix_diagonal<INPUT,OUTPUT> const &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cend())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>
  // const &
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cend();
}

#endif  /*FUNCTIONAL_MATRIX_DIAGONAL_HPP*/