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


#ifndef FDAGWR_FUNCTIONAL_MATRIX_HPP
#define FDAGWR_FUNCTIONAL_MATRIX_HPP


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include "functional_matrix_expression_wrapper.hpp"

#include <cassert>


//! A class for matrices of functions: STORING COLUMN WISE
/*!
  It is build around std::vector<double> and indeed
  std::vector:double> is the only variable member of the class.
 */
class functional_matrix : public Expr<functional_matrix>
{
private:
    /*!Number of rows*/
    std::size_t m_rows;
    /*!Number of cols*/
    std::size_t m_cols;
    /*!Matrix of functions*/
    std::vector<FDAGWR_TRAITS::f_type> m_data;

public:
    //! default constructor
    functional_matrix() = default;
    //! A vector may be converted to a Vector
    functional_matrix(std::vector<FDAGWR_TRAITS::f_type> const &fm,
                      std::size_t n_rows,
                      std::size_t n_cols)
                :   m_rows(n_rows), m_cols(n_cols), m_data{fm} 
                {
                    //cheack input consistency
                    assert((void("Number of rows times number of cols has to be equal to the number of stored functions"), m_rows * m_cols == m_data.size()));
                }
    //! A vector may also be moved in a Vector
    functional_matrix(std::vector<FDAGWR_TRAITS::f_type> &&fm,
                      std::size_t n_rows,
                      std::size_t n_cols) 
                :   m_rows(n_rows), m_cols(n_cols), m_data{std::move(fm)}
                {
                    //cheack input consistency
                    assert((void("Number of rows times number of cols has to be equal to the number of stored functions"), m_rows * m_cols == m_data.size()));
                }
    //! Construct a Vector of n elements initialised by value
    functional_matrix(std::size_t n_rows, std::size_t n_cols, FDAGWR_TRAITS::f_type f=[](const double &){return static_cast<double>(1.0);}) : m_rows(n_rows), m_cols(n_cols), m_data(m_rows*m_cols,f)   {};
    //! Copy constructor
    functional_matrix(functional_matrix const &) = default;
    //! Move constructor
    functional_matrix(functional_matrix &&) = default;
    //! Copy assign
    functional_matrix &operator=(functional_matrix const &) = default;
    //! MOve assign
    functional_matrix &operator=(functional_matrix &&) = default;

    //! I may build a Vector from an expression!
    template <class T> 
    functional_matrix(const Expr<T> &e)
        :   m_data()
    {
        const T &et(e); // casting!
        m_rows = et.rows(); 
        m_cols = et.cols(); 
        m_data.reserve(et.rows()*et.cols());
        for(std::size_t i = 0; i < et.rows(); ++i){
            for (std::size_t j = 0; j < et.cols(); ++j){
                m_data.emplace_back(et(i,j));}}
    }

    //! Assigning an expression
    /*!
        This method is fundamental for expression template technique.
    */
    template <class T>
    functional_matrix &
    operator=(const Expr<T> &e)
    {
        const T &et(e); // casting!
        m_rows = et.rows(); 
        m_cols = et.cols();
        m_data.resize(et.rows()*et.cols());
        for(std::size_t i = 0; i < et.rows(); ++i){   
            for (std::size_t j = 0; j < et.cols(); ++j){
                m_data[j * et.rows() + i] = et(i,j);}}
        return *this;
    }

    /*!
    * @brief Returns element (i,j)
    */
    FDAGWR_TRAITS::f_type &
    operator()
    (std::size_t i, std::size_t j)
    {
        return m_data[j * m_rows + i];
    }

    /*!
    * @brief Returns element (i,j) (const version)
    */
    FDAGWR_TRAITS::f_type
    operator()
    (std::size_t i, std::size_t j)
    const
    {
        return m_data[j * m_rows + i];
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

    //! May be cast to a std::vector &
    /*!
    This way I can use all the methods of a std::vector!
    @code
    Vector a;
    std::vector<double> & av(a); // you cannot use {a} here!
    av.emplace_back(10.0);
    @endcode
    */
    operator std::vector<FDAGWR_TRAITS::f_type> const &() const { return m_data; }
    //! Non const version of casting operator
    operator std::vector<FDAGWR_TRAITS::f_type> &() { return m_data; }
  
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
    std::vector<FDAGWR_TRAITS::f_type> const &
    as_vector() 
    const
    {
        return m_data;
    }
    //! Non const version of casting operator
    std::vector<FDAGWR_TRAITS::f_type> &
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
inline 
auto
begin(functional_matrix &fm) 
    -> decltype(std::declval<std::vector<FDAGWR_TRAITS::f_type> >().begin())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>&
  return static_cast<std::vector<FDAGWR_TRAITS::f_type> &>(fm).begin();
  // If you prefer
  // return fm.as_vector().begin();
}

inline auto
end(functional_matrix &fm) 
    -> decltype(std::declval<std::vector<FDAGWR_TRAITS::f_type> >().end())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>&
  return static_cast<std::vector<FDAGWR_TRAITS::f_type> &>(fm).end();
}

inline 
auto
cbegin(functional_matrix const &fm)
  -> decltype(std::declval<std::vector<FDAGWR_TRAITS::f_type> >().cbegin())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>
  // const &
  return static_cast<std::vector<FDAGWR_TRAITS::f_type> const &>(fm).cbegin();
}

inline 
auto
cend(functional_matrix const &fm) 
    -> decltype(std::declval<std::vector<FDAGWR_TRAITS::f_type> >().cend())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>
  // const &
  return static_cast<std::vector<FDAGWR_TRAITS::f_type> const &>(fm).cend();
}

#endif  /*FDAGWR_FUNCTIONAL_MATRIX_HPP*/