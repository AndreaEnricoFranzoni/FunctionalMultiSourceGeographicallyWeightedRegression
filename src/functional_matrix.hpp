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


#ifndef FUNCTIONAL_MATRIX_HPP
#define FUNCTIONAL_MATRIX_HPP


#include "functional_matrix_expression_wrapper.hpp"
#include "functional_matrix_utils.hpp"
#include <utility>
#include <vector>
#include <cassert>


//! A class for matrices of functions: STORING COLUMN WISE
/*!
  It is build around std::vector<double> and indeed
  std::vector:double> is the only variable member of the class.
 */
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class functional_matrix : public Expr< functional_matrix<INPUT,OUTPUT>, INPUT, OUTPUT >
{
//type of the function stored
using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

private:
    /*!Number of rows*/
    std::size_t m_rows;
    /*!Number of cols*/
    std::size_t m_cols;
    /*!Matrix of functions*/
    std::vector< F_OBJ > m_data;

public:
    //! default constructor
    functional_matrix() = default;
    //! A vector may be converted to a Vector
    functional_matrix(std::vector< F_OBJ > const &fm,
                      std::size_t n_rows,
                      std::size_t n_cols)
                :   m_rows(n_rows), m_cols(n_cols), m_data{fm} 
                {
                    //cheack input consistency
                    assert((void("Number of rows times number of cols has to be equal to the number of stored functions"), m_rows * m_cols == m_data.size()));
                }
    //! A vector may also be moved in a Vector
    functional_matrix(std::vector< F_OBJ > &&fm,
                      std::size_t n_rows,
                      std::size_t n_cols) 
                :   m_rows(n_rows), m_cols(n_cols), m_data{std::move(fm)}
                {
                    //cheack input consistency
                    assert((void("Number of rows times number of cols has to be equal to the number of stored functions"), m_rows * m_cols == m_data.size()));
                }
    //! Construct a Vector of n elements initialised by value   //DA METTERE IL DEFAULT f=[](const INPUT &){return static_cast<OUTPUT>(1.0);} 
    functional_matrix(std::size_t n_rows, std::size_t n_cols, F_OBJ f = [](F_OBJ_INPUT){return static_cast<OUTPUT>(1);}) : m_rows(n_rows), m_cols(n_cols), m_data(m_rows*m_cols,f)   {};
    //! Copy constructor
    functional_matrix(functional_matrix const &) = default;
    //! Move constructor
    functional_matrix(functional_matrix &&) = default;
    //! Copy assign
    functional_matrix &operator=(functional_matrix const &) = default;
    //! Move assign
    functional_matrix &operator=(functional_matrix &&) = default;

    //! I may build a Vector from an expression!
    template <class T> 
    functional_matrix(const Expr<T,INPUT,OUTPUT> &e)
        :   m_data()
    {
        const T &et(e); // casting!
        m_rows = et.rows(); 
        m_cols = et.cols(); 
        m_data.reserve(et.size());
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
    operator=(const Expr<T,INPUT,OUTPUT> &e)
    {
        const T &et(e); // casting!
        m_rows = et.rows(); 
        m_cols = et.cols();
        m_data.resize(et.size());
        for(std::size_t i = 0; i < et.rows(); ++i){   
            for (std::size_t j = 0; j < et.cols(); ++j){
                m_data[j * et.rows() + i] = et(i,j);}}
        return *this;
    }

    /*!
    * @brief Returns element (i,j)
    */
    F_OBJ &
    operator()
    (std::size_t i, std::size_t j)
    {
        return m_data[j * m_rows + i];
    }

    /*!
    * @brief Returns element (i,j) (const version)
    */
    F_OBJ
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

    /*!
    * @brief Number of elements
    */
    std::size_t
    size() 
    const
    {
        return m_rows*m_cols;
    }

    /*!
    * @brief Transposing the functional matrix
    */
    void
    transpose()
    {
        if(m_rows!=static_cast<std::size_t>(1) && m_cols!=static_cast<std::size_t>(1))
        {
            std::vector< F_OBJ > temp;
            temp.resize(this->size());

#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(m_rows,m_cols,temp,m_data) num_threads(8)
            for (std::size_t i = 0; i < m_rows; ++i)
            {
                for (std::size_t j = 0; j < m_cols; ++j)
                {
                    temp[i*m_cols + j] = m_data[j*m_rows + i];      //swaps appropriately
                }
            }
#endif
            //swap them
            std::swap(m_data,temp);
            temp.clear();
        }

        std::swap(m_rows,m_cols);
    }

    /*!
    * @brief Getting a specific row, as row vector
    */
    functional_matrix
    get_row(std::size_t idx)
    const
    {
        std::vector< F_OBJ > row;
        row.resize(m_cols);

#ifdef _OPENMP
#pragma omp parallel for shared(row,idx,m_rows,m_cols,m_data) num_threads(8)
    for(std::size_t j = 0; j < m_cols; ++j)
    {
        row[j] = m_data[j*m_rows + idx];
    }
#endif
        functional_matrix res(row,1,m_cols);
        return res;
    }

    /*!
    * @brief Getting a specific column, as column vector
    */
    functional_matrix
    get_col(std::size_t idx)
    const
    {
        std::vector< F_OBJ > col;
        col.resize(m_rows);

#ifdef _OPENMP
#pragma omp parallel for shared(col,idx,m_rows,m_data) num_threads(8)
    for(std::size_t i = 0; i < m_rows; ++i)
    {
        col[i] = m_data[idx*m_rows + i];
    }
#endif
        functional_matrix res(col,m_rows,1);
        return res;
    }

    /*!
    * @brief Reducing all the elements of the matrix by summation
    */
    F_OBJ
    reduce()
    const
    {
/*
        F_OBJ reduction = [](F_OBJ_INPUT x){return static_cast<OUTPUT>(0);};

        for(std::size_t i = 0; i < this->size(); ++i)
        {
            reduction = [reduction,i,this](F_OBJ_INPUT x){return reduction(x) + this->m_data[i](x);};
        }
        return reduction;
*/
        F_OBJ f_null = [](F_OBJ_INPUT x){return static_cast<OUTPUT>(0);};
        std::function<F_OBJ(F_OBJ,F_OBJ)> f_sum = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)+f2(x);};};

        return std::reduce(this->m_data.cbegin(),
                           this->m_data.cend(),
                           f_null,
                           f_sum);
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
begin(functional_matrix<INPUT,OUTPUT> &fm) 
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
end(functional_matrix<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().end())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).end();
}

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cbegin(functional_matrix<INPUT,OUTPUT> const &fm)
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
cend(functional_matrix<INPUT,OUTPUT> const &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cend())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>
  // const &
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cend();
}

#endif  /*FUNCTIONAL_MATRIX_HPP*/