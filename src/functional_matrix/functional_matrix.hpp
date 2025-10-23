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


#ifndef FUNCTIONAL_MATRIX_HPP
#define FUNCTIONAL_MATRIX_HPP


#include "functional_matrix_expression_wrapper.hpp"
#include "functional_matrix_views.hpp"
#include "functional_matrix_utils.hpp"
#include <utility>
#include <vector>
#include <cassert> 
#include <Eigen/Dense>
#include <algorithm>



/*!
* @file functional_matrix.hpp
* @brief Contains the definition of a dense matrix containing univariate 1D domain std::function objects
* @author Andrea Enrico Franzoni
*/




/*!
* @class functional_matrix
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Class for dense matrices storing, column-wise, univariate 1D domain std::function objects
* @details Static polymorphism: deriving from a expression for expression templates
* @note Functions are stored column-wise
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class functional_matrix : public Expr< functional_matrix<INPUT,OUTPUT>, INPUT, OUTPUT >
{
    /*!std::function object stored*/
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    /*!std::function input type*/
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;
    /*!Alias for row expression, non-const*/
    typedef RowView<F_OBJ> RowXpr;              
    /*!Alias for row expression, const*/    
    typedef RowView<const F_OBJ> ConstRowXpr; 
    /*!Alias for column expression, non-const*/
    typedef ColView<F_OBJ> ColXpr;       
    /*!Alias for column expression, const*/
    typedef ColView<const F_OBJ> ConstColXpr;   


private:
    /*!Number of rows*/
    std::size_t m_rows;
    /*!Number of cols*/
    std::size_t m_cols;
    /*!Container for the std::function. The storage order is column-wise*/
    std::vector< F_OBJ > m_data;

public:
    /*!
    * @brief Default constructor
    */
    functional_matrix() = default;

    /*!
    * @brief Constructor
    * @param fm vector of std::function objects
    * @param n_rows number of rows
    * @param n_cols number of columns
    */
    functional_matrix(std::vector< F_OBJ > const &fm,
                      std::size_t n_rows,
                      std::size_t n_cols)
                :   m_rows(n_rows), m_cols(n_cols), m_data{fm} 
                {
                    //cheack input consistency
                    assert((void("Number of rows times number of cols has to be equal to the number of stored functions"), m_rows * m_cols == m_data.size()));
                }

    /*!
    * @brief Constructor with move semantic
    * @param fm vector of std::function objects
    * @param n_rows number of rows
    * @param n_cols number of columns
    */
    functional_matrix(std::vector< F_OBJ > &&fm,
                      std::size_t n_rows,
                      std::size_t n_cols) 
                :   m_rows(n_rows), m_cols(n_cols), m_data{std::move(fm)}
                {
                    //cheack input consistency
                    assert((void("Number of rows times number of cols has to be equal to the number of stored functions"), m_rows * m_cols == m_data.size()));
                }

    /*!
    * @brief Constructor that initializes all the matrix elements with the same std::function object
    * @param n_rows number of rows
    * @param n_cols number of columns
    * @param f value used to initialize all the matrices elements (default is unit function)
    */
    functional_matrix(std::size_t n_rows, std::size_t n_cols, F_OBJ f = [](F_OBJ_INPUT){return static_cast<OUTPUT>(1);}) : m_rows(n_rows), m_cols(n_cols), m_data(m_rows*m_cols,f)   {};

    /*!
    * @brief Copy constructor
    */
    functional_matrix(functional_matrix const &) = default;

    /*!
    * @brief Move constructor
    */
    functional_matrix(functional_matrix &&) = default;

    /*!
    * @brief Copy assignment
    */
    functional_matrix &operator=(functional_matrix const &) = default;

    /*!
    * @brief Move assignment
    */
    functional_matrix &operator=(functional_matrix &&) = default;

    /*!
    * @brief Constructor that builds a functional_matrix from an Expr 
    * @tparam T template param indicating the derived type from expression from which the cast is done
    * @param e expression from which constructing a functional_matrix
    * @details necessary for ETs design
    */
    template <class T> 
    functional_matrix(const Expr<T,INPUT,OUTPUT> &e)
        :   m_data()
    {
        //casting
        const T &et(e);                                     
        m_rows = et.rows(); 
        m_cols = et.cols(); 
        m_data.reserve(et.size());
        //storing column-wise
        for (std::size_t j = 0; j < et.cols(); ++j){       
            for(std::size_t i = 0; i < et.rows(); ++i){
                m_data.emplace_back(et(i,j));}}
    }

    /*!
    * @brief Copy assignment from an Expr
    * @tparam T template param indicating the derived type from expression from which the cast is done
    * @param e expression from which constructing a functional_matrix
    * @details necessary for ETs design
    */
    template <class T>
    functional_matrix &
    operator=(const Expr<T,INPUT,OUTPUT> &e)
    {
        //casting
        const T &et(e); 
        m_rows = et.rows(); 
        m_cols = et.cols();
        m_data.resize(et.size());
        //copying column-wise
        for(std::size_t i = 0; i < et.rows(); ++i){   
            for (std::size_t j = 0; j < et.cols(); ++j){
                m_data[j * et.rows() + i] = et(i,j);}}
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
        return m_data[j * m_rows + i];
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
        return m_data[j * m_rows + i];
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
        return m_rows*m_cols;
    }

    /*!
    * @brief View of the idx-th row, non-const
    * @param idx the row index
    * @return a view to the idx-th row
    */
    RowXpr
    row(std::size_t idx)
    {
        return {  m_data.data(), idx, m_rows, m_cols};
    }

    /*!
    * @brief View of the idx-th row, const
    * @param idx the row index
    * @return a view to the idx-th row, read-only
    */
    ConstRowXpr
    row(std::size_t idx)
    const
    {
        return {  m_data.data(), idx, m_rows, m_cols};
    }

    /*!
    * @brief View of the idx-th column, non-const
    * @param idx the column index
    * @return a view to the idx-th column
    */
    ColXpr
    col(std::size_t idx)
    {
        return {  m_data.data(), idx, m_rows};
    }

    /*!
    * @brief View of the idx-th column, const
    * @param idx the column index
    * @return a view to the idx-th column, read-only
    */
    ConstColXpr
    col(std::size_t idx)
    const
    {
        return {  m_data.data(), idx, m_rows};
    }

    /*!
    * @brief Substituting the idx-th row
    * @param new_row a vector of std::function, containing the row
    * @param idx index of the row to be substituted
    */
    void
    row_sub(const std::vector< F_OBJ > &new_row, std::size_t idx)
    {
        assert(new_row.size() == m_cols);
        assert(idx < m_rows); 

#ifdef _OPENMP
#pragma omp parallel for shared(new_row,idx,m_data,m_rows) num_threads(8)
#endif
        for(std::size_t j = 0; j < new_row.size(); ++j){
            m_data[j * m_rows + idx] = new_row[j];}
    }

    /*!
    * @brief Substituting the idx-th column
    * @param new_col a vector of std::function, containing the column 
    * @param idx index of the column to be substituted
    */
    void
    col_sub(const std::vector< F_OBJ > &new_col, std::size_t idx)
    {
        assert(new_col.size() == m_rows);
        assert(idx < m_cols); 

#ifdef _OPENMP
#pragma omp parallel for shared(new_col,idx,m_data,m_rows) num_threads(8)
#endif
        for(std::size_t i = 0; i < new_col.size(); ++i){
            m_data[idx * m_rows + i] = new_col[i];}
    }

    /*!
    * @brief Transposing the functional matrix
    */
    void
    transposing()
    {
        if(m_rows!=static_cast<std::size_t>(1) && m_cols!=static_cast<std::size_t>(1))
        {
            std::vector< F_OBJ > temp;
            temp.resize(this->size());

#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(m_rows,m_cols,temp,m_data) num_threads(8)
#endif
            for (std::size_t i = 0; i < m_rows; ++i)
            {
                for (std::size_t j = 0; j < m_cols; ++j)
                {
                    temp[i*m_cols + j] = m_data[j*m_rows + i];      //swaps appropriately
                }
            }
            //swap them
            std::swap(m_data,temp);
            temp.clear();
        }
        //swapping number of rows and cols
        std::swap(m_rows,m_cols);
    }

    /*!
    * @brief Tranpost of the functional matrix 
    * @return a copy of the transpost 
    * @note does not transpose the original object
    */
    functional_matrix<INPUT,OUTPUT>
    transpose()
    const
    {
        functional_matrix<INPUT,OUTPUT> transpost_fm(*this);
        transpost_fm.transposing();

        return transpost_fm;
    }

    /*!
    * @brief Reducing all the elements of the matrix by summation
    * @return the sum function of all the elements in the matrix
    */
    F_OBJ
    reduce()
    const
    {
        //null function: starting point for reduction
        F_OBJ f_null = [](F_OBJ_INPUT x){return static_cast<OUTPUT>(0);};
        //function that operates summation within two functions
        std::function<F_OBJ(F_OBJ,F_OBJ)> f_sum = [](F_OBJ f1, F_OBJ f2){return [f1,f2](F_OBJ_INPUT x){return f1(x)+f2(x);};};
        //reduction
        return std::reduce(this->m_data.cbegin(),this->m_data.cend(),f_null,f_sum);
    }
    
    /*!
    * @brief Casting operator to a std::vector &, const version
    * @return a const reference to a std::vector of std::function, containing the function stored into the matrix
    * @code
    * functional_matrix<INPUT,OUTPUT>  fm;
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
    * functional_matrix<INPUT,OUTPUT>  fm;
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
* @param fm a reference to a functional_matrix object
* @return an iterator to the begin of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
begin(functional_matrix<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().begin())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).begin();
}

/*!
* @brief Function to use range for loops over the stored functions, end iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a reference to a functional_matrix object
* @return an iterator to the end of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
end(functional_matrix<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().end())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).end();
}

/*!
* @brief Function to use range for loops over the stored functions, const begin iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a const reference to a functional_matrix object
* @return a constant iterator to the begin of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cbegin(functional_matrix<INPUT,OUTPUT> const &fm)
  -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cbegin())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>const &
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cbegin();
}

/*!
* @brief Function to use range for loops over the stored functions, const end iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a const reference to a functional_matrix object
* @return a constant iterator to the end of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cend(functional_matrix<INPUT,OUTPUT> const &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cend())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>const &
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cend();
}

/*!
* @brief Function to convert a matrix of scalar into a functional_matrix containing constant functions
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param Ms const reference to an Eigen::Matrix< OUTPUT, Eigen::Dynamic, Eigen::Dynamic >
* @return a functional_matrix containing constant functions with the values in Ms
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline  
functional_matrix<INPUT,OUTPUT>
scalar_to_functional(const Eigen::Matrix< OUTPUT, Eigen::Dynamic, Eigen::Dynamic > &Ms)
{
    //type stored by the functional matrix
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    //input type of the elements of the functional matrix
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;
    //function to convert a scalar into a constant function
    std::function< F_OBJ(const OUTPUT &) > scalar_to_const_func = [](const OUTPUT &a){  return [a](F_OBJ_INPUT x){return static_cast<OUTPUT>(a);};};

    //reshaping the scalar matrix (.reshaped() aligns column-wise!) in order to iterate along it
    auto R = Ms.reshaped(); 
    //container to store the transformation from scalar to f
    std::vector< F_OBJ > f_vec;
    f_vec.resize(Ms.size());
    //transforming each element of S into a constant function
    std::transform(R.cbegin(),R.cend(),f_vec.begin(),scalar_to_const_func);      
    //constructing the functional matrix
    functional_matrix<INPUT,OUTPUT> Mf(f_vec,Ms.rows(),Ms.cols());

    return Mf;
}

#endif  /*FUNCTIONAL_MATRIX_HPP*/