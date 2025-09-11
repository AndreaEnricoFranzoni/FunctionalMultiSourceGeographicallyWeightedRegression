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
#include "functional_matrix_views.hpp"
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
    //type of the function stored and its input
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;
    //aliases for row expression
    typedef RowView<F_OBJ> RowXpr;              //non-const         
    typedef ConstRowView<F_OBJ> ConstRowXpr;    //const
    //aliases for col expression
    typedef ColView<F_OBJ> ColXpr;              //non-const 
    typedef ConstColView<F_OBJ> ConstColXpr;    //const


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
    transposing()
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
    * @brief Tranpose functional matrix
    */
    functional_matrix
    transpose()
    {
        this->transposing();
        return *this;
    }





    /*!
    * @brief Getting row idx-th, view
    */
    RowXpr
    get_row(std::size_t idx)
    {
        return {  m_data.data(), idx, m_rows, m_cols};
    }

    /*!
    * @brief Getting row idx-th, view, const
    */
    ConstRowXpr
    get_row(std::size_t idx)
    const
    {
        return {  m_data.data(), idx, m_rows, m_cols};
    }

    /*!
    * @brief Getting col idx-th, view
    */
    ColXpr
    get_col(std::size_t idx)
    {
        return {  m_data.data(), idx, m_rows};
    }

    /*!
    * @brief Getting col idx-th, view, const
    */
    ConstColXpr
    get_col(std::size_t idx)
    const
    {
        return {  m_data.data(), idx, m_rows};
    }






    /*!
    * @brief Getting row idx-th, as row vector
    */
    functional_matrix
    row(std::size_t idx)
    const
    {
        std::vector< F_OBJ > row_idx;
        row_idx.resize(m_cols);

#ifdef _OPENMP
#pragma omp parallel for shared(row_idx,idx,m_rows,m_cols,m_data) num_threads(8)
    for(std::size_t j = 0; j < m_cols; ++j)
    {
        row_idx[j] = m_data[j*m_rows + idx];
    }
#endif
        functional_matrix res(row_idx,1,m_cols);
        return res;
    }

    /*!
    * @brief Getting a specific column, as column vector
    */
    functional_matrix
    col(std::size_t idx)
    const
    {
        std::vector< F_OBJ > col_idx;
        col_idx.resize(m_rows);

#ifdef _OPENMP
#pragma omp parallel for shared(col_idx,idx,m_rows,m_data) num_threads(8)
    for(std::size_t i = 0; i < m_rows; ++i)
    {
        col_idx[i] = m_data[idx*m_rows + i];
    }
#endif
        functional_matrix res(col_idx,m_rows,1);
        return res;
    }

    /*!
    * @brief Reducing all the elements of the matrix by summation
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




/*
Block<MatrixType, Dynamic, 1> col(Index j) {
    return Block<MatrixType, Dynamic, 1>(*this, 0, j, rows(), 1);
}




// column view (contigua)
template<typename T>
struct ColView {
    T* base;
    size_t col, rows;

    T& operator[](size_t i) { return base[col * rows + i]; }
    const T& operator[](size_t i) const { return base[col * rows + i]; }

    T* begin() { return base + col * rows; }
    T* end()   { return base + (col + 1) * rows; }
};

// row view (stride = rows)
template<typename T>
struct RowView {
    T* base;
    size_t row, rows, cols;

    using iterator = StridedIterator<T>;
    iterator begin() { return iterator(base + row, rows); }
    iterator end()   { return iterator(base + cols * rows + row, rows); }
};


*/


/*
#include <cstddef>   // std::ptrdiff_t
#include <iterator>  // std::forward_iterator_tag

template<typename Ptr>
struct StridedIterator {
    using value_type        = typename std::remove_pointer<Ptr>::type;
    using difference_type   = std::ptrdiff_t;
    using pointer           = Ptr;
    using reference         = typename std::add_lvalue_reference<value_type>::type;
    using iterator_category = std::forward_iterator_tag;

    Ptr ptr;
    std::ptrdiff_t step;

    StridedIterator(Ptr p, std::ptrdiff_t s) : ptr(p), step(s) {}

    reference operator*() const { return *ptr; }

    StridedIterator& operator++() { ptr += step; return *this; }

    bool operator!=(const StridedIterator& other) const { return ptr != other.ptr; }
};


template<typename T>
struct RowView {
    T* base;
    size_t row;
    size_t rows, cols;

    using iterator = StridedIterator<T*>;
    using const_iterator = StridedIterator<const T*>;

    iterator begin() { return iterator(base + row, rows); }
    iterator end()   { return iterator(base + cols*rows + row, rows); }

    const_iterator cbegin() const { return const_iterator(base + row, rows); }
    const_iterator cend()   const { return const_iterator(base + cols*rows + row, rows); }
};


template<typename T>
struct ColView {
    T* base;
    size_t col;
    size_t rows;

    using iterator = T*;
    using const_iterator = const T*;

    iterator begin() { return base + col*rows; }
    iterator end()   { return base + (col+1)*rows; }

    const_iterator cbegin() const { return base + col*rows; }
    const_iterator cend()   const { return base + (col+1)*rows; }
};







#include <iostream>
#include <numeric>

template<typename T>
struct Matrix {
    size_t rows, cols;
    std::vector<T> data;

    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r*c) {}

    T& operator()(size_t i, size_t j) { return data[j*rows + i]; }
    const T& operator()(size_t i, size_t j) const { return data[j*rows + i]; }

    RowView<T> row(size_t i) { return { data.data(), i, rows, cols }; }
    ColView<T> col(size_t j) { return { data.data(), j, rows }; }
};

int main() {
    Matrix<double> A(3,3);
    double v = 1;
    for (size_t i=0; i<A.rows; ++i)
        for (size_t j=0; j<A.cols; ++j)
            A(i,j) = v++;

    auto r1 = A.row(1);
    std::cout << "Row 1 via cbegin: ";
    for (auto it = r1.cbegin(); it != r1.cend(); ++it)
        std::cout << *it << " ";
    std::cout << "\n";

    auto c2 = A.col(2);
    double sum = std::accumulate(c2.cbegin(), c2.cend(), 0.0);
    std::cout << "Sum of Col 2 = " << sum << "\n";
}



*/