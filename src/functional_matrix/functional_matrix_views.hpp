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


#ifndef FUNCTIONAL_MATRIX_VIEWS_HPP
#define FUNCTIONAL_MATRIX_VIEWS_HPP


#include <type_traits>
#include <cstddef>   
#include <iterator>  


/*!
* @file functional_matrix_views.hpp
* @brief Contains the views for accessing rows and columns of a dense matrix containing univariate 1D domain std::function objects
* @author Andrea Enrico Franzoni
*/


/*!
* @struct StridedIterator
* @brief Struct to construct an iterator with stride (necessary for getting the rows if storing as col-major, adding the offset)
* @tparam Ptr iterator
*/
template<typename Ptr>
struct StridedIterator 
{
    /*!Type pointed*/
    using value_type        = typename std::remove_pointer<Ptr>::type;
    /*!Difference iteratpr*/
    using difference_type   = std::ptrdiff_t;
    /*!Iterator type*/
    using pointer           = Ptr;
    /*!Reference to the type pointed*/
    using reference         = typename std::add_lvalue_reference<value_type>::type;
    /*!Forward iterator type*/
    using iterator_category = std::forward_iterator_tag;

    /*!Iterator*/
    Ptr ptr;
    /*!Offset*/
    std::ptrdiff_t step;

    /*!
    * @brief Constructor
    * @param p iterator
    * @param s offset added to the iterator
    */
    StridedIterator(Ptr p, std::ptrdiff_t s) : ptr(p), step(s) {}

    /*!
    * @brief Getter for the pointer object
    * @return a reference to the object pointed by the iterator
    */
    reference operator*() const { return *ptr; }

    /*!
    * @brief Getter for the iterator with offset
    * @return the iterator with offset
    */
    StridedIterator& operator++() { ptr += step; return *this; }

    /*!
    * @brief Check if two iterators are pointing to the same object
    * @param other an iterator with offest
    * @return true if the two iterators are pointing to the same object, false otherwise
    */
    bool operator!=(const StridedIterator& other) const { return ptr != other.ptr; }
};


/*!
* @struct RowView
* @brief Row-view: accessing to a row of a dense matrix of callable object with a pair of iterators
* @tparam T the object type stored in the matrix
* @note since the matrices store object column-wise, the iterator with offset has to be used
*/
template<typename T>
struct RowView 
{
    /*!Iterator*/
    T* base;
    /*!Row that has to be accessed*/
    std::size_t row;
    /*!Total number of rows and columns*/
    std::size_t rows, cols;

    /*!
    * @brief Getter for the start of the row, non-const verion
    * @return a pointer to the beginning of the row
    */
    StridedIterator<T*> begin() {   return StridedIterator<T*>(base + row, rows); }

    /*!
    * @brief Getter for the end of the row, non-const verion
    * @return a pointer to the end of the row
    */
    StridedIterator<T*> end()   {   return StridedIterator<T*>(base + cols*rows + row, rows); }

    /*!
    * @brief Getter for the start of the row, const verion
    * @return a const pointer to the beginning of the row
    */
    StridedIterator<const T*> begin() const {   return StridedIterator<const T*>(base + row, rows); }

    /*!
    * @brief Getter for the end of the row, const verion
    * @return a const pointer to the end of the row
    */
    StridedIterator<const T*> end()   const {   return StridedIterator<const T*>(base + cols*rows + row, rows); }

    /*!
    * @brief Getter for the start of the row, const verion
    * @return a const pointer to the beginning of the row
    */
    StridedIterator<const T*> cbegin() const {  return begin(); }

    /*!
    * @brief Getter for the end of the row, const verion
    * @return a const pointer to the end of the row
    */
    StridedIterator<const T*> cend()   const {  return end(); }
};


/*!
* @struct ColView
* @brief Column-view: accessing to a column of a dense matrix of callable object with a pair of iterators
* @tparam T the object type stored in the matrix
*/
template<typename T>
struct ColView 
{
    /*!Iterator*/
    T* base;
    /*!Column that has to be accessed*/
    std::size_t col;
    /*!Total number of rows in the matrix*/
    std::size_t rows;

    /*!
    * @brief Getter for the start of the column, non-const verion
    * @return a pointer to the beginning of the column
    */
    T* begin() { return base + col*rows; }

    /*!
    * @brief Getter for the end of the column, non-const verion
    * @return a pointer to the end of the column
    */
    T* end()   { return base + (col+1)*rows; }

    /*!
    * @brief Getter for the start of the column, const verion
    * @return a const pointer to the beginning of the column
    */
    const T* begin() const { return base + col*rows; }

    /*!
    * @brief Getter for the end of the column, const verion
    * @return a const pointer to the end of the column
    */
    const T* end()   const { return base + (col+1)*rows; }

    /*!
    * @brief Getter for the start of the column, const verion
    * @return a const pointer to the beginning of the column
    */
    const T* cbegin() const { return begin(); }

    /*!
    * @brief Getter for the end of the column, const verion
    * @return a const pointer to the end of the column
    */
    const T* cend()   const { return end(); }
};

#endif  /*FUNCTIONAL_MATRIX_VIEWS_HPP*/