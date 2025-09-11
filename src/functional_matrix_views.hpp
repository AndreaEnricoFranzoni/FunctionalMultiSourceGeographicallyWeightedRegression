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


#ifndef FUNCTIONAL_MATRIX_VIEWS_HPP
#define FUNCTIONAL_MATRIX_VIEWS_HPP


#include <type_traits>
#include <cstddef>   
#include <iterator>  


/*!
* @brief Struct to construct an iterator with stride (for the row-major)
*/
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


/*!
* @brief Row-view
*/
template<typename T>
struct RowView {
    T* base;
    std::size_t row;
    std::size_t rows, cols;

    using iterator = StridedIterator<T*>;

    iterator begin() { return iterator(base + row, rows); }
    iterator end()   { return iterator(base + cols*rows + row, rows); }
};


/*!
* @brief Const row-view
*/
template<typename T>
struct ConstRowView {
    const T* base;
    std::size_t row;
    std::size_t rows, cols;

    using const_iterator = StridedIterator<const T*>;

    const_iterator cbegin() const { return iterator(base + row, rows); }
    const_iterator cend()   const { return iterator(base + cols*rows + row, rows); }
};


/*!
* @brief Col-view
*/
template<typename T>
struct ColView {
    T* base;
    std::size_t col;
    std::size_t rows;

    using iterator = T*;

    iterator begin() { return base + col*rows; }
    iterator end()   { return base + (col+1)*rows; }
};


/*!
* @brief Const col-view
*/
template<typename T>
struct ColView {
    const T* base;
    std::size_t col;
    std::size_t rows;

    using const_iterator = const T*;

    iterator begin() { return base + col*rows; }
    iterator end()   { return base + (col+1)*rows; }
};


#endif  /*FUNCTIONAL_MATRIX_VIEWS_HPP*/