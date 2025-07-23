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


#include "traits_fdagwr.hpp"



template< typename T = double >
class functional_matrix
{

using function_type = std::vector<std::function<T(T)>>;

private:
    std::size_t m_rows;

    std::size_t m_cols;

    std::vector<function_type> m_matrix;

    static inline T default_f(T el) {   return static_cast<T>(1);};


public:
    functional_matrix(std::size_t m,
                      std::size_t n,
                      const function_type& f)
        :
            m_rows(m),
            m_cols(n),
            m_matrix(m_rows*m_cols,f)
        {}


    template<typename... Args>
    functional_matrix(std::size_t m,
                      std::size_t n,
                      Args&&...args)
        :   m_rows(m), m_cols(n)
        {
            m_matrix.reserve(m_rows*m_cols);
            m_matrix.emplace_back(std::forward<Args>(args),...);

            if (sizeof...(args) != (m_rows*m_cols)){
                std::size_t number_of_func_passed = sizeof...(args);

                for (std::size_t i = number_of_func_passed; i < m_rows*m_cols; ++i){
                    m_matrix.push_back(this->default_f);}}
    }

};


#endif  /*FDAGWR_FUNCTIONAL_MATRIX_HPP*/