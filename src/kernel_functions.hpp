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

#ifndef FDAGWR_KERNEL_FUNC_HPP
#define FDAGWR_KERNEL_FUNC_HPP

#include <cmath>

#include "traits_fdagwr.hpp"

/*!
* @file kernel_functions.hpp
* @brief Containing the definitions of the different kernel functions
* @author Andrea Enrico Franzoni
*/


/*!
* @brief Template function for the gaussian kernel
* @tparam T type of the data
* @param distance a double indicating the distance
* @param bandwith kernel bandawith
* @return the evaluation of the kernel in the distance
*/
template<typename T>
T gaussian_kernel(T distance, T bandwith)
{
        return std::exp((-1/2)*std::pow((distance/bandwith),static_cast<int>(2)));
};

#endif /*FDAGWR_KERNEL_FUNC_HPP*/