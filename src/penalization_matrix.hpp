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


#ifndef FDAGWR_PENALIZATION_MATRIX_HPP
#define FDAGWR_PENALIZATION_MATRIX_HPP


#include "traits_fdagwr.hpp"

/*!
* @file penalization_matrix.hpp
* @brief Contains the class to define the penalization matrix used in the fmsgwr algoroithm
* @author Andrea Enrico Franzoni
*/

class penalization_matrix
{

private:
    /*!Number of functional covariates described by a basis expansion*/
    std::size_t m_q;

    /*!Number of basis for each covariate*/
    std::vector<std::size_t> m_Lj;

    /*!Number of total basis*/;
    std::size_t m_L;

    /*!Penalization matrix*/
    fdagwr_traits::Sparse_Matrix m_PenalizationMatrix;


public:
    penalization_matrix(/* args */);
    
};





#endif  /*FDAGWR_PENALIZATION_MATRIX_HPP*/