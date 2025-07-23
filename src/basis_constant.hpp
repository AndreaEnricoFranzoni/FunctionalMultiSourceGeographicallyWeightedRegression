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


#ifndef FDAGWR_CONSTANT_BASIS_HPP
#define FDAGWR_CONSTANT_BASIS_HPP


#include "basis.hpp"


/*!
* @brief Class for constant basis
*/
template< typename domain = fdagwr_traits::Domain >
class constant_basis :  public basis<domain>
{
public:
    /*!Constructor*/
    constant_basis(const fdagwr_traits::Dense_Vector & knots)    :  basis<domain>(knots)  {}

    /*!
    * @brief evaluating the system of basis basis_i-th in location location. Overriding the method
    */
    inline 
    fdagwr_traits::Dense_Matrix 
    eval_base(double location) 
    const
    override
    {   
        std::cout << "eval_base della costante" << std::endl;
        //wrap the output into a dense matrix: HA UNA RIGA, N_BASIS COLONNE
        return fdagwr_traits::Dense_Matrix::Ones(1,1);
    }
};

#endif  /*FDAGWR_CONSTANT_BASIS_HPP*/