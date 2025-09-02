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


#ifndef FGWR_FOS_ALGO_HPP
#define FGWR_FOS_ALGO_HPP

#include "fgwr.hpp"


template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fgwr_fos final : public fgwr<INPUT,OUTPUT>
{
private:

public:
    /*!
    * @brief Constructor
    */ 
    fgwr_fos(INPUT a,
             INPUT b,
             int n_intervals,
             int number_threads)
        :
            fgwr<INPUT,OUTPUT>(a,b,n_intervals,number_threads)
            {}

    /*!
    * @brief Override of the base class method
    */ 
    inline 
    void 
    compute() 
    const 
    override
    {}

};

#endif  /*FGWR_FOS_ALGO_HPP*/