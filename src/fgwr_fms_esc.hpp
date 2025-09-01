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


#ifndef FGWR_FMS_ESC_ALGO_HPP
#define FGWR_FMS_ESC_ALGO_HPP

#include "fgwr.hpp"


class fgwr_fms_esc final : public fgwr
{
private:

public:
    /*!
    * @brief Constructor
    */ 
    fgwr_fms_esc(int number_threads)
      :
          fgwr(number_threads)
          {}

    /*!
    * @brief Override of the base class method
    */ 
    inline 
    void 
    compute() 
    const 
    override
    {
      std::cout << "ESC, with OMP threads: " << this->number_threads() << std::endl;
    }

};

#endif  /*FGWR_FMS_ESC_ALGO_HPP*/