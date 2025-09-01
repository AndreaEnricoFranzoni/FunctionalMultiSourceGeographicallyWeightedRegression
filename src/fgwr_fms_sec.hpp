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


#ifndef FGWR_FMS_SEC_ALGO_HPP
#define FGWR_FMS_SEC_ALGO_HPP

#include "fgwr.hpp"


/// Computes the jacobian by finite differences.
class fgwr_fms_sec final : public fgwr
{
private:
    double m_a;
    int m_b;

public:
    /// Constructor.
    fgwr_fms_sec(double a, int b): m_a(a), m_b(b) {}

    /// Override of the base class method.
    inline 
    void 
    compute() 
    const 
    override
    {
        std::cout << "SEC: a: " << m_a << ", b: " << m_b << std::endl;
    }

};

#endif  /*FGWR_FMS_SEC_ALGO_HPP*/