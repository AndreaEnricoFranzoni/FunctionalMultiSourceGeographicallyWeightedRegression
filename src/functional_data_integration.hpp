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


#ifndef FDAGWR_FUNCTIONAL_DATA_INTEGRATION_HPP
#define FDAGWR_FUNCTIONAL_DATA_INTEGRATION_HPP


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include "concepts_fdagwr.hpp"

#include "mesh.hpp"
#include "numerical_integration.hpp"
#include "Adams_rule.hpp"
#include "QuadratureRuleAdaptive.hpp"
using namespace Geometry;
using namespace apsc::NumericalIntegration;



class fd_integration
{

using integrand_type = FunPoint;

private:
    /*!Integration domain*/
    Domain1D m_integration_domain;
    /*!Integration mesh*/
    Mesh1D m_integration_mesh;

    
public:
    fd_integration(double a, double b, int intervals):
        m_integration_domain(a,b), m_integration_mesh(m_integration_domain,intervals)
        {}

    double integrate(const integrand_type &f){
        Quadrature sa(QuadratureRuleAdaptive<Trapezoidal>(1.e-6, 10000), m_integration_mesh);
        return sa.apply(f);}
};


#endif  /*FDAGWR_FUNCTIONAL_DATA_INTEGRATION_HPP*/