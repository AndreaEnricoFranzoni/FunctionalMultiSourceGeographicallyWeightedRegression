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

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fgwr_fms_esc final : public fgwr<INPUT,OUTPUT>
{
private:
    /*!Functional response*/
    functional_matrix<INPUT,OUTPUT> m_y;
    /*!Basis for response*/
    functional_matrix<INPUT,OUTPUT> m_phi;
    /*!Coefficients of the basis expansion for response*/
    FDAGWR_TRAITS::Dense_Matrix m_c;

    /*!Functional stationary covariates*/
    functional_matrix<INPUT,OUTPUT> m_Xc;
    /*!Functional weights for stationary covariates*/
    functional_matrix<INPUT,OUTPUT> m_Wc;
    /*!Penalization on the stationary covariates*/
    FDAGWR_TRAITS::Dense_Matrix m_Rc;
    /*!Basis for stationary covariates regressors*/
    functional_matrix<INPUT,OUTPUT> m_omega;
    /*!Coefficients of the basis expansion for stationary covariates regressors: TO BE COMPUTED*/
    FDAGWR_TRAITS::Dense_Matrix m_bc;

    /*!Functional event-dependent covariates*/
    functional_matrix<INPUT,OUTPUT> m_Xe;
    /*!Functional weights for event-dependent covariates*/
    std::vector< functional_matrix<INPUT,OUTPUT> > m_We;
    /*!Penalization on the event-dependent covariates*/
    FDAGWR_TRAITS::Dense_Matrix m_Re;
    /*!Basis for event-dependent covariates regressors*/
    functional_matrix<INPUT,OUTPUT> m_theta;
    /*!Coefficients of the basis expansion for event-dependent covariates regressors: TO BE COMPUTED*/
    FDAGWR_TRAITS::Dense_Matrix m_be;

    /*!Functional station-dependent covariates*/
    functional_matrix<INPUT,OUTPUT> m_Xs;
    /*!Functional weights for station-dependent covariates*/
    std::vector< functional_matrix<INPUT,OUTPUT> > m_Ws;
    /*!Penalization on the station-dependent covariates*/
    FDAGWR_TRAITS::Dense_Matrix m_Rs;
    /*!Basis for station-dependent covariates regressors*/
    functional_matrix<INPUT,OUTPUT> m_psi;
    /*!Coefficients of the basis expansion for station-dependent covariates regressors: TO BE COMPUTED*/
    FDAGWR_TRAITS::Dense_Matrix m_bs;


public:
    /*!
    * @brief Constructor
    */ 
    fgwr_fms_esc(int number_threads)
      :
          fgwr<INPUT,OUTPUT>(number_threads)
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