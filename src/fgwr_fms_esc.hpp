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
    /*!Functional response (nx1)*/
    functional_matrix<INPUT,OUTPUT> m_y;
    /*! @brief Basis for response (nxLy)
    * @todo CAPIRE LE DIMENSIONI E LA STRUTTURA + MATRICE DI FUNZIONI SPARSA
    */
    functional_matrix<INPUT,OUTPUT> m_phi;
    /*!Coefficients of the basis expansion for response ()*/
    FDAGWR_TRAITS::Dense_Matrix m_c;

    /*!Functional stationary covariates (n x qc)*/
    functional_matrix<INPUT,OUTPUT> m_Xc;
    /*!Functional weights for stationary covariates (n elements of diagonal n x n)*/
    functional_matrix_diagonal<INPUT,OUTPUT> m_Wc;
    /*!Scalar matrix with the penalization on the stationary covariates (sparse Lc x Lc, where Lc is the sum of the basis of each C covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Rc;
    /*!Basis for stationary covariates regressors (sparse qC x Lc)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_omega;
    /*!Coefficients of the basis expansion for stationary covariates regressors: TO BE COMPUTED*/
    FDAGWR_TRAITS::Dense_Matrix m_bc;

    /*!Functional event-dependent covariates (n x qe)*/
    functional_matrix<INPUT,OUTPUT> m_Xe;
    /*!Functional weights for event-dependent covariates (n elements of diagonal n x n)*/
    std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > m_We;
    /*!Scalar matrix with the penalization on the event-dependent covariates (sparse Le x Le, where Le is the sum of the basis of each E covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Re;
    /*! @brief Basis for event-dependent covariates regressors (sparse qE x Le)
    * @todo MATRICE DI FUNZIONI SPARSA
    */
    functional_matrix<INPUT,OUTPUT> m_theta;
    /*!Coefficients of the basis expansion for event-dependent covariates regressors: TO BE COMPUTED*/
    FDAGWR_TRAITS::Dense_Matrix m_be;

    /*!Functional station-dependent covariates (n x qs)*/
    functional_matrix<INPUT,OUTPUT> m_Xs;
    /*!Functional weights for station-dependent covariates (n elements of diagonal n x n)*/
    std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > m_Ws;
    /*!Scalar matrix with the penalization on the station-dependent covariates (sparse Ls x Ls, where Ls is the sum of the basis of each S covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Rs;
    /*! @brief Basis for station-dependent covariates regressors (sparse qS x Ls)
    * @todo MATRICE DI FUNZIONI SPARSA
    */
    functional_matrix<INPUT,OUTPUT> m_psi;
    /*!Coefficients of the basis expansion for station-dependent covariates regressors: TO BE COMPUTED*/
    FDAGWR_TRAITS::Dense_Matrix m_bs;


public:
    /*!
    * @brief Constructor
    */
    template<typename FUNC_MATRIX_OBJ, 
             typename FUNC_SPARSE_MATRIX_OBJ,
             typename FUNC_DIAG_MATRIX_OBJ, 
             typename FUNC_DIAG_MATRIX_VEC_OBJ, 
             typename SCALAR_MATRIX_OBJ, 
             typename SCALAR_SPARSE_MATRIX_OBJ> 
    fgwr_fms_esc(FUNC_MATRIX_OBJ &&y,
                 SCALAR_MATRIX_OBJ &&c,
                 FUNC_MATRIX_OBJ &&Xc,
                 FUNC_DIAG_MATRIX_OBJ &&Wc,
                 SCALAR_SPARSE_MATRIX_OBJ &&Rc,
                 FUNC_SPARSE_MATRIX_OBJ &&omega,
                 FUNC_MATRIX_OBJ &&Xe,
                 FUNC_DIAG_MATRIX_VEC_OBJ &&We,
                 SCALAR_SPARSE_MATRIX_OBJ &&Re,
                 FUNC_MATRIX_OBJ &&Xs,
                 FUNC_DIAG_MATRIX_VEC_OBJ &&Ws,
                 SCALAR_SPARSE_MATRIX_OBJ &&Rs,
                 INPUT a,
                 INPUT b,
                 int n_intervals_integration,
                 int number_threads)
        :
            fgwr<INPUT,OUTPUT>(a,b,n_intervals_integration,number_threads),
            m_y{std::forward<FUNC_MATRIX_OBJ>(y)},
            m_c{std::forward<SCALAR_MATRIX_OBJ>(c)},
            m_Xc{std::forward<FUNC_MATRIX_OBJ>(Xc)},
            m_Wc{std::forward<FUNC_DIAG_MATRIX_OBJ>(Wc)},
            m_Rc{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rc)},
            m_omega{std::forward<FUNC_SPARSE_MATRIX_OBJ>(omega)},
            m_Xe{std::forward<FUNC_MATRIX_OBJ>(Xe)},
            m_We{std::forward<FUNC_DIAG_MATRIX_VEC_OBJ>(We)},
            m_Re{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Re)},
            m_Xs{std::forward<FUNC_MATRIX_OBJ>(Xs)},
            m_Ws{std::forward<FUNC_DIAG_MATRIX_VEC_OBJ>(Ws)},
            m_Rs{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rs)}
            {}

    /*
    *   !!!!!!!!!!!NB: RITORNARE LE MATRICI SCALARI SIA COME SCALARI CHE COME MATRICI DI FUNZIONI COSTANTI!!!!!!!
    *   !!!!!!!!!!! NON HO ANCORA SCRITTO UN PRODOTTO TRA MATRICE FUNZIONALE E MATRICE SCALARE!!!!!!!!!!!!!!!!!!!
    */


    /*!
    * @brief Override of the base class method
    */ 
    inline 
    void 
    compute()  
    override
    {
        double loc = 0.3;

std::cout << "In compute" << std::endl;

        for(std::size_t i = 0; i < omega.rows(); ++i){
    for(std::size_t j = 0; j < omega.cols(); ++j){
            std::string present_s;
            if(omega.check_elem_presence(i,j)){present_s="present";}  else{present_s="not present";}
            std::cout << "Elem of OMEGA (" << i << "," << j << ") is " << present_s << " evaluated in " << loc << ": " << omega(i,j)(loc) << std::endl;
    }}

/*
        std::cout << "In compute" << std::endl;


        
        auto row_test = m_Xc.get_row(3);
        for(std::size_t i = 0; i < row_test.cols(); ++i){
            std::cout << "Elem of col " << i+1 << " in row 4 evaluated in " << loc << ": " << row_test(0,i)(loc) << std::endl;
        }

        auto col_test = m_Xc.get_col(4);
        for(std::size_t i = 0; i < col_test.rows(); ++i){
            std::cout << "Elem of row " << i+1 << " in col 5 evaluated in " << loc << ": " << col_test(i,0)(loc) << std::endl;
        }

        m_Xc.transpose();

        auto col_test2 = m_Xc.get_row(4);
        for(std::size_t i = 0; i < col_test2.cols(); ++i){
            std::cout << "In Xc transpose, the row 5, previous col 5, elem in col " << i+1 << ", evaluated in " << loc << ": " << col_test2(0,i)(loc) << std::endl;
        }
*/


        //auto red = m_Xc.reduce();
        //std::cout << "Reduction in " << loc << ": " << red(loc) << std::endl;




    }

};

#endif  /*FGWR_FMS_ESC_ALGO_HPP*/