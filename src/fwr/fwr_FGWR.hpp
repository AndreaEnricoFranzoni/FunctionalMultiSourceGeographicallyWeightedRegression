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


#ifndef FWR_FGWR_ALGO_HPP
#define FWR_FGWR_ALGO_HPP

#include "fwr.hpp"
#include <iostream>

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_FGWR final : public fwr<INPUT,OUTPUT>
{
private:
    /*!Functional response (nx1)*/
    functional_matrix<INPUT,OUTPUT> m_y;

    /*!Functional event-dependent covariates (n x qnc)*/
    functional_matrix<INPUT,OUTPUT> m_Xnc;
    /*!Their transpost (qnc x n)*/
    functional_matrix<INPUT,OUTPUT> m_Xnc_t;
    /*!Functional weights for event-dependent covariates (n elements of diagonal n x n)*/
    std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > m_Wnc;
    /*!Scalar matrix with the penalization on the event-dependent covariates (sparse Le x Le, where Le is the sum of the basis of each E covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Rnc;
    /*!Basis for event-dependent covariates regressors (sparse qnc x Lnc)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_eta;
    /*!Their transpost (sparse Lnc x qNC)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_eta_t;
    /*!Coefficients of the basis expansion for event-dependent regressors: Lncx1, every element of the vector is referring to a specific unit TO BE COMPUTED*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_bnc;
    /*!Coefficients of the basis expansion for event-dependent regressors coefficients: every of the qe elements are n 1xLnc_j matrices, one for each statistical unit*/
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> m_Bnc;
    /*!Discrete evaluation of all the beta_nc: a vector of dimension qnc, containing, for all the event-dependent covariates, the discrete ev of the respective beta, for each statistical unit*/
    std::vector< std::vector< std::vector< OUTPUT >>> m_beta_nc;
    /*!Number of non stationary covariates*/
    std::size_t m_qnc;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the event-dependent regressors coefficients*/
    std::size_t m_Lnc; 
    /*!Number of basis, for each event-dependent covariate, to perform the basis expansion of the regressors coefficients for the event-dependent regressors coefficients*/
    std::vector<std::size_t> m_Lnc_j;


public:
    /*!
    * @brief Constructor
    */
    template<typename FUNC_MATRIX_OBJ, 
             typename FUNC_SPARSE_MATRIX_OBJ, 
             typename FUNC_DIAG_MATRIX_VEC_OBJ,  
             typename SCALAR_SPARSE_MATRIX_OBJ> 
    fwr_FGWR(FUNC_MATRIX_OBJ &&y,
             FUNC_MATRIX_OBJ &&Xnc,
             FUNC_DIAG_MATRIX_VEC_OBJ &&Wnc,
             SCALAR_SPARSE_MATRIX_OBJ &&Rnc,
             FUNC_SPARSE_MATRIX_OBJ &&eta,
             std::size_t qnc,
             std::size_t Lnc,
             const std::vector<std::size_t> & Lnc_j,
             INPUT a,
             INPUT b,
             int n_intervals_integration,
             const std::vector<INPUT> & abscissa_points,
             std::size_t n,
             int number_threads)
        :
            fwr<INPUT,OUTPUT>(a,b,n_intervals_integration,abscissa_points,n,number_threads,false),
            m_y{std::forward<FUNC_MATRIX_OBJ>(y)},
            m_Xnc{std::forward<FUNC_MATRIX_OBJ>(Xnc)},
            m_Wnc{std::forward<FUNC_DIAG_MATRIX_VEC_OBJ>(Wnc)},
            m_Rnc{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rnc)},
            m_eta{std::forward<FUNC_SPARSE_MATRIX_OBJ>(eta)},
            m_qnc(qnc),
            m_Lnc(Lnc),
            m_Lnc_j(Lnc_j)
            {
                //checking input consistency
                //response
                assert((m_y.rows() == this->n()) && (m_y.cols() == 1));
                //non stationary covariates
                assert((m_Xnc.rows() == this->n()) && (m_Xnc.cols() == m_qnc));
                assert(m_Wnc.size() == this->n());
                for(std::size_t i = 0; i < m_Wnc.size(); ++i){   assert((m_Wnc[i].rows() == this->n()) && (m_Wnc[i].cols() == this->n()));}
                assert((m_Rnc.rows() == m_Lnc) && (m_Rnc.cols() == m_Lnc));
                assert((m_eta.rows() == m_qnc) && (m_eta.cols() == m_Lnc));
                assert((m_Lnc_j.size() == m_qnc) && (std::reduce(m_Lnc_j.cebgin(),m_Lnc_j.cend(),static_cast<std::size_t>(0)) == m_Lnc));

                //compute all the transpost necessary for the computations
                m_Xnc_t = m_Xnc.transpose();
                m_eta_t = m_eta.transpose();
            }
    

    /*!
    * @brief Override of the base class method to perform fgwr fms esc algorithm
    */ 
    inline 
    void 
    compute()  
    override
    {

        //(j + Rnc)^-1
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_Rnc_inv = this->operator_comp().compute_penalty(m_eta_t,m_Xnc_t,m_Wnc,m_Xnc,m_eta,m_Rnc);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
        //COMPUTING all the m_bnc, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATION-DEPENDENT BETAS
        m_bnc = this->operator_comp().compute_operator(m_eta_t,m_Xnc_t,m_Wnc,m_y,j_Rnc_inv);

        //
        //wrapping the b from the shape useful for the computation into a more useful format: TENERE
        //
        //non-stationary covariates
        m_Bnc = this->operator_comp().wrap_operator(m_bnc,m_Lnc_j,m_qnc,this->n());
    }

    /*!
    * @brief Virtual method to obtain a discrete version of the betas
    */
    inline 
    void 
    evalBetas()
    override
    {      
        //BETA_NC
        m_beta_nc = this->operator_comp().eval_func_betas(m_Bnc,m_eta,m_Lnc_j,m_qnc,this->n(),this->abscissa_points());
    }

    /*!
    * @brief Getter for the coefficient of the basis expansion of the stationary regressors coefficients
    */
    inline 
    BTuple 
    bCoefficients()
    const 
    override
    {
        return std::tuple{m_Bnc};
    }

    /*!
    * @brief Getter for the. etas evaluated along the abscissas
    */
    inline 
    BetasTuple 
    betas() 
    const
    override
    {
        return std::tuple{m_beta_nc};
    }

    /*!
    * @brief Getter for the objects needed for reconstructing the partial residuals
    */
    inline
    PartialResidualTuple
    PRes()
    const
    override
    {
        return std::monostate{};
    }
};

#endif  /*FWR_FGWR_ALGO_HPP*/