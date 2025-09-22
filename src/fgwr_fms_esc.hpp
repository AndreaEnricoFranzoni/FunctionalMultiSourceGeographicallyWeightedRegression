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
    /*!Basis for response (nx(nxLy))*/
    functional_matrix_sparse<INPUT,OUTPUT> m_phi;
    /*!Coefficients of the basis expansion for response ((n*Ly)x1): coefficients for each unit are columnized one below the other*/
    FDAGWR_TRAITS::Dense_Matrix m_c;
    /*!Number of basis used to make basis expansion for y*/
    std::size_t m_Ly;

    /*!Functional stationary covariates (n x qc)*/
    functional_matrix<INPUT,OUTPUT> m_Xc;
    /*!Their transpost (qc x n)*/
    functional_matrix<INPUT,OUTPUT> m_Xc_t;
    /*!Functional weights for stationary covariates (n elements of diagonal n x n)*/
    functional_matrix_diagonal<INPUT,OUTPUT> m_Wc;
    /*!Scalar matrix with the penalization on the stationary covariates (sparse Lc x Lc, where Lc is the sum of the basis of each C covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Rc;
    /*!Basis for stationary covariates regressors (sparse qc x Lc)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_omega;
    /*!Their transpost (sparse Lc x qC)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_omega_t;
    /*!Coefficients of the basis expansion for stationary regressors coefficients: Lcx1: TO BE COMPUTED*/
    FDAGWR_TRAITS::Dense_Matrix m_bc;
    /*!Number of stationary covariates*/
    std::size_t m_qc;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the stationary regressors coefficients*/
    std::size_t m_Lc; 

    /*!Functional event-dependent covariates (n x qe)*/
    functional_matrix<INPUT,OUTPUT> m_Xe;
    /*!Their transpost (qe x n)*/
    functional_matrix<INPUT,OUTPUT> m_Xe_t;
    /*!Functional weights for event-dependent covariates (n elements of diagonal n x n)*/
    std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > m_We;
    /*!Scalar matrix with the penalization on the event-dependent covariates (sparse Le x Le, where Le is the sum of the basis of each E covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Re;
    /*!Basis for event-dependent covariates regressors (sparse qe x Le)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_theta;
    /*!Their transpost (sparse Le x qE)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_theta_t;
    /*!Coefficients of the basis expansion for event-dependent regressors: Lex1, every element of the vector is referring to a specific unit TO BE COMPUTED*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_be;
    /*!Number of event-dependent covariates*/
    std::size_t m_qe;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the event-dependent regressors coefficients*/
    std::size_t m_Le; 

    /*!Functional station-dependent covariates (n x qs)*/
    functional_matrix<INPUT,OUTPUT> m_Xs;
    /*!Their transpost (qs x n)*/
    functional_matrix<INPUT,OUTPUT> m_Xs_t;
    /*!Functional weights for station-dependent covariates (n elements of diagonal n x n)*/
    std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > m_Ws;
    /*!Scalar matrix with the penalization on the station-dependent covariates (sparse Ls x Ls, where Ls is the sum of the basis of each S covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Rs;
    /*!Basis for station-dependent covariates regressors (sparse qs x Ls)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_psi;
    /*!Their transpost (sparse Ls x qS)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_psi_t;
    /*!Coefficients of the basis expansion for station-dependent covariates regressors: Lsx1, every element of the vector is referring to a specific unit TO BE COMPUTED*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_bs;
    /*!Number of station-dependent covariates*/
    std::size_t m_qs;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the station-dependent regressors coefficients*/
    std::size_t m_Ls; 

    //A operators
    /*!A_E_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_A_e;
    /*!A_S_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_A_s;
    /*!A_SE_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_A_se;
    /*!A_ES_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_A_es;
    /*!A_ESE_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_A_ese;

    //B operators
    /*!B_E_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_e;
    /*!B_S_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_s;
    /*!B_SE_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_se;
    /*!B_ES_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_es;
    /*!B_ESE_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_ese;


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
                 FUNC_SPARSE_MATRIX_OBJ &&phi,
                 SCALAR_MATRIX_OBJ &&c,
                 std::size_t Ly,
                 FUNC_MATRIX_OBJ &&Xc,
                 FUNC_DIAG_MATRIX_OBJ &&Wc,
                 SCALAR_SPARSE_MATRIX_OBJ &&Rc,
                 FUNC_SPARSE_MATRIX_OBJ &&omega,
                 std::size_t qc,
                 std::size_t Lc,
                 FUNC_MATRIX_OBJ &&Xe,
                 FUNC_DIAG_MATRIX_VEC_OBJ &&We,
                 SCALAR_SPARSE_MATRIX_OBJ &&Re,
                 FUNC_SPARSE_MATRIX_OBJ &&theta,
                 std::size_t qe,
                 std::size_t Le,
                 FUNC_MATRIX_OBJ &&Xs,
                 FUNC_DIAG_MATRIX_VEC_OBJ &&Ws,
                 SCALAR_SPARSE_MATRIX_OBJ &&Rs,
                 FUNC_SPARSE_MATRIX_OBJ &&psi,
                 std::size_t qs,
                 std::size_t Ls,
                 INPUT a,
                 INPUT b,
                 int n_intervals_integration,
                 double target_error_integration,
                 int max_iterations_integration,
                 std::size_t n,
                 int number_threads)
        :
            fgwr<INPUT,OUTPUT>(a,b,n_intervals_integration,target_error_integration,max_iterations_integration,n,number_threads),
            m_y{std::forward<FUNC_MATRIX_OBJ>(y)},
            m_phi{std::forward<FUNC_SPARSE_MATRIX_OBJ>(phi)},
            m_c{std::forward<SCALAR_MATRIX_OBJ>(c)},
            m_Ly(Ly),
            m_Xc{std::forward<FUNC_MATRIX_OBJ>(Xc)},
            m_Wc{std::forward<FUNC_DIAG_MATRIX_OBJ>(Wc)},
            m_Rc{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rc)},
            m_omega{std::forward<FUNC_SPARSE_MATRIX_OBJ>(omega)},
            m_qc(qc),
            m_Lc(Lc),
            m_Xe{std::forward<FUNC_MATRIX_OBJ>(Xe)},
            m_We{std::forward<FUNC_DIAG_MATRIX_VEC_OBJ>(We)},
            m_Re{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Re)},
            m_theta{std::forward<FUNC_SPARSE_MATRIX_OBJ>(theta)},
            m_qe(qe),
            m_Le(Le),
            m_Xs{std::forward<FUNC_MATRIX_OBJ>(Xs)},
            m_Ws{std::forward<FUNC_DIAG_MATRIX_VEC_OBJ>(Ws)},
            m_Rs{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rs)},
            m_psi{std::forward<FUNC_SPARSE_MATRIX_OBJ>(psi)},
            m_qs(qs),
            m_Ls(Ls)
            {
                //checking input consistency
                //response
                assert((m_y.rows() == this->n()) && (m_y.cols() == 1));
                assert((m_phi.rows() == this->n()) && (m_phi.cols() == m_Ly*(this->n())));
                assert((m_c.rows() == m_Ly*(this->n())) && (m_c.cols() == 1));
                //stationary covariates
                assert((m_Xc.rows() == this->n()) && (m_Xc.cols() == m_qc));
                assert((m_Wc.rows() == this->n()) && (m_Wc.cols() == this->n()));
                assert((m_Rc.rows() == m_Lc) && (m_Rc.cols() == m_Lc));
                assert((m_omega.rows() == m_qc) && (m_omega.cols() == m_Lc));
                //event-dependent covariates
                assert((m_Xe.rows() == this->n()) && (m_Xe.cols() == m_qe));
                assert(m_We.size() == this->n());
                for (const auto& w : m_We) {    assert((w.rows() == this->n()) && (w.cols() == this->n()));}
                assert((m_Re.rows() == m_Le) && (m_Re.cols() == m_Le));
                assert((m_theta.rows() == m_qe) && (m_theta.cols() == m_Le));
                //station-dependent covariates
                assert((m_Xs.rows() == this->n()) && (m_Xs.cols() == m_qs));
                assert(m_Ws.size() == this->n());
                for (const auto& w : m_Ws) {    assert((w.rows() == this->n()) && (w.cols() == this->n()));}
                assert((m_Rs.rows() == m_Ls) && (m_Rs.cols() == m_Ls));
                assert((m_psi.rows() == m_qs) && (m_psi.cols() == m_Ls));

                //compute all the transpost necessary for the computations
                m_Xc_t = m_Xc.transpose();
                m_omega_t = m_omega.transpose();
                m_Xe_t = m_Xe.transpose();
                m_theta_t = m_theta.transpose();
                m_Xc_t = m_Xc.transpose();
                m_psi_t = m_psi.transpose();
            }

    

    /*!
    * @brief Override of the base class method to perform fgwr fms esc algorithm
    */ 
    inline 
    void 
    compute()  
    override
    {
        //(j_tilde_tilde + Re)^-1
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_double_tilde_Re_inv = this->compute_penalty(m_theta_t,m_Xe_t,m_We,m_Xe,m_theta,m_Re);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
        //A_E_i
        m_A_e = this->compute_operator(m_theta_t,m_Xe_t,m_We,m_phi,j_double_tilde_Re_inv);
        //B_E_i
        m_B_e = this->compute_operator(m_theta_t,m_Xe_t,m_We,m_Xc,m_omega,j_double_tilde_Re_inv);
        //K_e_s(t)
        std::vector< FDAGWR_TRAITS::Dense_Matrix > Be_for_K_e_s = this->compute_operator(m_theta_t,m_Xe_t,m_We,m_Xs,m_psi,j_double_tilde_Re_inv);
        functional_matrix<INPUT,OUTPUT> K_e_s = this->compute_functional_operator(m_Xe,m_theta,Be_for_K_e_s);
        //X_s_crossed(t)
        functional_matrix<INPUT,OUTPUT> X_s_crossed = fm_prod(m_Xs,m_psi) - K_e_s;
        functional_matrix<INPUT,OUTPUT> X_s_crossed_t = X_s_crossed.transpose();
        //(j_tilde + Rs)^-1
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_tilde_Rs_inv = this->compute_penalty(X_s_crossed_t,m_Ws,X_s_crossed,m_Rs);
        
        //A_S_i
        m_A_s = this->compute_operator(X_s_crossed_t,m_Ws,m_phi,j_tilde_Rs_inv);
        //H_e(t)
        functional_matrix<INPUT,OUTPUT> H_e = this->compute_functional_operator(m_Xe,m_theta,m_A_e);
        //H_s(t)
        functional_matrix<INPUT,OUTPUT> H_s = this->compute_functional_operator(m_Xs,m_psi,m_A_s);
        //A_SE_i
        m_A_se = this->compute_operator(X_s_crossed_t,m_Ws,H_e,j_tilde_Rs_inv);
        //H_se(t)
        functional_matrix<INPUT,OUTPUT> H_se = this->compute_functional_operator(m_Xs,m_psi,m_A_se);
        //A_ES_i
        m_A_es = this->compute_operator(m_omega_t,m_Xe_t,m_We,H_s,j_double_tilde_Re_inv);
        //H_es(t)
        functional_matrix<INPUT,OUTPUT> H_es = this->compute_functional_operator(m_Xe,m_omega,m_A_es);
        //A_ESE_i
        m_A_ese = this->compute_operator(m_omega_t,m_Xe_t,m_We,H_se,j_double_tilde_Re_inv);
        //H_ese(t)
        functional_matrix<INPUT,OUTPUT> H_ese = this->compute_functional_operator(m_Xe,m_omega,m_A_ese);

        //B_S_i
        m_B_s = this->compute_operator(X_s_crossed_t,m_Ws,m_Xc,m_omega,j_tilde_Rs_inv);
        //K_e_c(t)
        functional_matrix<INPUT,OUTPUT> K_e_c = this->compute_functional_operator(m_Xe,m_theta,m_B_e);
        //K_s_c(t)
        functional_matrix<INPUT,OUTPUT> K_s_c = this->compute_functional_operator(m_Xs,m_psi,m_B_s);
        //B_SE_i
        m_B_se = this->compute_operator(X_s_crossed_t,m_Ws,K_e_c,j_tilde_Rs_inv);
        //B_ES_i
        m_B_es = this->compute_operator(m_theta_t,m_Xe_t,m_We,K_s_c,j_double_tilde_Re_inv);
        //K_se_c(t)
        functional_matrix<INPUT,OUTPUT> K_se_c = this->compute_functional_operator(m_Xs,m_psi,m_B_se);
        //K_es_c(t)
        functional_matrix<INPUT,OUTPUT> K_es_c = this->compute_functional_operator(m_Xe,m_theta,m_B_es);
        //B_ESE_i
        m_B_ese = this->compute_operator(m_theta_t,m_Xe_t,m_We,K_se_c,j_double_tilde_Re_inv);
        //K_ese_c(t)
        functional_matrix<INPUT,OUTPUT> K_ese_c = this->compute_functional_operator(m_Xe,m_theta,m_B_ese); 


        //y_new(t)
        functional_matrix<INPUT,OUTPUT> y_new = fm_prod(functional_matrix<INPUT,OUTPUT>(m_phi - H_e - H_s + H_se + H_es - H_ese),m_c,this->number_threads());
        functional_matrix<INPUT,OUTPUT> X_c_crossed = fm_prod(m_Xc,m_omega) - K_e_c - K_s_c + K_se_c + K_es_c - K_ese_c;
        functional_matrix<INPUT,OUTPUT> X_c_crossed_t = X_c_crossed.transpose();
        //[J + Rc]^-1
        Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> j_Rc_inv = this->compute_penalty(X_c_crossed_t,m_Wc,X_c_crossed,m_Rc);
        

        //COMPUTING m_bc, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATIONARY BETAS
        m_bc = this->compute_operator(X_c_crossed_t,m_Wc,y_new,j_Rc_inv);


        //y_tilde_hat(t)
        functional_matrix<INPUT,OUTPUT> y_tilde_hat = m_y - fm_prod(fm_prod(m_Xc,m_omega),m_bc,this->number_threads());
        //c_tilde_hat
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        /// TODO: QUI BISOGNA SMOOTHARE CON LE PHI y_tilde_hat PER OTTENERE c_tilde_hat (E INCOLONNARLI) /////
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        FDAGWR_TRAITS::Dense_Matrix c_tilde_hat = m_c;
        //y_tilde_new(t)
        functional_matrix<INPUT,OUTPUT> y_tilde_new = fm_prod(functional_matrix<INPUT,OUTPUT>(m_phi - H_e),c_tilde_hat,this->number_threads());


        //COMPUTING all the m_bs, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATION-DEPENDENT BETAS
        m_bs = this->compute_operator(X_s_crossed_t,m_Ws,y_tilde_new,j_tilde_Rs_inv);


        //y_tilde_tilde_hat(t)
        functional_matrix<INPUT,OUTPUT> y_tilde_tilde_hat = y_tilde_hat - this->compute_functional_operator(m_Xs,m_psi,m_B_s);


        //COMPUTING all the m_be, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE EVENT-DEPENDENT BETAS
        m_be = this->compute_operator(m_theta_t,m_Xe_t,m_We,y_tilde_tilde_hat,j_double_tilde_Re_inv);
    }

    /*!
    * @brief Getter for the coefficient of the basis expansion of the stationary regressors coefficients
    */
    inline 
    CoefficientsTuple 
    regressorCoefficients()
    const 
    override
    {
        return std::tuple{m_bc,m_be,m_bs};
    }


};

#endif  /*FGWR_FMS_ESC_ALGO_HPP*/