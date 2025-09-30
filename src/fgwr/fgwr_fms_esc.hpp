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
    /*!Basis used for y (the functions put in m_phi)*/
    std::unique_ptr<basis_base_class<FDAGWR_TRAITS::basis_geometry>> m_basis_y;
    /*!Knots for the response, used at the beginning to obtain y basis expansion coefficients via smoothing*/
    FDAGWR_TRAITS::Dense_Matrix m_knots_smoothing;

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
    /*!Coefficients of the basis expansion for stationary regressors coefficients: Lcx1 (used for the computation): TO BE COMPUTED*/
    FDAGWR_TRAITS::Dense_Matrix m_bc;
    /*!Coefficients of the basis expansion for stationary regressors coefficients: every element is 1xLc_j*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_Bc;
    /*!Discrete evaluation of all the beta_c: a vector of dimension qc, containing, for all the stationary covariates, the discrete ev of the respective beta*/
    std::vector< std::vector<OUTPUT> > m_beta_c;
    /*!Number of stationary covariates*/
    std::size_t m_qc;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the stationary regressors coefficients*/
    std::size_t m_Lc; 
    /*!Number of basis, for each stationary covariate, to perform the basis expansion of the regressors coefficients for the stationary regressors coefficients*/
    std::vector<std::size_t> m_Lc_j;

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
    /*!Coefficients of the basis expansion for event-dependent regressors coefficients: every of the qe elements are n 1xLe_j matrices, one for each statistical unit*/
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > > m_Be;
    /*!Discrete evaluation of all the beta_e: a vector of dimension qe, containing, for all the event-dependent covariates, the discrete ev of the respective beta, for each statistical unit*/
    std::vector< std::vector< std::vector<OUTPUT> > > m_beta_e;
    /*!Number of event-dependent covariates*/
    std::size_t m_qe;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the event-dependent regressors coefficients*/
    std::size_t m_Le; 
    /*!Number of basis, for each event-dependent covariate, to perform the basis expansion of the regressors coefficients for the event-dependent regressors coefficients*/
    std::vector<std::size_t> m_Le_j;

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
    /*!Coefficients of the basis expansion for station-dependent regressors coefficients: every of the qe elements are n 1xLs_j matrices, one for each statistical unit*/
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix > > m_Bs;
    /*!Discrete evaluation of all the beta_s: a vector of dimension qs, containing, for all the station-dependent covariates, the discrete ev of the respective beta, for each statistical unit*/
    std::vector< std::vector< std::vector<OUTPUT> > > m_beta_s;
    /*!Number of station-dependent covariates*/
    std::size_t m_qs;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the station-dependent regressors coefficients*/
    std::size_t m_Ls; 
    /*!Number of basis, for each station-dependent covariate, to perform the basis expansion of the regressors coefficients for the station-dependent regressors coefficients*/
    std::vector<std::size_t> m_Ls_j;

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
                 std::unique_ptr<basis_base_class<FDAGWR_TRAITS::basis_geometry>> basis_y,
                 SCALAR_MATRIX_OBJ &&knots_smoothing,
                 FUNC_MATRIX_OBJ &&Xc,
                 FUNC_DIAG_MATRIX_OBJ &&Wc,
                 SCALAR_SPARSE_MATRIX_OBJ &&Rc,
                 FUNC_SPARSE_MATRIX_OBJ &&omega,
                 std::size_t qc,
                 std::size_t Lc,
                 const std::vector<std::size_t> & Lc_j,
                 FUNC_MATRIX_OBJ &&Xe,
                 FUNC_DIAG_MATRIX_VEC_OBJ &&We,
                 SCALAR_SPARSE_MATRIX_OBJ &&Re,
                 FUNC_SPARSE_MATRIX_OBJ &&theta,
                 std::size_t qe,
                 std::size_t Le,
                 const std::vector<std::size_t> & Le_j,
                 FUNC_MATRIX_OBJ &&Xs,
                 FUNC_DIAG_MATRIX_VEC_OBJ &&Ws,
                 SCALAR_SPARSE_MATRIX_OBJ &&Rs,
                 FUNC_SPARSE_MATRIX_OBJ &&psi,
                 std::size_t qs,
                 std::size_t Ls,
                 const std::vector<std::size_t> & Ls_j,
                 INPUT a,
                 INPUT b,
                 int n_intervals_integration,
                 double target_error_integration,
                 int max_iterations_integration,
                 const std::vector<INPUT> & abscissa_points,
                 std::size_t n,
                 int number_threads)
        :
            fgwr<INPUT,OUTPUT>(a,b,n_intervals_integration,target_error_integration,max_iterations_integration,abscissa_points,n,number_threads),
            m_y{std::forward<FUNC_MATRIX_OBJ>(y)},
            m_phi{std::forward<FUNC_SPARSE_MATRIX_OBJ>(phi)},
            m_c{std::forward<SCALAR_MATRIX_OBJ>(c)},
            m_Ly(Ly),
            m_basis_y(std::move(basis_y)),
            m_knots_smoothing{std::forward<SCALAR_MATRIX_OBJ>(knots_smoothing)},
            m_Xc{std::forward<FUNC_MATRIX_OBJ>(Xc)},
            m_Wc{std::forward<FUNC_DIAG_MATRIX_OBJ>(Wc)},
            m_Rc{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rc)},
            m_omega{std::forward<FUNC_SPARSE_MATRIX_OBJ>(omega)},
            m_qc(qc),
            m_Lc(Lc),
            m_Lc_j(Lc_j),
            m_Xe{std::forward<FUNC_MATRIX_OBJ>(Xe)},
            m_We{std::forward<FUNC_DIAG_MATRIX_VEC_OBJ>(We)},
            m_Re{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Re)},
            m_theta{std::forward<FUNC_SPARSE_MATRIX_OBJ>(theta)},
            m_qe(qe),
            m_Le(Le),
            m_Le_j(Le_j),
            m_Xs{std::forward<FUNC_MATRIX_OBJ>(Xs)},
            m_Ws{std::forward<FUNC_DIAG_MATRIX_VEC_OBJ>(Ws)},
            m_Rs{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rs)},
            m_psi{std::forward<FUNC_SPARSE_MATRIX_OBJ>(psi)},
            m_qs(qs),
            m_Ls(Ls),
            m_Ls_j(Ls_j)
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
                assert((m_Lc_j.size() == m_qc) && (std::reduce(m_Lc_j.cebgin(),m_Lc_j.cend(),static_cast<std::size_t>(0)) == m_Lc));
                //event-dependent covariates
                assert((m_Xe.rows() == this->n()) && (m_Xe.cols() == m_qe));
                assert(m_We.size() == this->n());
                for (const auto& w : m_We) {    assert((w.rows() == this->n()) && (w.cols() == this->n()));}
                assert((m_Re.rows() == m_Le) && (m_Re.cols() == m_Le));
                assert((m_theta.rows() == m_qe) && (m_theta.cols() == m_Le));
                assert((m_Le_j.size() == m_qe) && (std::reduce(m_Le_j.cebgin(),m_Le_j.cend(),static_cast<std::size_t>(0)) == m_Le));
                //station-dependent covariates
                assert((m_Xs.rows() == this->n()) && (m_Xs.cols() == m_qs));
                assert(m_Ws.size() == this->n());
                for (const auto& w : m_Ws) {    assert((w.rows() == this->n()) && (w.cols() == this->n()));}
                assert((m_Rs.rows() == m_Ls) && (m_Rs.cols() == m_Ls));
                assert((m_psi.rows() == m_qs) && (m_psi.cols() == m_Ls));
                assert((m_Ls_j.size() == m_qs) && (std::reduce(m_Ls_j.cebgin(),m_Ls_j.cend(),static_cast<std::size_t>(0)) == m_Ls));

                //compute all the transpost necessary for the computations
                m_Xc_t = m_Xc.transpose();
                m_omega_t = m_omega.transpose();
                m_Xe_t = m_Xe.transpose();
                m_theta_t = m_theta.transpose();
                m_Xc_t = m_Xc.transpose();
                m_psi_t = m_psi.transpose();
         

/*
                //PRINT FOR DEBUGGING
                std::cout << "Input dimensions" << std::endl;
                std::cout << "m_y rows: " << m_y.rows() << ", m_y cols: " << m_y.cols() << std::endl;
                std::cout << "m_phi rows: " << m_phi.rows() << ", m_phi cols: " << m_phi.cols() << std::endl;
                std::cout << "m_c rows: " << m_c.rows() << ", m_c cols: " << m_c.cols() << std::endl;
                std::cout << "m_knots_smoothing rows: " << m_knots_smoothing.rows() << ", m_knots_smoothing cols: " << m_knots_smoothing.cols() << std::endl;
                std::cout << "m_Xc rows: " << m_Xc.rows() << ", m_Xc cols: " << m_Xc.cols() << std::endl;
                std::cout << "m_Wc rows: " << m_Wc.rows() << ", m_Wc cols: " << m_Wc.cols() << std::endl;
                std::cout << "m_Rc rows: " << m_Rc.rows() << ", m_Rc cols: " << m_Rc.cols() << std::endl;
                std::cout << "m_omega rows: " << m_omega.rows() << ", m_omega cols: " << m_omega.cols() << std::endl;
                std::cout << "Numero basi per qc" << std::endl;
                for(std::size_t i = 0; i  < m_Lc_j.size(); ++i){std::cout << m_Lc_j[i] << std::endl;}
                std::cout << "m_Xe rows: " << m_Xe.rows() << ", m_Xe cols: " << m_Xe.cols() << std::endl;
                std::cout << "Number of elements in m_We: " << m_We.size() << std::endl;
                for(std::size_t i = 0; i < m_We.size(); ++i){   std::cout << "m_We[" << i << "] rows: " << m_We[i].rows() << ", cols: " << m_We[i].cols() << std::endl;}
                std::cout << "m_Re rows: " << m_Re.rows() << ", m_Re cols: " << m_Re.cols() << std::endl;
                std::cout << "m_theta rows: " << m_theta.rows() << ", m_theta cols: " << m_theta.cols() << std::endl;
                std::cout << "Numero basi per qe" << std::endl;
                for(std::size_t i = 0; i < m_Le_j.size(); ++i){std::cout << m_Le_j[i] << std::endl;}
                std::cout << "m_Xs rows: " << m_Xs.rows() << ", m_Xs cols: " << m_Xs.cols() << std::endl;
                std::cout << "Number of elements in m_Ws: " << m_Ws.size() << std::endl;
                for(std::size_t i = 0; i < m_Ws.size(); ++i){   std::cout << "m_Ws[" << i << "] rows: " << m_Ws[i].rows() << ", cols: " << m_Ws[i].cols() << std::endl;}
                std::cout << "m_Rs rows: " << m_Rs.rows() << ", m_Rs cols: " << m_Rs.cols() << std::endl;
                std::cout << "m_psi rows: " << m_psi.rows() << ", m_psi cols: " << m_psi.cols() << std::endl;
                std::cout << "Numero basi per qs" << std::endl;
                for(std::size_t i = 0; i < m_Ls_j.size(); ++i){std::cout << m_Ls_j[i] << std::endl;}


                for (std::size_t i = 0; i < this->abscissa_points().size(); ++i)
                {
                    std::cout << "Abscissa " << i << ": " << this->abscissa_points()[i] << std::endl;
                }
                
                for (std::size_t i = 0; i < m_qc; ++i)
                {
                    std::cout << "Stationary cov " << i << " has " << m_Lc_j[i] << " basis" << std::endl;
                }

                for (std::size_t i = 0; i < m_qe; ++i)
                {
                    std::cout << "Event cov " << i << " has " << m_Le_j[i] << " basis" << std::endl;
                }

                for (std::size_t i = 0; i < m_qs; ++i)
                {
                    std::cout << "Station cov " << i << " has " << m_Ls_j[i] << " basis" << std::endl;
                }
*/
            }
    

    /*!
    * @brief Override of the base class method to perform fgwr fms esc algorithm
    */ 
    inline 
    void 
    compute()  
    override
    {

/*
        std::cout << "Computing (j_tilde_tilde + Re)^-1" << std::endl;
        //(j_tilde_tilde + Re)^-1
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_double_tilde_Re_inv = this->compute_penalty(m_theta_t,m_Xe_t,m_We,m_Xe,m_theta,m_Re);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
        //A_E_i
        std::cout << "Computing A_e_i" << std::endl;
        m_A_e = this->compute_operator(m_theta_t,m_Xe_t,m_We,m_phi,j_double_tilde_Re_inv);
        std::cout << "A_e_i rows: " << m_A_e[0].rows() << ", A_e_i cols: " << m_A_e[0].cols() << std::endl;
        //B_E_i
        std::cout << "Computing B_e_i" << std::endl;
        m_B_e = this->compute_operator(m_theta_t,m_Xe_t,m_We,m_Xc,m_omega,j_double_tilde_Re_inv);
        std::cout << "B_e_i rows: " << m_B_e[0].rows() << ", B_e_i cols: " << m_B_e[0].cols() << std::endl;
        //K_e_s(t)
        std::cout << "Computing Be_for_K_e_s" << std::endl;
        std::vector< FDAGWR_TRAITS::Dense_Matrix > Be_for_K_e_s = this->compute_operator(m_theta_t,m_Xe_t,m_We,m_Xs,m_psi,j_double_tilde_Re_inv);
        std::cout << "Be_for_K_e_s rows: " << Be_for_K_e_s[0].rows() << ", Be_for_K_e_s cols: " << Be_for_K_e_s[0].cols() << std::endl;
        std::cout << "Computing K_e_s" << std::endl;
        functional_matrix<INPUT,OUTPUT> K_e_s = this->compute_functional_operator(m_Xe,m_theta,Be_for_K_e_s);
        std::cout << "K_e_s rows: " << K_e_s.rows() << ", K_e_s cols: " << K_e_s.cols() << std::endl;
        //X_s_crossed(t)
        std::cout << "Computing X_s_crossed" << std::endl;
        functional_matrix<INPUT,OUTPUT> X_s_crossed = fm_prod(m_Xs,m_psi) - K_e_s;
        functional_matrix<INPUT,OUTPUT> X_s_crossed_t = X_s_crossed.transpose();
        std::cout << "X_s_crossed rows: " << X_s_crossed.rows() << ", X_s_crossed cols: " << X_s_crossed.cols() << std::endl;
        //(j_tilde + Rs)^-1
        std::cout << "Computing (j_tilde + Rs)^-1" << std::endl;
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_tilde_Rs_inv = this->compute_penalty(X_s_crossed_t,m_Ws,X_s_crossed,m_Rs);
        
        //A_S_i
        std::cout << "Computing m_A_s" << std::endl;
        m_A_s = this->compute_operator(X_s_crossed_t,m_Ws,m_phi,j_tilde_Rs_inv);
        std::cout << "m_A_s rows: " << m_A_s[0].rows() << ", m_A_s cols: " << m_A_s[0].cols() << std::endl;
        //H_e(t)
        std::cout << "Computing H_e" << std::endl;
        functional_matrix<INPUT,OUTPUT> H_e = this->compute_functional_operator(m_Xe,m_theta,m_A_e);
        std::cout << "H_e rows: " << H_e.rows() << ", H_e cols: " << H_e.cols() << std::endl;
        //H_s(t)
        std::cout << "Computing H_s" << std::endl;
        functional_matrix<INPUT,OUTPUT> H_s = this->compute_functional_operator(m_Xs,m_psi,m_A_s);
        std::cout << "H_s rows: " << H_s.rows() << ", H_s cols: " << H_s.cols() << std::endl;
        //A_SE_i
        //std::cout << "Computing m_A_se" << std::endl;
        //m_A_se = this->compute_operator(X_s_crossed_t,m_Ws,H_e,j_tilde_Rs_inv);
        //std::cout << "m_A_se rows: " << m_A_se[0].rows() << ", m_A_se cols: " << m_A_se[0].cols() << std::endl;
        //H_se(t)
        //std::cout << "Computing H_se" << std::endl;
        //functional_matrix<INPUT,OUTPUT> H_se = this->compute_functional_operator(m_Xs,m_psi,m_A_se);
        //std::cout << "H_se rows: " << H_se.rows() << ", H_se cols: " << H_se.cols() << std::endl;
        //A_ES_i
        std::cout << "Computing A_ES_i" << std::endl;
        m_A_es = this->compute_operator(m_theta_t,m_Xe_t,m_We,H_s,j_double_tilde_Re_inv);
        std::cout << "m_A_es rows: " << m_A_es[0].rows() << ", m_A_es cols: " << m_A_es[0].cols() << std::endl;
        //H_es(t)
        std::cout << "Computing H_es" << std::endl;
        functional_matrix<INPUT,OUTPUT> H_es = this->compute_functional_operator(m_Xe,m_theta,m_A_es);
        std::cout << "H_es rows: " << H_es.rows() << ", H_es cols: " << H_es.cols() << std::endl;
        //A_ESE_i
        //std::cout << "Computing A_ESE_i" << std::endl;
        //m_A_ese = this->compute_operator(m_theta_t,m_Xe_t,m_We,H_se,j_double_tilde_Re_inv);
        //std::cout << "m_A_ese rows: " << m_A_ese[0].rows() << ", m_A_ese cols: " << m_A_ese[0].cols() << std::endl;
        //H_ese(t)
        //std::cout << "Computing H_ese" << std::endl;
        //functional_matrix<INPUT,OUTPUT> H_ese = this->compute_functional_operator(m_Xe,m_omega,m_A_ese);
        //std::cout << "H_ese rows: " << H_ese.rows() << ", H_ese cols: " << H_ese.cols() << std::endl;

        //B_S_i
        std::cout << "Computing m_B_s" << std::endl;
        m_B_s = this->compute_operator(X_s_crossed_t,m_Ws,m_Xc,m_omega,j_tilde_Rs_inv);
        std::cout << "m_B_s rows: " << m_B_s[0].rows() << ", m_B_s cols: " << m_B_s[0].cols() << std::endl;
        //K_e_c(t)
        std::cout << "Computing K_e_c" << std::endl;
        functional_matrix<INPUT,OUTPUT> K_e_c = this->compute_functional_operator(m_Xe,m_theta,m_B_e);
        std::cout << "K_e_c rows: " << K_e_c.rows() << ", K_e_c cols: " << K_e_c.cols() << std::endl;
        //K_s_c(t)
        std::cout << "Computing K_s_c" << std::endl;
        functional_matrix<INPUT,OUTPUT> K_s_c = this->compute_functional_operator(m_Xs,m_psi,m_B_s);
        std::cout << "K_s_c rows: " << K_s_c.rows() << ", K_s_c cols: " << K_s_c.cols() << std::endl;
        //B_SE_i
        //std::cout << "Computing m_B_se" << std::endl;
        //m_B_se = this->compute_operator(X_s_crossed_t,m_Ws,K_e_c,j_tilde_Rs_inv);
        //std::cout << "m_B_se rows: " << m_B_se[0].rows() << ", m_B_se cols: " << m_B_se[0].cols() << std::endl;
        //B_ES_i
        std::cout << "Computing m_B_es" << std::endl;
        m_B_es = this->compute_operator(m_theta_t,m_Xe_t,m_We,K_s_c,j_double_tilde_Re_inv);
        std::cout << "m_B_es rows: " << m_B_es[0].rows() << ", m_B_es cols: " << m_B_es[0].cols() << std::endl;
        //K_se_c(t)
        //std::cout << "Computing K_se_c" << std::endl;
        //functional_matrix<INPUT,OUTPUT> K_se_c = this->compute_functional_operator(m_Xs,m_psi,m_B_se);
        //std::cout << "K_se_c rows: " << K_se_c.rows() << ", K_se_c cols: " << K_se_c.cols() << std::endl;
        //K_es_c(t)
        std::cout << "Computing K_es_c" << std::endl;
        functional_matrix<INPUT,OUTPUT> K_es_c = this->compute_functional_operator(m_Xe,m_theta,m_B_es);
        std::cout << "K_es_c rows: " << K_es_c.rows() << ", K_es_c cols: " << K_es_c.cols() << std::endl;
        //B_ESE_i
        //std::cout << "Computing m_B_ese" << std::endl;
        //m_B_ese = this->compute_operator(m_theta_t,m_Xe_t,m_We,K_se_c,j_double_tilde_Re_inv);
        //std::cout << "m_B_ese rows: " << m_B_ese[0].rows() << ", m_B_ese cols: " << m_B_ese[0].cols() << std::endl;
        //K_ese_c(t)
        //std::cout << "Computing K_ese_c" << std::endl;
        //functional_matrix<INPUT,OUTPUT> K_ese_c = this->compute_functional_operator(m_Xe,m_theta,m_B_ese); 
        //std::cout << "K_ese_c rows: " << K_ese_c.rows() << ", K_ese_c cols: " << K_ese_c.cols() << std::endl;


        //y_new(t)
        std::cout << "Computing y_new" << std::endl;
        //functional_matrix<INPUT,OUTPUT> y_new = fm_prod(functional_matrix<INPUT,OUTPUT>(m_phi - H_e - H_s + H_se + H_es - H_ese),m_c,this->number_threads());
        functional_matrix<INPUT,OUTPUT> y_new = fm_prod(functional_matrix<INPUT,OUTPUT>(m_phi - H_e - H_s + H_es),m_c,this->number_threads());
        std::cout << "y_new rows: " << y_new.rows() << ", y_new cols: " << y_new.cols() << std::endl;
        std::cout << "Computing X_c_crossed" << std::endl;
        //functional_matrix<INPUT,OUTPUT> X_c_crossed = fm_prod(m_Xc,m_omega) - K_e_c - K_s_c + K_se_c + K_es_c - K_ese_c;
        functional_matrix<INPUT,OUTPUT> X_c_crossed = fm_prod(m_Xc,m_omega) - K_e_c - K_s_c + K_es_c;
        functional_matrix<INPUT,OUTPUT> X_c_crossed_t = X_c_crossed.transpose();
        std::cout << "X_c_crossed rows: " << X_c_crossed.rows() << ", X_c_crossed cols: " << X_c_crossed.cols() << std::endl;
        //[J + Rc]^-1
        std::cout << "Computing [J + Rc]^-1" << std::endl;
        Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> j_Rc_inv = this->compute_penalty(X_c_crossed_t,m_Wc,X_c_crossed,m_Rc);
        

        //COMPUTING m_bc, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATIONARY BETAS
        std::cout << "Computing m_bc" << std::endl;
        m_bc = this->compute_operator(X_c_crossed_t,m_Wc,y_new,j_Rc_inv);
        std::cout << "m_bc rows: " << m_bc.rows() << ", m_bc cols: " << m_bc.cols() << std::endl;


        //y_tilde_hat(t)
        std::cout << "Computing y_tilde_hat" << std::endl;
        functional_matrix<INPUT,OUTPUT> y_tilde_hat = m_y - fm_prod(fm_prod(m_Xc,m_omega),m_bc,this->number_threads());
        std::cout << "y_tilde_hat rows: " << y_tilde_hat.rows() << ", y_tilde_hat cols: " << y_tilde_hat.cols() << std::endl;
        //c_tilde_hat: smoothing on y_tilde_hat(t) with respect of the basis of y
        std::cout << "Computing c_tilde_hat" << std::endl;
        FDAGWR_TRAITS::Dense_Matrix c_tilde_hat = columnize_coeff_resp(fm_smoothing<INPUT,OUTPUT,FDAGWR_TRAITS::basis_geometry>(y_tilde_hat,*m_basis_y,m_knots_smoothing));
        std::cout << "c_tilde_hat rows: " << c_tilde_hat.rows() << ", c_tilde_hat cols: " << c_tilde_hat.cols() << std::endl;
        std::cout << c_tilde_hat << std::endl;
        //y_tilde_new(t)
        std::cout << "Computing y_tilde_new" << std::endl;
        functional_matrix<INPUT,OUTPUT> y_tilde_new = fm_prod(functional_matrix<INPUT,OUTPUT>(m_phi - H_e),c_tilde_hat,this->number_threads());
        std::cout << "y_tilde_new rows: " << y_tilde_new.rows() << ", y_tilde_new cols: " << y_tilde_new.cols() << std::endl;


        //COMPUTING all the m_bs, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATION-DEPENDENT BETAS
        std::cout << "Computing m_bs" << std::endl;
        m_bs = this->compute_operator(X_s_crossed_t,m_Ws,y_tilde_new,j_tilde_Rs_inv);
        std::cout << "m_bs rows: " << m_bs[0].rows() << ", m_bs cols: " << m_bs[0].cols() << std::endl;


        //y_tilde_tilde_hat(t)
        std::cout << "Computing y_tilde_tilde_hat" << std::endl;
        functional_matrix<INPUT,OUTPUT> y_tilde_tilde_hat = y_tilde_hat - this->compute_functional_operator(m_Xs,m_psi,m_B_s);
        std::cout << "y_tilde_tilde_hat rows: " << y_tilde_tilde_hat.rows() << ", y_tilde_tilde_hat cols: " << y_tilde_tilde_hat.cols() << std::endl;


        //COMPUTING all the m_be, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE EVENT-DEPENDENT BETAS
        std::cout << "Computing m_be" << std::endl;
        m_be = this->compute_operator(m_theta_t,m_Xe_t,m_We,y_tilde_tilde_hat,j_double_tilde_Re_inv);
        std::cout << "m_be rows: " << m_be[0].rows() << ", m_be cols: " << m_be[0].cols() << std::endl;
*/

        //DEFAULT AI B: PARTE DA TOGLIERE
        m_bc = Eigen::MatrixXd::Constant(m_Lc,1,4.6);
        m_be.reserve(this->n());
        m_bs.reserve(this->n());

        for(std::size_t i = 0; i < this->n(); ++i)
        {
            m_be.push_back(Eigen::MatrixXd::Constant(m_Le,1,i+4));
            m_bs.push_back(Eigen::MatrixXd::Constant(m_Ls,1,i+2.4));
        }






        //
        //wrapping the b from the shape useful for the computation into a more useful format: TENERE
        //
        //stationary covariates
        m_Bc.reserve(m_qc);
        for(std::size_t j = 0; j < m_qc; ++j){
            //for each stationary covariates
            std::size_t start_idx = j == 0    ? 0 : m_Lc_j[j-1];
            //taking the right coefficients of the basis expansion
            FDAGWR_TRAITS::Dense_Matrix Bc_j = m_bc.block(start_idx,0,m_Lc_j[j],1);
            m_Bc.push_back(Bc_j);}
        //event-dependent covariates
        m_Be.reserve(m_qe);
        for(std::size_t j = 0; j < m_qe; ++j){
            //for each event-dependent covariates
            std::size_t start_idx = j == 0    ? 0 : m_Le_j[j-1];
            std::vector< FDAGWR_TRAITS::Dense_Matrix > Be_j;
            Be_j.reserve(this->n());
            //for all the units
            for(std::size_t i = 0; i < this->n(); ++i){
                //taking the right coefficients of the basis expansion
                FDAGWR_TRAITS::Dense_Matrix Be_j_i = m_be[i].block(start_idx,0,m_Le_j[j],1);
                Be_j.push_back(Be_j_i);}
            m_Be.push_back(Be_j);}
        //station-dependent covariates
        m_Bs.reserve(m_qs);
        for(std::size_t j = 0; j < m_qs; ++j){
            //for each event-dependent covariates
            std::size_t start_idx = j == 0    ? 0 : m_Ls_j[j-1];
            std::vector< FDAGWR_TRAITS::Dense_Matrix > Bs_j;
            Bs_j.reserve(this->n());
            //for all the units
            for(std::size_t i = 0; i < this->n(); ++i){
                //taking the right coefficients of the basis expansion
                FDAGWR_TRAITS::Dense_Matrix Bs_j_i = m_bs[i].block(start_idx,0,m_Ls_j[j],1);
                Bs_j.push_back(Bs_j_i);}
            m_Bs.push_back(Bs_j);}





/*
            std::cout << ".compute() done" <<std::endl;

            for(std::size_t i = 0; i < m_Bc.size(); ++i){
                std::cout << "Cov C " << i+1 << "-th dim: " << m_Bc[i].rows() << " rows, " << m_Bc[i].cols() << "cols" << std::endl;
                std::cout << m_Bc[i] << std::endl;}

            for(std::size_t i = 0; i < m_Be.size(); ++i){
                std::cout << "Cov E " << i+1 << "-th" << std::endl;
                for(std::size_t j = 0;  j < m_Be[i].size(); ++j){
                    std::cout << "Unit " << j+1 << "-th" << m_Be[i][j].rows() << " rows, " << m_Be[i][j].cols() << "cols" << std::endl;
                    std::cout << m_Be[i][j] << std::endl;}}

            for(std::size_t i = 0; i < m_Bs.size(); ++i){
                std::cout << "Cov S " << i+1 << "-th" << std::endl;
                for(std::size_t j = 0;  j < m_Bs[i].size(); ++j){
                    std::cout << "Unit " << j+1 << "-th" << m_Bs[i][j].rows() << " rows, " << m_Bs[i][j].cols() << "cols" << std::endl;
                    std::cout << m_Bs[i][j] << std::endl;}}
*/
    }

    /*!
    * @brief Virtual method to obtain a discrete version of the betas
    */
    inline 
    void 
    evalBetas()
    override
    {
        //BETA_C
        m_beta_c.reserve(m_qc);

        for(std::size_t j = 0; j < m_qc; ++j)
        {
            //retrieving the basis
            std::vector< FUNC_OBJ<INPUT,OUTPUT> > basis_j;
            basis_j.reserve(m_Bc[j].rows());
            std::size_t start_idx = j == 0    ? 0 : m_Lc_j[j-1];
            for(std::size_t k = start_idx; k < m_Lc_j[j]; ++k)
            {
                basis_j.push_back(m_omega(j,k));
            }
            functional_matrix<INPUT,OUTPUT> basis_c_j(basis_j,1,m_Bc[j].rows());

            //compute the beta
            FUNC_OBJ<INPUT,OUTPUT> beta_c_j = fm_prod<INPUT,OUTPUT>(basis_c_j,m_Bc[j],this->number_threads())(0,0);
            //eval the beta
            std::vector< OUTPUT > beta_c_j_ev; 
            beta_c_j_ev.resize(this->abscissa_points().size());
            std::transform(this->abscissa_points().cbegin(),this->abscissa_points().cend(),beta_c_j_ev.begin(),[&beta_c_j](const INPUT &x){return beta_c_j(x);});
            m_beta_c.push_back(beta_c_j_ev);
        }


/*

        //BETA_E
        m_beta_e.reserve(m_qe);

        for(std::size_t j = 0; j < m_qe; ++j)
        {
            //retrieving the basis
            std::vector< FUNC_OBJ<INPUT,OUTPUT> > basis_j;
            basis_j.reserve(m_Be[j][0].rows());
            std::size_t start_idx = j == 0    ? 0 : m_Le_j[j-1];
            for(std::size_t k = start_idx; k < m_Le_j[j]; ++k)
            {
                basis_j.push_back(m_theta(j,k));
            }
            functional_matrix<INPUT,OUTPUT> basis_e_j(basis_j,1,m_Be[j][0].rows());

            //evaluating the betas in every unit
            std::vector< std::vector<OUTPUT> > beta_e_j_ev;
            beta_e_j_ev.reserve(this->n());
            for(std::size_t i = 0; i < this->n(); ++i)
            {
                //compute the beta j-th for unit i-th
                FUNC_OBJ<INPUT,OUTPUT> beta_e_j_i = fm_prod<INPUT,OUTPUT>(basis_e_j,m_Be[j][i],this->number_threads())(0,0);
                //eval the beta
                std::vector< OUTPUT > beta_e_j_i_ev; 
                beta_e_j_i_ev.reserve(this->abscissa_points().size());
                std::transform(this->abscissa_points().cbegin(),this->abscissa_points().cend(),beta_e_j_i_ev.begin(),[&beta_e_j_i](const INPUT &x){return beta_e_j_i(x);});
                beta_e_j_ev.push_back(beta_e_j_i_ev);
            }

            m_beta_e.push_back(beta_e_j_ev);
        }


        //BETA_S
        m_beta_s.reserve(m_qs);

        for(std::size_t j = 0; j < m_qs; ++j)
        {
            //retrieving the basis
            std::vector< FUNC_OBJ<INPUT,OUTPUT> > basis_j;
            basis_j.reserve(m_Bs[j][0].rows());
            std::size_t start_idx = j == 0    ? 0 : m_Ls_j[j-1];
            for(std::size_t k = start_idx; k < m_Ls_j[j]; ++k)
            {
                basis_j.push_back(m_psi(j,k));
            }
            functional_matrix<INPUT,OUTPUT> basis_s_j(basis_j,1,m_Bs[j][0].rows());

            //evaluating the betas in every unit
            std::vector< std::vector<OUTPUT> > beta_s_j_ev;
            beta_s_j_ev.reserve(this->n());
            for(std::size_t i = 0; i < this->n(); ++i)
            {
                //compute the beta j-th for unit i-th
                FUNC_OBJ<INPUT,OUTPUT> beta_s_j_i = fm_prod<INPUT,OUTPUT>(basis_s_j,m_Bs[j][i],this->number_threads())(0,0);
                //eval the beta
                std::vector< OUTPUT > beta_s_j_i_ev; 
                beta_s_j_i_ev.reserve(this->abscissa_points().size());
                std::transform(this->abscissa_points().cbegin(),this->abscissa_points().cend(),beta_s_j_i_ev.begin(),[&beta_s_j_i](const INPUT &x){return beta_s_j_i(x);});
                beta_s_j_ev.push_back(beta_s_j_i_ev);
            }
            
            m_beta_s.push_back(beta_s_j_ev);
        }

*/
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
        return std::tuple{m_Bc,m_Be,m_Bs};
    }

    /*!
    * @brief
    */
    inline 
    BetasTuple 
    betas() 
    const
    override
    {
        return std::tuple{m_beta_c,m_beta_e,m_beta_s};
    }


};

#endif  /*FGWR_FMS_ESC_ALGO_HPP*/