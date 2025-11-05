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
// OUT OF OR IN CONNECTION WITH fdagwr OR THE USE OR OTHER DEALINGS IN
// fdagwr.


#ifndef FWR_FMSGWR_SEC_ALGO_HPP
#define FWR_FMSGWR_SEC_ALGO_HPP

#include "fwr.hpp"


/*!
* @file fwr_FMSGWR_SEC.hpp
* @brief Contains the definition of the Functional Multi-Source Geographically Weighted Regression SEC model
* @author Andrea Enrico Franzoni
*/


/*!
* @class fwr_FMSGWR_SEC
* @brief Concrete class for the Functional Multi-Source Geographically Weighted Regression SEC model, estimating, in order, stationary, event-dependent and station-dependent functional regression coefficients
* @tparam INPUT type of functional data abscissa
* @tparam OUTPUT type of functional data image
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_FMSGWR_SEC final : public fwr<INPUT,OUTPUT>
{
private:
    /*!Functional response (nx1)*/
    functional_matrix<INPUT,OUTPUT> m_y;
    /*!Basis for response (nx(n*Ly)). Each row contains the basis, every time shifted by (row_i-1)+Ly, always the same basis*/
    functional_matrix_sparse<INPUT,OUTPUT> m_phi;
    /*!Coefficients of the basis expansion for response ((n*Ly)x1): coefficients for each unit are columnized one below the other*/
    FDAGWR_TRAITS::Dense_Matrix m_c;
    /*!Number of basis used to make basis expansion for the response*/
    std::size_t m_Ly;
    /*!Basis used for the response (the functions put in m_phi)*/
    std::unique_ptr<basis_base_class<FDAGWR_TRAITS::basis_geometry>> m_basis_y;
    /*!Knots for the response, used at the beginning to obtain y basis expansion coefficients via smoothing*/
    FDAGWR_TRAITS::Dense_Matrix m_knots_smoothing;

    /*!Functional stationary covariates (n x qc)*/
    functional_matrix<INPUT,OUTPUT> m_Xc;
    /*!Their transpost (qc x n)*/
    functional_matrix<INPUT,OUTPUT> m_Xc_t;
    /*!Functional weights for stationary covariates (diagonal n x n)*/
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
    std::vector< std::vector< OUTPUT >> m_beta_c;
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
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> m_Be;
    /*!Discrete evaluation of all the beta_e: a vector of dimension qe, containing, for all the event-dependent covariates, the discrete ev of the respective beta, for each statistical unit*/
    std::vector< std::vector< std::vector< OUTPUT >>> m_beta_e;
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
    std::vector< std::vector< std::vector< OUTPUT >>> m_beta_s;
    /*!Number of station-dependent covariates*/
    std::size_t m_qs;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the station-dependent regressors coefficients*/
    std::size_t m_Ls; 
    /*!Number of basis, for each station-dependent covariate, to perform the basis expansion of the regressors coefficients for the station-dependent regressors coefficients*/
    std::vector<std::size_t> m_Ls_j;

    //A operators
    /*!A_S_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_A_s;  
    /*!A_E_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_A_e;
    /*!A_SE_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_A_se;

    //B operators
    /*!B_S_i while computing K_E_S(t)*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_s_for_K_s_e;
    /*!B_S_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_s;
    /*!B_E_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_e;
    /*!B_SE_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_se;

    /*!c_tilde_hat (necessary to save partial residuals and performing predictions)*/
    FDAGWR_TRAITS::Dense_Matrix m_c_tilde_hat;



public:
    /*!
    * @brief Constructor
    * @param y functional matrix containing the response (nx1)
    * @param phi functional sparse matrix containing the basis for the response (nx(n*Ly)). Each row contains the basis, every time shifted by (row_i-1)+Ly, always the same basis
    * @param c response basis expansion coefficients ((n*Ly)x1)
    * @param Ly number of basis for the response
    * @param basis_y basis for the the response, the same basis put in phi
    * @param knots_smoothing knots used to perform the smoothing for the response without the effect of the stationary part of the model
    * @param Xc functional matrix containing the stationary covariates (nxqc)
    * @param Wc functional diagonal matrix containing the functional stationary weights (nxn)
    * @param Rc penalization matrix of the stationary covariates (diagonal block matrix containing the the scalar product within the second order derivatives of the functional regression coefficients basis. LcxLc)
    * @param omega functional sparse matrix containing the basis of the functional regression coefficients of the stationary covariates (qcxLc, row i-th contains zeros and the basis of the i-th stationary covariate, their position shifted of sum_i_0_to_i(Lc_i))
    * @param qc number of stationary covariates
    * @param Lc total number of basis used for the stationary covariates functional regression coefficients
    * @param Lc_j vector containing in element i-th the number of basis for the stationary covariate i-th functional regression coefficients
    * @param Xe functional matrix containing the event-dependent covariates (nxqe)
    * @param We vector of functional diagonal matrix containing, as element i-th, the functional event-dependent weights (nxn) of unit i-th
    * @param Re penalization matrix of the event-dependent covariates (diagonal block matrix containing the the scalar product within the second order derivatives of the functional regression coefficients basis. LexLe)
    * @param theta functional sparse matrix containing the basis of the functional regression coefficients of the event-dependent covariates (qexLe, row i-th contains zeros and the basis of the i-th event-dependent covariate, their position shifted of sum_i_0_to_i(Le_i))
    * @param qe number of event-dependent covariates
    * @param Le total number of basis used for the event-dependent covariates functional regression coefficients
    * @param Le_j vector containing in element i-th the number of basis for the event-dependent covariate i-th functional regression coefficients
    * @param Xs functional matrix containing the station-dependent covariates (nxqs)
    * @param Ws vector of functional diagonal matrix containing, as element i-th, the functional station-dependent weights (nxn) of unit i-th
    * @param Rs penalization matrix of the station-dependent covariates (diagonal block matrix containing the the scalar product within the second order derivatives of the functional regression coefficients basis. LsxLs)
    * @param psi functional sparse matrix containing the basis of the functional regression coefficients of the station-dependent covariates (qsxLs, row i-th contains zeros and the basis of the i-th station-dependent covariate, their position shifted of sum_i_0_to_i(Ls_i))
    * @param qs number of station-dependent covariates
    * @param Ls total number of basis used for the station-dependent covariates functional regression coefficients
    * @param Ls_j vector containing in element i-th the number of basis for the station-dependent covariate i-th functional regression coefficients
    * @param a left extreme functional data domain 
    * @param a right extreme functional data domain 
    * @param n_intervals_integration number of intervals used by the midpoint quadrature rule
    * @param abscissa_points abscissa points over which there are the evaluations of the raw functional data
    * @param n number of training statistical units
    * @param number_threads number of threads for OMP
    * @param in_cascade_estimation if true, for more than one source covariates, the estimation is made in cascade. If false, exact
    * @note input dimensions check and transpose computation
    */
    template<typename FUNC_MATRIX_OBJ, 
             typename FUNC_SPARSE_MATRIX_OBJ,
             typename FUNC_DIAG_MATRIX_OBJ, 
             typename FUNC_DIAG_MATRIX_VEC_OBJ, 
             typename SCALAR_MATRIX_OBJ, 
             typename SCALAR_SPARSE_MATRIX_OBJ> 
    fwr_FMSGWR_SEC(FUNC_MATRIX_OBJ &&y,
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
                   const std::vector<INPUT> & abscissa_points,
                   std::size_t n,
                   int number_threads,
                   bool in_cascade_estimation)
        :
            fwr<INPUT,OUTPUT>(a,b,n_intervals_integration,abscissa_points,n,number_threads,in_cascade_estimation),
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
                assert((m_Lc_j.size() == m_qc) && (std::reduce(m_Lc_j.cbegin(),m_Lc_j.cend(),static_cast<std::size_t>(0)) == m_Lc));
                //event-dependent covariates
                assert((m_Xe.rows() == this->n()) && (m_Xe.cols() == m_qe));
                assert(m_We.size() == this->n());
                for(std::size_t i = 0; i < m_We.size(); ++i){   assert((m_We[i].rows() == this->n()) && (m_We[i].cols() == this->n()));}
                assert((m_Re.rows() == m_Le) && (m_Re.cols() == m_Le));
                assert((m_theta.rows() == m_qe) && (m_theta.cols() == m_Le));
                assert((m_Le_j.size() == m_qe) && (std::reduce(m_Le_j.cbegin(),m_Le_j.cend(),static_cast<std::size_t>(0)) == m_Le));
                //station-dependent covariates
                assert((m_Xs.rows() == this->n()) && (m_Xs.cols() == m_qs));
                assert(m_Ws.size() == this->n());
                for(std::size_t i = 0; i < m_Ws.size(); ++i){   assert((m_Ws[i].rows() == this->n()) && (m_Ws[i].cols() == this->n()));}
                assert((m_Rs.rows() == m_Ls) && (m_Rs.cols() == m_Ls));
                assert((m_psi.rows() == m_qs) && (m_psi.cols() == m_Ls));
                assert((m_Ls_j.size() == m_qs) && (std::reduce(m_Ls_j.cbegin(),m_Ls_j.cend(),static_cast<std::size_t>(0)) == m_Ls));

                //compute all the transpost necessary for the computations
                m_Xc_t = m_Xc.transpose();
                m_omega_t = m_omega.transpose();
                m_Xe_t = m_Xe.transpose();
                m_theta_t = m_theta.transpose();
                m_Xs_t = m_Xs.transpose();
                m_psi_t = m_psi.transpose();
            }
    

    /*!
    * @brief Method to compute the Functional Weighted Regression basis expansion coefficients of the functional regression coefficients
    */
    inline 
    void 
    compute()  
    override
    { 
        //exact estimation
        if (!this->in_cascade_estimation())
        {
            //(j_tilde_tilde + Rs)^-1 (LsxLs)
            std::cout << "Computing (j_tilde_tilde + Rs)^-1" << std::endl;
            std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_double_tilde_Rs_inv = this->operator_comp().compute_penalty(m_psi_t,m_Xs_t,m_Ws,m_Xs,m_psi,m_Rs);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
            //A_S_i (n elements Lsx(n*Ly))
            std::cout << "Computing A_s_i" << std::endl;
            m_A_s = this->operator_comp().compute_operator(m_psi_t,m_Xs_t,m_Ws,m_phi,j_double_tilde_Rs_inv);
            //H_s(t) (nx(n*Ly))
            std::cout << "Computing H_s(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> H_s = this->operator_comp().compute_functional_operator(m_Xs,m_psi,m_A_s);
            //B_S_i (n elements LsxLc)
            std::cout << "Computing B_s_i" << std::endl;
            m_B_s = this->operator_comp().compute_operator(m_psi_t,m_Xs_t,m_Ws,m_Xc,m_omega,j_double_tilde_Rs_inv);
            //K_s_c(t) (nxLc)
            std::cout << "Computing K_s_c(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> K_s_c = this->operator_comp().compute_functional_operator(m_Xs,m_psi,m_B_s);
            //B_S_i_for_K_s_e (n elements LsxLe)
            std::cout << "Computing B_S_i_for_K_s_e" << std::endl;
            m_B_s_for_K_s_e = this->operator_comp().compute_operator(m_psi_t,m_Xs_t,m_Ws,m_Xe,m_theta,j_double_tilde_Rs_inv);
            //K_s_e(t) (nxLe)
            std::cout << "Computing K_s_e(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> K_s_e = this->operator_comp().compute_functional_operator(m_Xs,m_psi,m_B_s_for_K_s_e);
            //X_e_crossed(t) (nxLe)
            functional_matrix<INPUT,OUTPUT> X_e_crossed = fm_prod(m_Xe,m_theta) - K_s_e;
            functional_matrix<INPUT,OUTPUT> X_e_crossed_t = X_e_crossed.transpose();
            //(j_tilde + Re)^-1 (LexLe)
            std::cout << "Computing (j_tilde + Re)^-1" << std::endl;
            std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_tilde_Re_inv = this->operator_comp().compute_penalty(X_e_crossed_t,m_We,X_e_crossed,m_Re);
        
            //A_E_i (n elements Lex(n*Ly))
            functional_matrix<INPUT,OUTPUT> rhs_Ae = m_phi - H_s;
            std::cout << "Computing A_e_i" << std::endl;
            m_A_e = this->operator_comp().compute_operator(X_e_crossed_t,m_We,rhs_Ae,j_tilde_Re_inv);
            //H_e(t) (nx(n*Ly))
            std::cout << "Computing H_e(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> H_e = this->operator_comp().compute_functional_operator(m_Xe,m_theta,m_A_e);
            //A_SE_i (n elements Lsx(n*Ly))
            std::cout << "Computing A_se_i" << std::endl;
            m_A_se = this->operator_comp().compute_operator(m_psi_t,m_Xs_t,m_Ws,H_e,j_double_tilde_Rs_inv);
            //H_se(t) (nx(n*Ly))
            std::cout << "Computing H_se(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> H_se = this->operator_comp().compute_functional_operator(m_Xs,m_psi,m_A_se);

            //B_E_i (n elements LexLc)
            functional_matrix<INPUT,OUTPUT> rhs_Be = fm_prod(m_Xc,m_omega) - K_s_c;
            std::cout << "Computing B_e_i" << std::endl;
            m_B_e = this->operator_comp().compute_operator(X_e_crossed_t,m_We,rhs_Be,j_tilde_Re_inv);
            //K_e_c(t) (nXLc)
            std::cout << "Computing K_e_c(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> K_e_c = this->operator_comp().compute_functional_operator(m_Xe,m_theta,m_B_e);
            //B_SE_i (n elements LsxLc)
            std::cout << "Computing B_se_i" << std::endl;
            m_B_se = this->operator_comp().compute_operator(m_psi_t,m_Xs_t,m_Ws,K_e_c,j_double_tilde_Rs_inv);
            //K_se_c(t) (nxLc)
            std::cout << "Computing K_se_c(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> K_se_c = this->operator_comp().compute_functional_operator(m_Xs,m_psi,m_B_se);

            //y_new(t) (nx1)
            std::cout << "Computing y_new(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> y_new = fm_prod(functional_matrix<INPUT,OUTPUT>(m_phi - H_s - H_e + H_se),m_c,this->number_threads());
            //X_c_crossed (nxqc)
            functional_matrix<INPUT,OUTPUT> X_c_crossed = fm_prod(m_Xc,m_omega) - K_s_c - K_e_c + K_se_c;
            functional_matrix<INPUT,OUTPUT> X_c_crossed_t = X_c_crossed.transpose();
            //[J + Rc]^-1(LcxLc)
            std::cout << "Computing (j + Rc)^-1" << std::endl;
            Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> j_Rc_inv = this->operator_comp().compute_penalty(X_c_crossed_t,m_Wc,X_c_crossed,m_Rc);
        

            //COMPUTING m_bc, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATIONARY BETAS (Lcx1)
            std::cout << "Computing bc" << std::endl;
            m_bc = this->operator_comp().compute_operator(X_c_crossed_t,m_Wc,y_new,j_Rc_inv);


            //y_tilde_hat(t) (nx1)
            std::cout << "Computing y_tilde_hat(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> y_tilde_hat = m_y - fm_prod(fm_prod(m_Xc,m_omega),m_bc,this->number_threads());
            //c_tilde_hat: smoothing on y_tilde_hat(t) with respect of the basis of y ((n*Ly)x1)
            std::cout << "Computing c_tilde_hat" << std::endl;
            m_c_tilde_hat = columnize_coeff_resp(fm_smoothing<INPUT,OUTPUT,FDAGWR_TRAITS::basis_geometry>(y_tilde_hat,*m_basis_y,m_knots_smoothing));
            //y_tilde_new(t) (nx1)
            std::cout << "Computing y_tilde_new(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> y_tilde_new = fm_prod(functional_matrix<INPUT,OUTPUT>(m_phi - H_s),m_c_tilde_hat,this->number_threads());

            //COMPUTING all the m_be, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE EVENT-DEPENDENT BETAS (n elements Lex1)
            std::cout << "Computing be" << std::endl;
            m_be = this->operator_comp().compute_operator(X_e_crossed_t,m_We,y_tilde_new,j_tilde_Re_inv);

            //y_tilde_tilde_hat(t) (nx1)
            std::cout << "Computing y_tilde_tilde_hat(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> y_tilde_tilde_hat = y_tilde_hat - this->operator_comp().compute_functional_operator(m_Xe,m_theta,m_be);

            //COMPUTING all the m_bs, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATION-DEPENDENT BETAS (n elementd Lsx1)
            std::cout << "Computing bs" << std::endl;
            m_bs = this->operator_comp().compute_operator(m_psi_t,m_Xs_t,m_Ws,y_tilde_tilde_hat,j_double_tilde_Rs_inv);
        }
        //in cascade estimation
        else
        {
            //[J + Rc]^-1 (LcxLc)
            std::cout << "Computing (j + Rc)^-1" << std::endl;
            Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> j_Rc_inv = this->operator_comp().compute_penalty(m_omega_t,m_Xc_t,m_Wc,m_Xc,m_omega,m_Rc);
            //COMPUTING m_bc, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATIONARY BETAS (Lcx1)
            std::cout << "Computing bc" << std::endl;
            m_bc = this->operator_comp().compute_operator(m_omega_t,m_Xc_t,m_Wc,m_y,j_Rc_inv);
            //y_tilde (nx1)
            functional_matrix<INPUT,OUTPUT> y_tilde = m_y - fm_prod(fm_prod(m_Xc,m_omega),m_bc,this->number_threads());
            //[J_i + Re]^-1 (LexLe)
            std::cout << "Computing (j + Re)^-1" << std::endl;
            std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_i_Re_inv = this->operator_comp().compute_penalty(m_theta_t,m_Xe_t,m_We,m_Xe,m_theta,m_Re);
            //COMPUTING m_be, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATIONARY BETAS (n elements Lex1)
            std::cout << "Computing be" << std::endl;
            m_be = this->operator_comp().compute_operator(m_theta_t,m_Xe_t,m_We,y_tilde,j_i_Re_inv);
            //y_tilde_tilde (nx1)
            functional_matrix<INPUT,OUTPUT> y_tilde_tilde(this->n(),1); 

            //extra objects (default values of 0s)
            m_A_s.resize(this->n());
            m_B_s_for_K_s_e.resize(this->n());

#ifdef _OPENMP
#pragma omp parallel for shared(y_tilde_tilde,m_Xe,m_theta,y_tilde) num_threads(this->number_threads())
#endif
            for(std::size_t i = 0; i < this->n(); ++i)
            {
                std::vector< FUNC_OBJ<INPUT,OUTPUT> > xe_i(m_Xe.row(i).cbegin(),m_Xe.row(i).cend()); //1xqe
                functional_matrix<INPUT,OUTPUT> Xe_i(xe_i,1,m_qe);
                functional_matrix<INPUT,OUTPUT> y_tilde_i(1,1,y_tilde(i,0));
                y_tilde_tilde(i,0) = (y_tilde_i - fm_prod(fm_prod(Xe_i,m_theta),m_be[i],this->number_threads()))(0,0);

                //default values of 0 for returning elements
                m_A_s[i] = FDAGWR_TRAITS::Dense_Matrix::Zero(m_Ls,m_Ly*this->n());
                m_B_s_for_K_s_e[i] = FDAGWR_TRAITS::Dense_Matrix::Zero(m_Ls,m_Le);
            }

            //[J_i + Rs]^-1 (LsxLs)
            std::cout << "Computing (j + Rs)^-1" << std::endl;
            std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_i_Rs_inv = this->operator_comp().compute_penalty(m_psi_t,m_Xs_t,m_Ws,m_Xs,m_psi,m_Rs);
            //COMPUTING m_bs, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATIONARY BETAS (n elements Lsx1)
            std::cout << "Computing bs" << std::endl;
            m_bs = this->operator_comp().compute_operator(m_psi_t,m_Xs_t,m_Ws,y_tilde_tilde,j_i_Rs_inv);
            //default values of 0 for returning elements
            m_c_tilde_hat = FDAGWR_TRAITS::Dense_Matrix::Zero(m_Ly*this->n(),1);
        }
        
        //
        //wrapping the b from the shape useful for the computation into a more useful format
        //
        //stationary covariates (qc elements 1xLc_i)
        m_Bc = this->operator_comp().wrap_operator(m_bc,m_Lc_j,m_qc);
        //event-dependent covariates (qe elements of n elements 1xLe_i)
        m_Be = this->operator_comp().wrap_operator(m_be,m_Le_j,m_qe,this->n());
        //station-dependent covariates (qs elements of n elements 1xLs_i)
        m_Bs = this->operator_comp().wrap_operator(m_be,m_Ls_j,m_qs,this->n());
    }

    /*!
    * @brief Evaluating the functional regression coefficients over a grid of points (m_abscissa_points)
    */
    inline 
    void 
    evalBetas()
    override
    {
        //BETA_C
        m_beta_c = this->operator_comp().eval_func_betas(m_Bc,m_omega,m_Lc_j,m_qc,this->abscissa_points());        
        //BETA_E
        m_beta_e = this->operator_comp().eval_func_betas(m_Be,m_theta,m_Le_j,m_qe,this->n(),this->abscissa_points());
        //BETA_S
        m_beta_s = this->operator_comp().eval_func_betas(m_Bs,m_psi,m_Ls_j,m_qs,this->n(),this->abscissa_points());
    }

    /*!
    * @brief Function to return the basis expansion coefficients of the functional regression coefficitens
    * @return a tuple containing m_Bc,m_Be and m_Bs
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
    * @brief Function to return the the functional regression coefficients evaluated
    * @return a tuple containing m_beta_c,m_beta_e and m_beta_s
    */
    inline 
    BetasTuple 
    betas() 
    const
    override
    {
        return std::tuple{m_beta_c,m_beta_e,m_beta_s};
    }

    /*!
    * @brief Function to return objects useful for reconstructing the functional partial residuals
    * @return a tuple containing m_c_tilde_hat,m_A_s and m_B_s_for_K_s_e
    */
    inline
    PartialResidualTuple
    PRes()
    const
    override
    {
        return std::tuple{m_c_tilde_hat,m_A_s,m_B_s_for_K_s_e};
    }
};

#endif  /*FWR_FMSGWR_SEC_ALGO_HPP*/