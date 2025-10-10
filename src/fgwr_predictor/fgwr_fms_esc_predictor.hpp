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


#ifndef FGWR_FMS_ESC_PREDICT_HPP
#define FGWR_FMS_ESC_PREDICT_HPP

#include "fgwr_predictor.hpp"

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fgwr_fms_esc_predictor final : public fgwr_predictor<INPUT,OUTPUT>
{
private:
    //
    //FITTED MODEL
    //
    //Basis expansion coefficients for the regression coefficients: fitted values
    /*!Coefficients of the basis expansion for stationary regressors coefficients: Lcx1*/
    FDAGWR_TRAITS::Dense_Matrix m_bc_fitted;
    /*!Coefficients of the basis expansion for stationary regressors coefficients: every element is Lc_jx1*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_Bc_fitted;
    /*!Coefficients of the basis expansion for station-dependent covariates regressors: Lsx1, every element of the vector is referring to a specific unit TO BE COMPUTED*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_bs_fitted;
    /*!Coefficients of the basis expansion for station-dependent regressors coefficients: every of the qe elements are n 1xLs_j matrices, one for each statistical unit*/
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> m_Bs_fitted;
    //
    //Computing the betas in the new stance
    //
    /*!Coefficients of the basis expansion for event-dependent regressors: Lex1, every element of the vector is referring to a specific unit TO BE COMPUTED*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_be_pred;
    /*!Coefficients of the basis expansion for event-dependent regressors coefficients: every of the qe elements are n 1xLe_j matrices, one for each statistical unit*/
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> m_Be_pred;
    /*!Coefficients of the basis expansion for station-dependent covariates regressors: Lsx1, every element of the vector is referring to a specific unit TO BE COMPUTED*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_bs_pred;
    /*!Coefficients of the basis expansion for station-dependent regressors coefficients: every of the qe elements are n 1xLs_j matrices, one for each statistical unit*/
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> m_Bs_pred;

    //Basis used for the regression coefficients
    /*!Basis for stationary covariates regressors (sparse qc x Lc)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_omega;
    /*!Their transpost (sparse Lc x qC)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_omega_t;
    /*!Number of stationary covariates*/
    std::size_t m_qc;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the stationary regressors coefficients*/
    std::size_t m_Lc; 
    /*!Number of basis, for each stationary covariate, to perform the basis expansion of the regressors coefficients for the stationary regressors coefficients*/
    std::vector<std::size_t> m_Lc_j;
    /*!Basis for event-dependent covariates regressors (sparse qe x Le)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_theta;
    /*!Their transpost (sparse Le x qE)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_theta_t;
    /*!Number of event-dependent covariates*/
    std::size_t m_qe;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the event-dependent regressors coefficients*/
    std::size_t m_Le; 
    /*!Number of basis, for each event-dependent covariate, to perform the basis expansion of the regressors coefficients for the event-dependent regressors coefficients*/
    std::vector<std::size_t> m_Le_j;
    /*!Basis for station-dependent covariates regressors (sparse qs x Ls)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_psi;
    /*!Their transpost (sparse Ls x qS)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_psi_t;
    /*!Number of station-dependent covariates*/
    std::size_t m_qs;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the station-dependent regressors coefficients*/
    std::size_t m_Ls; 
    /*!Number of basis, for each station-dependent covariate, to perform the basis expansion of the regressors coefficients for the station-dependent regressors coefficients*/
    std::vector<std::size_t> m_Ls_j;

    //Objects to reconstruct the functional partial residuals
    /*!Basis used for response (n_trainx(n_trainxxLy)) in the training process*/
    functional_matrix_sparse<INPUT,OUTPUT> m_phi;
    /*!Number of basis used to make basis expansion for y in the training process*/
    std::size_t m_Ly;
    /*!c_tilde_hat ((n_trainxLy)x1)*/
    FDAGWR_TRAITS::Dense_Matrix m_c_tilde_hat;
    /*!A_E_i n_train matrices n_trainx(n_train*Ly)*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_A_e;
    /*!B_E_i while computing K_E_S(t)*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_e_for_K_e_s;
    /*!Functional event-dependent covariates (n_train x qe)*/
    functional_matrix<INPUT,OUTPUT> m_Xe_train;
    /*!Their transpost (qe x n_train)*/
    functional_matrix<INPUT,OUTPUT> m_Xe_train_t;
    /*!Scalar matrix with the penalization on the event-dependent covariates (sparse Le x Le, where Le is the sum of the basis of each E covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Re;
    /*!Functional station-dependent covariates (n_train x qs)*/
    functional_matrix<INPUT,OUTPUT> m_Xs_train;
    /*!Their transpost (qs x n_train)*/
    functional_matrix<INPUT,OUTPUT> m_Xs_train_t;
    /*!Scalar matrix with the penalization on the station-dependent covariates (sparse Ls x Ls, where Ls is the sum of the basis of each S covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Rs;

    //Part to be computed
    functional_matrix<INPUT,OUTPUT> m_K_e_s;
    functional_matrix<INPUT,OUTPUT> m_X_s_train_crossed;
    functional_matrix<INPUT,OUTPUT> m_X_s_train_crossed_t;
    functional_matrix<INPUT,OUTPUT> m_y_tilde_hat;
    functional_matrix<INPUT,OUTPUT> m_H_e;
    functional_matrix<INPUT,OUTPUT> m_y_tilde_new;
    functional_matrix<INPUT,OUTPUT> m_y_tilde_tilde_hat;


    //Betas
    //stationary: qc x 1
    functional_matrix<INPUT,OUTPUT> m_BetaC;
    //events-dependent: n_pred matrices of qe x 1
    std::vector< functional_matrix<INPUT,OUTPUT>> m_BetaE;
    //stations-dependent: n_pred matrices of qs x 1
    std::vector< functional_matrix<INPUT,OUTPUT>> m_BetaS;

public:
    /*!
    * @brief Constructor
    */
    template<typename FUNC_MATRIX_OBJ, 
             typename FUNC_SPARSE_MATRIX_OBJ,
             typename SCALAR_MATRIX_OBJ, 
             typename SCALAR_MATRIX_OBJ_VEC,
             typename SCALAR_MATRIX_OBJ_VEC_VEC,
             typename SCALAR_SPARSE_MATRIX_OBJ>
    fgwr_fms_esc_predictor(SCALAR_MATRIX_OBJ_VEC &&Bc_fitted,
                           SCALAR_MATRIX_OBJ_VEC_VEC &&Bs_fitted,
                           FUNC_SPARSE_MATRIX_OBJ &&omega,
                           std::size_t qc,
                           std::size_t Lc,
                           const std::vector<std::size_t> &Lc_j,
                           FUNC_SPARSE_MATRIX_OBJ &&theta,
                           std::size_t qe,
                           std::size_t Le,
                           const std::vector<std::size_t> &Le_j,
                           FUNC_SPARSE_MATRIX_OBJ &&psi,
                           std::size_t qs,
                           std::size_t Ls,
                           const std::vector<std::size_t> &Ls_j,
                           FUNC_SPARSE_MATRIX_OBJ &&phi,
                           std::size_t Ly,
                           SCALAR_MATRIX_OBJ &&c_tilde_hat,
                           SCALAR_MATRIX_OBJ_VEC &&A_e,
                           SCALAR_MATRIX_OBJ_VEC &&B_e_for_K_e_s,
                           FUNC_MATRIX_OBJ &&Xe_train,
                           SCALAR_SPARSE_MATRIX_OBJ &&Re,
                           FUNC_MATRIX_OBJ &&Xs_train,
                           SCALAR_SPARSE_MATRIX_OBJ &&Rs,
                           INPUT a, 
                           INPUT b, 
                           int n_intervals_integration, 
                           double target_error, 
                           int max_iterations, 
                           std::size_t n_train, 
                           int number_threads)
            :   
                    fgwr_predictor<INPUT,OUTPUT>(a,b,n_intervals_integration,target_error,max_iterations,n_train,number_threads),
                    m_Bc_fitted{std::forward<SCALAR_MATRIX_OBJ_VEC>(Bc_fitted)},
                    m_Bs_fitted{std::forward<SCALAR_MATRIX_OBJ_VEC_VEC>(Bs_fitted)},
                    m_omega{std::forward<FUNC_SPARSE_MATRIX_OBJ>(omega)},
                    m_qc(qc),
                    m_Lc(Lc),
                    m_Lc_j(Lc_j),
                    m_theta{std::forward<FUNC_SPARSE_MATRIX_OBJ>(theta)},
                    m_qe(qe),
                    m_Le(Le),
                    m_Le_j(Le_j),
                    m_psi{std::forward<FUNC_SPARSE_MATRIX_OBJ>(psi)},
                    m_qs(qs),
                    m_Ls(Ls),
                    m_Ls_j(Ls_j),
                    m_phi{std::forward<FUNC_SPARSE_MATRIX_OBJ>(phi)},
                    m_Ly(Ly),
                    m_c_tilde_hat{std::forward<SCALAR_MATRIX_OBJ>(c_tilde_hat)},
                    m_A_e{std::forward<SCALAR_MATRIX_OBJ_VEC>(A_e)},
                    m_B_e_for_K_e_s{std::forward<SCALAR_MATRIX_OBJ_VEC>(B_e_for_K_e_s)},
                    m_Xe_train{std::forward<FUNC_MATRIX_OBJ>(Xe_train)},
                    m_Re{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Re)},
                    m_Xs_train{std::forward<FUNC_MATRIX_OBJ>(Xs_train)},
                    m_Rs{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rs)}
            {
                //input coherency
                assert(m_Bc_fitted.size() == m_qc);
                assert(m_Bs_fitted.size() == m_qs);
                for(std::size_t j = 0; j < m_qs; ++j){  assert(m_Bs_fitted[j].size() == this->n_train());}
                assert((m_omega.rows() == m_qc) && (m_omega.cols() == Lc));
                assert((m_theta.rows() == m_qe) && (m_theta.cols() == Le));
                assert((m_psi.rows() == m_qs) && (m_psi.cols() == Ls));
                assert((m_phi.rows() == this->n_train()) && (m_phi.cols() == (this->n_train()*m_Ly)));
                assert((m_c_tilde_hat.rows() == (this->n_train()*m_Ly)) && (m_c_tilde_hat.cols() == 1));
                assert(m_A_e.size() == this->n_train());
                for(std::size_t i = 0; i < this->n_train(); ++i){   assert((m_A_e[i].rows() == m_Le) && (m_A_e[i].cols() == (this->n_train()*m_Ly)));}
                assert(m_B_e_for_K_e_s.size() == this->n_train());
                for(std::size_t i = 0; i < this->n_train(); ++i){   assert((m_B_e_for_K_e_s[i].rows() == m_Le) && (m_B_e_for_K_e_s[i].cols() == m_Ls));}

                //compute the transpost
                m_omega_t = m_omega.transpose();
                m_theta_t = m_theta.transpose();
                m_psi_t = m_psi.transpose();
                m_Xe_train_t = m_Xe_train.transpose();
                m_Xs_train_t = m_Xs_train.transpose();

                //dewrappare i b_train (li incolonna)
                m_bc_fitted = this->dewrap_b(m_Bc_fitted,m_Lc_j);
                //bs_fitted
                m_bs_fitted = this->dewrap_b(m_Bs_fitted,m_Ls_j,this->n_train());
            }

    
    inline 
    void
    computePartialResiduals()
    override
    {
        //retrieve the partial residuals from the fitted model
        //K_e_s(t) n_train x Ls
        m_K_e_s = this->compute_functional_operator(m_Xe_train,m_theta,m_B_e_for_K_e_s);
        //X_s_crossed(t) n_train x Ls
        m_X_s_train_crossed = fm_prod(m_Xs_train,m_psi) - m_K_e_s;
        m_X_s_train_crossed_t = m_X_s_train_crossed.transpose();
        //y_tilde_hat(t) n_trainx1
        m_y_tilde_hat = fm_prod(m_phi,m_c_tilde_hat);
        //He(t) n_trainx(n_train*Ly)
        m_H_e = this->compute_functional_operator(m_Xe_train,m_theta,m_A_e);
        //y_tilde_new(t) n_trainx1
        m_y_tilde_new = fm_prod(functional_matrix<INPUT,OUTPUT>(m_phi - m_H_e),m_c_tilde_hat,this->number_threads());
        //y_tilde_tilde_hat(t) n_trainx1
        m_y_tilde_tilde_hat = m_y_tilde_hat - this->compute_functional_operator(m_Xs_train,m_psi,m_bs_fitted);
    }


    inline
    void
    computeBNew(const std::map<std::string,std::vector< functional_matrix_diagonal<INPUT,OUTPUT> >> &W)
    override
    {
        assert(W.size() == 2);
        std::string id_e = covariate_type<FDAGWR_COVARIATES_TYPES::EVENT>();
        std::string id_s = covariate_type<FDAGWR_COVARIATES_TYPES::STATION>();
        assert(W.at(id_e).size() == W.at(id_s).size());
        std::size_t n_pred = W.at(id_e).size();

        //compute the non-stationary betas in the new locations
        //penalties in the new locations
        //(j_tilde_tilde + Re)^-1
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_double_tilde_Re_inv = this->compute_penalty(m_theta_t,m_Xe_train_t,W.at(id_e),m_Xe_train,m_theta,m_Re);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
        //(j_tilde + Rs)^-1
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_tilde_Rs_inv        = this->compute_penalty(m_X_s_train_crossed_t,W.at(id_s),m_X_s_train_crossed,m_Rs);
        //COMPUTING all the m_bs in the new locations, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATION-DEPENDENT BETAS
        m_bs_pred = this->compute_operator(m_X_s_train_crossed_t,W.at(id_s),m_y_tilde_new,j_tilde_Rs_inv);
        //COMPUTING all the m_bs in the new locations, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATION-DEPENDENT BETAS
        m_be_pred = this->compute_operator(m_theta_t,m_Xe_train_t,W.at(id_e),m_y_tilde_tilde_hat,j_double_tilde_Re_inv);

        //
        //wrapping the b from the shape useful for the computation into a more useful format for reporting the results: TENERE
        //
        //event-dependent covariates
        m_Be_pred = this->wrap_b(m_be_pred,m_Le_j,m_qe,n_pred);
        //station-dependent covariates
        m_Bs_pred = this->wrap_b(m_bs_pred,m_Ls_j,m_qs,n_pred);


        //crearli come una std::function
    }

    /*!
    * @brief Compute stationary betas
    */
    inline
    void 
    computeStationaryBetas()
    override
    {
        m_BetaC = fm_prod(m_omega,m_bc_fitted);
    }

    /*!
    * @brief Compute non-stationary betas
    */
    inline
    void 
    computeNonStationaryBetas()
    override
    {
        std::size_t n_pred = m_be_pred.size();
        m_BetaE.resize(n_pred);
        m_BetaS.resize(n_pred);

#ifdef _OPENMP
#pragma omp parallel for shared(m_BetaE,m_BetaS,m_theta,m_psi,m_be_pred,m_bs_pred,n_pred) num_threads(this->number_threads())
#endif
        for(std::size_t i = 0; i < n_pred; ++i)
        {
            m_BetaE[i] = fm_prod(m_theta,m_be_pred[i]);
            m_BetaS[i] = fm_prod(m_psi,m_bs_pred[i]);
        }
    }

    /*!
    * @brief Compute prediction
    */
    inline
    functional_matrix<INPUT,OUTPUT>
    predict(const std::map<std::string,functional_matrix<INPUT,OUTPUT>> &X_new)
    const
    override
    {
        assert(X_new.size() == 3);
        std::string id_c = covariate_type<FDAGWR_COVARIATES_TYPES::STATIONARY>();
        std::string id_e = covariate_type<FDAGWR_COVARIATES_TYPES::EVENT>();
        std::string id_s = covariate_type<FDAGWR_COVARIATES_TYPES::STATION>();
        assert(X_new.at(id_c).size() == X_new.at(id_e).size());
        assert(X_new.at(id_e).size() == X_new.at(id_s).size());
        std::size_t n_pred = X_new.at(id_c).size();

        auto Xc_new = X_new.at(id_c);
        auto Xe_new = X_new.at(id_e);
        auto Xs_new = X_new.at(id_s);

        //y_new = X_new*beta = Xc_new*beta_c + Xe_new*beta_e + Xs_new*beta_s
        functional_matrix<INPUT,OUTPUT> y_new_C = fm_prod(Xc_new,m_BetaC,this->num_threads());    //n_pred x 1
        functional_matrix<INPUT,OUTPUT> y_new_E(n_pred,1);
        functional_matrix<INPUT,OUTPUT> y_new_S(n_pred,1);
        
#ifdef _OPENMP
#pragma omp parallel for shared(Xe_new,m_BetaE,y_new_E,Xs_new,m_BetaS,y_new_S,n_pred,m_qe,m_qs) num_threads(this->number_threads())
#endif
        for(std::size_t i = 0; i < n_pred; ++i)
        {
            std::vector< FUNC_OBJ<INPUT,OUTPUT> > xe_new_i(Xe_new.row(i).cbegin(),Xe_new.row(i).cend()); //1xqe
            functional_matrix<INPUT,OUTPUT> Xe_new_i(xe_new_i,1,m_qe);
            y_new_E(i,0) = fm_prod(Xe_new_i,m_BetaE[i],this->num_threads())(0,0);

            std::vector< FUNC_OBJ<INPUT,OUTPUT> > xs_new_i(Xs_new.row(i).cbegin(),Xs_new.row(i).cend()); //1xqs
            functional_matrix<INPUT,OUTPUT> Xs_new_i(xs_new_i,1,m_qs);
            y_new_S(i,0) = fm_prod(Xs_new_i,m_BetaS[i],this->num_threads())(0,0);
        }

        return y_new_C + y_new_E + y_new_S;
    }
};

#endif  /*FGWR_FMS_ESC_PREDICT_HPP*/