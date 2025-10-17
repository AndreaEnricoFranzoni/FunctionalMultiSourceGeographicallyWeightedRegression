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


#ifndef FWR_FMGWR_PREDICT_HPP
#define FWR_FMGWR_PREDICT_HPP

#include "fwr_predictor.hpp"

#include <iostream>

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_FMGWR_predictor final : public fwr_predictor<INPUT,OUTPUT>
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
    //
    //Computing the betas in the new stance
    //
    /*!Coefficients of the basis expansion for event-dependent regressors: Lex1, every element of the vector is referring to a specific unit TO BE COMPUTED*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_bnc_pred;
    /*!Coefficients of the basis expansion for event-dependent regressors coefficients: every of the qe elements are n 1xLe_j matrices, one for each statistical unit*/
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> m_Bnc_pred;


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
    functional_matrix_sparse<INPUT,OUTPUT> m_eta;
    /*!Their transpost (sparse Le x qE)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_eta_t;
    /*!Number of event-dependent covariates*/
    std::size_t m_qnc;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the event-dependent regressors coefficients*/
    std::size_t m_Lnc; 
    /*!Number of basis, for each event-dependent covariate, to perform the basis expansion of the regressors coefficients for the event-dependent regressors coefficients*/
    std::vector<std::size_t> m_Lnc_j;

    //Objects to reconstruct the functional partial residuals
    /*!Basis used for response (n_trainx(n_trainxxLy)) in the training process*/
    functional_matrix_sparse<INPUT,OUTPUT> m_phi;
    /*!Number of basis used to make basis expansion for y in the training process*/
    std::size_t m_Ly;
    /*!c_tilde_hat ((n_trainxLy)x1)*/
    FDAGWR_TRAITS::Dense_Matrix m_c_tilde_hat;
    /*!Functional event-dependent covariates (n_train x qnc)*/
    functional_matrix<INPUT,OUTPUT> m_y_train;
    /*!Functional event-dependent covariates (n_train x qnc)*/
    functional_matrix<INPUT,OUTPUT> m_Xc_train;
    /*!Functional event-dependent covariates (n_train x qnc)*/
    functional_matrix<INPUT,OUTPUT> m_Xnc_train;
    /*!Their transpost (qnc x n_train)*/
    functional_matrix<INPUT,OUTPUT> m_Xnc_train_t;
    /*!Scalar matrix with the penalization on the event-dependent covariates (sparse Le x Le, where Le is the sum of the basis of each E covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Rnc;

    //Part to be computed
    functional_matrix<INPUT,OUTPUT> m_y_tilde_new;


    //Betas
    //stationary: qc x 1
    functional_matrix<INPUT,OUTPUT> m_BetaC;                        //elemento funzionale
    /*!Discrete evaluation of all the beta_c: a vector of dimension qc, containing, for all the stationary covariates, the discrete ev of the respective beta*/
    std::vector< std::vector< OUTPUT >> m_BetaC_ev;
    //non-stationary: n_pred matrices of qnc x 1
    std::vector< functional_matrix<INPUT,OUTPUT>> m_BetaNC;
    /*!Discrete evaluation of all the beta_nc: a vector of dimension qnc, containing, for all the non-stationary covariates, the discrete ev of the respective beta, for each statistical unit*/
    std::vector< std::vector< std::vector< OUTPUT >>> m_BetaNC_ev;


public:
    /*!
    * @brief Constructor
    */
    template<typename FUNC_MATRIX_OBJ, 
             typename FUNC_SPARSE_MATRIX_OBJ,
             typename SCALAR_MATRIX_OBJ, 
             typename SCALAR_MATRIX_OBJ_VEC,
             typename SCALAR_SPARSE_MATRIX_OBJ>
    fwr_FMGWR_predictor(SCALAR_MATRIX_OBJ_VEC &&Bc_fitted,
                        FUNC_SPARSE_MATRIX_OBJ &&omega,
                        std::size_t qc,
                        std::size_t Lc,
                        const std::vector<std::size_t> &Lc_j,
                        FUNC_SPARSE_MATRIX_OBJ &&eta,
                        std::size_t qnc,
                        std::size_t Lnc,
                        const std::vector<std::size_t> &Lnc_j,
                        FUNC_SPARSE_MATRIX_OBJ &&phi,
                        std::size_t Ly,
                        SCALAR_MATRIX_OBJ &&c_tilde_hat,
                        FUNC_MATRIX_OBJ &&y_train,
                        FUNC_MATRIX_OBJ &&Xc_train,
                        FUNC_MATRIX_OBJ &&Xnc_train,
                        SCALAR_SPARSE_MATRIX_OBJ &&Rnc,
                        INPUT a, 
                        INPUT b, 
                        int n_intervals_integration, 
                        double target_error, 
                        int max_iterations, 
                        std::size_t n_train, 
                        int number_threads,
                        bool bf_estimation)
            :   
                fwr_predictor<INPUT,OUTPUT>(a,b,n_intervals_integration,target_error,max_iterations,n_train,number_threads,bf_estimation),
                m_Bc_fitted{std::forward<SCALAR_MATRIX_OBJ_VEC>(Bc_fitted)},
                m_omega{std::forward<FUNC_SPARSE_MATRIX_OBJ>(omega)},
                m_qc(qc),
                m_Lc(Lc),
                m_Lc_j(Lc_j),
                m_eta{std::forward<FUNC_SPARSE_MATRIX_OBJ>(eta)},
                m_qnc(qnc),
                m_Lnc(Lnc),
                m_Lnc_j(Lnc_j),
                m_phi{std::forward<FUNC_SPARSE_MATRIX_OBJ>(phi)},
                m_Ly(Ly),
                m_c_tilde_hat{std::forward<SCALAR_MATRIX_OBJ>(c_tilde_hat)},
                m_y_train{std::forward<FUNC_MATRIX_OBJ>(y_train)},
                m_Xc_train{std::forward<FUNC_MATRIX_OBJ>(Xc_train)},
                m_Xnc_train{std::forward<FUNC_MATRIX_OBJ>(Xnc_train)},
                m_Rnc{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rnc)}
            {
                //input coherency
                assert(m_Bc_fitted.size() == m_qc);
                assert((m_omega.rows() == m_qc) && (m_omega.cols() == Lc));
                assert((m_eta.rows() == m_qnc) && (m_eta.cols() == Lnc));
                assert((m_phi.rows() == this->n_train()) && (m_phi.cols() == (this->n_train()*m_Ly)));
                assert((m_c_tilde_hat.rows() == (this->n_train()*m_Ly)) && (m_c_tilde_hat.cols() == 1));

                //compute the transpost
                m_omega_t = m_omega.transpose();
                m_eta_t = m_eta.transpose();
                m_Xnc_train_t = m_Xnc_train.transpose();

                //dewrappare i b_train (li incolonna)
                m_bc_fitted = this->operator_comp().dewrap_operator(m_Bc_fitted,m_Lc_j);
            }

    /*!
    * @brief Function to reconstruct the functional partial residuals
    */
    inline 
    void
    computePartialResiduals()
    override
    {
        if (!this->bf_estimation())
        {
            //y_tilde_new(t) n_trainx1
            m_y_tilde_new = fm_prod(m_phi,m_c_tilde_hat);
        }
        else
        {
            m_y_tilde_new = m_y_train - fm_prod(fm_prod(m_Xc_train,m_omega),m_bc_fitted,this->number_threads());
        }
        
    }


    inline
    void
    computeBNew(const std::map<std::string,std::vector< functional_matrix_diagonal<INPUT,OUTPUT> >> &W)
    override
    {
        assert(W.size() == 1);
        for(std::size_t i = 0; i < W.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_NC).size(); ++i){
            assert((W.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_NC)[i].rows() == this->n_train()) && (W.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_NC)[i].cols() == this->n_train()));}
        //number of units to be predicted
        std::size_t n_pred = W.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_NC).size();


        //compute the non-stationary betas in the new locations
        //penalties in the new locations
        //(j_tilde + Rnc)^-1
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_tilde_Rnc_inv = this->operator_comp().compute_penalty(m_eta_t,m_Xnc_train_t,W.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_NC),m_Xnc_train,m_eta,m_Rnc);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
        //COMPUTING all the m_bnc in the new locations, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE NON-STATIONARY BETAS
        m_bnc_pred = this->operator_comp().compute_operator(m_eta_t,m_Xnc_train_t,W.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_NC),m_y_tilde_new,j_tilde_Rnc_inv);


/*
        //INIZIO PARTE DA TOGLIERE
        m_bnc_pred.resize(n_pred);

        for(std::size_t i = 0; i < n_pred; ++i){
            m_bnc_pred[i] = Eigen::MatrixXd::Random(m_Lnc,1);
        }
        //FINE PARTE DA TOGLIERE
*/ 





     


        //
        //wrapping the b from the shape useful for the computation into a more useful format for reporting the results: TENERE
        //
        //non-stationary covariates
        m_Bnc_pred = this->operator_comp().wrap_operator(m_bnc_pred,m_Lnc_j,m_qnc,n_pred);

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
        std::size_t n_pred = m_bnc_pred.size();
        m_BetaNC.resize(n_pred);

#ifdef _OPENMP
#pragma omp parallel for shared(m_BetaNC,m_eta,m_bnc_pred,n_pred) num_threads(this->number_threads())
#endif
        for(std::size_t i = 0; i < n_pred; ++i)
        {
            m_BetaNC[i] = fm_prod(m_eta,m_bnc_pred[i]);
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
        assert(X_new.size() == 2);
        //controllo le unità statistiche
        assert(X_new.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_C).rows() == X_new.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_NC).rows());
        std::size_t n_pred = X_new.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_C).rows();
        assert((n_pred == m_BetaE.size()) && (n_pred == m_BetaS.size()));
        assert((X_new.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_C).cols() == m_qc) && (X_new.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_NC).cols() == m_qnc));

        auto Xc_new  = X_new.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_C);
        auto Xnc_new = X_new.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_NC);

        //y_new = X_new*beta = Xc_new*beta_c + Xnc_new*beta_nc
        functional_matrix<INPUT,OUTPUT> y_new_C = fm_prod(Xc_new,m_BetaC,this->number_threads());    //n_pred x 1
        functional_matrix<INPUT,OUTPUT> y_new_NC(n_pred,1);

        
#ifdef _OPENMP
#pragma omp parallel for shared(Xnc_new,m_BetaNC,y_new_NC,n_pred,m_qnc) num_threads(this->number_threads())
#endif
        for(std::size_t i = 0; i < n_pred; ++i)
        {
            std::vector< FUNC_OBJ<INPUT,OUTPUT> > xnc_new_i(Xnc_new.row(i).cbegin(),Xnc_new.row(i).cend()); //1xqnc
            functional_matrix<INPUT,OUTPUT> Xnc_new_i(xnc_new_i,1,m_qnc);
            y_new_NC(i,0) = fm_prod(Xnc_new_i,m_BetaNC[i],this->number_threads())(0,0);
        }

        return y_new_C + y_new_NC;
    }

    /*!
    * @brief Virtual method to obtain a discrete version of the betas
    */
    inline 
    void 
    evalBetas(const std::vector<INPUT> &abscissa)
    override
    {
        m_BetaC_ev  = this->operator_comp().eval_func_betas(m_BetaC,m_qc,abscissa);
        m_BetaNC_ev = this->operator_comp().eval_func_betas(m_BetaNC,m_qnc,abscissa);
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
        return std::tuple{m_Bc_fitted,m_Bnc_pred};
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
        return std::tuple{m_BetaC_ev,m_BetaNC_ev};
    }


};

#endif  /*FWR_FMGWR_PREDICT_HPP*/