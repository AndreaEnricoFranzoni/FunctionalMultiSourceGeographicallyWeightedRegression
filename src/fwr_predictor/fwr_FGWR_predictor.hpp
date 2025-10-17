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


#ifndef FWR_FGWR_PREDICT_HPP
#define FWR_FGWR_PREDICT_HPP

#include "fwr_predictor.hpp"

#include <iostream>

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_FGWR_predictor final : public fwr_predictor<INPUT,OUTPUT>
{
private:

    //
    //Computing the betas in the new stance
    //
    /*!Coefficients of the basis expansion for event-dependent regressors: Lex1, every element of the vector is referring to a specific unit TO BE COMPUTED*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_bnc_pred;
    /*!Coefficients of the basis expansion for event-dependent regressors coefficients: every of the qe elements are n 1xLe_j matrices, one for each statistical unit*/
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> m_Bnc_pred;


    //Basis used for the regression coefficients
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
    /*!y train*/
    functional_matrix_sparse<INPUT,OUTPUT> m_y_train;
    /*!Functional event-dependent covariates (n_train x qnc)*/
    functional_matrix<INPUT,OUTPUT> m_Xnc_train;
    /*!Their transpost (qnc x n_train)*/
    functional_matrix<INPUT,OUTPUT> m_Xnc_train_t;
    /*!Scalar matrix with the penalization on the event-dependent covariates (sparse Le x Le, where Le is the sum of the basis of each E covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Rnc;



    //Betas
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
             typename SCALAR_SPARSE_MATRIX_OBJ>
    fwr_FGWR_predictor(FUNC_SPARSE_MATRIX_OBJ &&eta,
                       std::size_t qnc,
                       std::size_t Lnc,
                       const std::vector<std::size_t> &Lnc_j,
                       FUNC_MATRIX_OBJ &&y_train,
                       FUNC_MATRIX_OBJ &&Xnc_train,
                       SCALAR_SPARSE_MATRIX_OBJ &&Rnc,
                       INPUT a, 
                       INPUT b, 
                       int n_intervals_integration, 
                       double target_error, 
                       int max_iterations, 
                       std::size_t n_train, 
                       int number_threads)
            :   
                fwr_predictor<INPUT,OUTPUT>(a,b,n_intervals_integration,target_error,max_iterations,n_train,number_threads,false),
                m_eta{std::forward<FUNC_SPARSE_MATRIX_OBJ>(eta)},
                m_qnc(qnc),
                m_Lnc(Lnc),
                m_Lnc_j(Lnc_j),
                m_y_train{std::forward<FUNC_MATRIX_OBJ>(y_train)},
                m_Xnc_train{std::forward<FUNC_MATRIX_OBJ>(Xnc_train)},
                m_Rnc{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rnc)}
            {
                //input coherency
                assert((m_eta.rows() == m_qnc) && (m_eta.cols() == Lnc));
                assert((m_y_train.rows() == this->n_train()) && (m_y_train.cols() == 1));

                //compute the transpost
                m_eta_t = m_eta.transpose();
                m_Xnc_train_t = m_Xnc_train.transpose();
            }

    /*!
    * @brief Function to reconstruct the functional partial residuals
    */
    inline 
    void
    computePartialResiduals()
    override
    {}


    inline
    void
    computeBNew(const std::map<std::string,std::vector< functional_matrix_diagonal<INPUT,OUTPUT> >> &W)
    override
    {
        assert(W.size() == 1);
        for(std::size_t i = 0; i < W.at(fwr_FGWR_predictor<INPUT,OUTPUT>::id_NC).size(); ++i){
            assert((W.at(fwr_FGWR_predictor<INPUT,OUTPUT>::id_NC)[i].rows() == this->n_train()) && (W.at(fwr_FGWR_predictor<INPUT,OUTPUT>::id_NC)[i].cols() == this->n_train()));}
        //number of units to be predicted
        std::size_t n_pred = W.at(fwr_FGWR_predictor<INPUT,OUTPUT>::id_NC).size();


        //compute the non-stationary betas in the new locations
        //penalties in the new locations
        //(j_tilde + Rnc)^-1
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_Rnc_inv = this->operator_comp().compute_penalty(m_eta_t,m_Xnc_train_t,W.at(fwr_FGWR_predictor<INPUT,OUTPUT>::id_NC),m_Xnc_train,m_eta,m_Rnc);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
        //COMPUTING all the m_bnc in the new locations, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE NON-STATIONARY BETAS
        m_bnc_pred = this->operator_comp().compute_operator(m_eta_t,m_Xnc_train_t,W.at(fwr_FMGWR_predictor<INPUT,OUTPUT>::id_NC),m_y_train,j_Rnc_inv);

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
    {}

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
        assert(X_new.size() == 1);
        //controllo le unità statistiche
        std::size_t n_pred = X_new.at(fwr_FGWR_predictor<INPUT,OUTPUT>::id_NC).rows();
        assert((n_pred == m_BetaE.size()) && (n_pred == m_BetaS.size()));
        assert(X_new.at(fwr_FGWR_predictor<INPUT,OUTPUT>::id_NC).cols() == m_qnc);

        auto Xnc_new = X_new.at(fwr_FGWR_predictor<INPUT,OUTPUT>::id_NC);

        //y_new = X_new*beta = Xnc_new*beta_nc
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

        return y_new_NC;
    }

    /*!
    * @brief Virtual method to obtain a discrete version of the betas
    */
    inline 
    void 
    evalBetas(const std::vector<INPUT> &abscissa)
    override
    {
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
        return std::tuple{m_Bnc_pred};
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
        return std::tuple{m_BetaNC_ev};
    }

};

/*
        //INIZIO PARTE DA TOGLIERE
        m_bnc_pred.resize(n_pred);

        for(std::size_t i = 0; i < n_pred; ++i){
            m_bnc_pred[i] = Eigen::MatrixXd::Random(m_Lnc,1);
        }
        //FINE PARTE DA TOGLIERE
*/ 

#endif  /*FWR_FGWR_PREDICT_HPP*/