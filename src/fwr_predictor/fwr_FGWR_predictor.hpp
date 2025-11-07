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


#ifndef FWR_FGWR_PREDICT_HPP
#define FWR_FGWR_PREDICT_HPP

#include "fwr_predictor.hpp"


/*!
* @file fwr_FGWR_predictor.hpp
* @brief Contains the definition of the Functional Geographically Weighted Regression predictor
* @author Andrea Enrico Franzoni
*/



/*!
* @class fwr_FGWR_predictor
* @brief Concrete class for the Functional Geographically Weighted Regression predictor
* @tparam INPUT type of functional data abscissa
* @tparam OUTPUT type of functional data image
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_FGWR_predictor final : public fwr_predictor<INPUT,OUTPUT>
{
private:
    //
    //Computing the betas in the new stance
    //
    /*!Coefficients of the basis expansion for event-dependent regressors: Lncx1, every element of the vector is referring to a specific unit: TO BE COMPUTED*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_bnc_pred;
    /*!Coefficients of the basis expansion for non-stationary regressors coefficients: every of the qnc elements are n 1xLnc_j matrices, one for each statistical unit*/
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> m_Bnc_pred;


    //Basis used for the regression coefficients
    /*!Basis for non-stationary covariates regressors (sparse qnc x Lnc)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_eta;
    /*!Their transpost (sparse Lnc x qnc)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_eta_t;
    /*!Number of non-stationary covariates*/
    std::size_t m_qnc;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the non-stationary regressors coefficients*/
    std::size_t m_Lnc; 
    /*!Number of basis, for each non-stationary covariate, to perform the basis expansion of the regressors coefficients for the non-stationary regressors coefficients*/
    std::vector<std::size_t> m_Lnc_j;

    //Objects to reconstruct the functional partial residuals
    /*!y train (n_trainx1)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_y_train;
    /*!Functional non-stationary covariates (n_train x qnc)*/
    functional_matrix<INPUT,OUTPUT> m_Xnc_train;
    /*!Their transpost (qnc x n_train)*/
    functional_matrix<INPUT,OUTPUT> m_Xnc_train_t;
    /*!Scalar matrix with the penalization on the non-stationary covariates (sparse Lnc x Lnc, where Lnc is the sum of the basis of each non-stationary covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Rnc;



    //Betas
    //non-stationary: n_pred matrices of qnc x 1
    std::vector< functional_matrix<INPUT,OUTPUT>> m_BetaNC;
    /*!Discrete evaluation of all the beta_nc: a vector of dimension qnc, containing, for all the non-stationary covariates, the discrete ev of the respective beta, for each statistical unit*/
    std::vector< std::vector< std::vector< OUTPUT >>> m_BetaNC_ev;


public:
    /*!
    * @brief Constructor
    * @param eta functional sparse matrix containing the basis of the functional regression coefficients of the non-stationary covariates (qncxLnc, row i-th contains zeros and the basis of the i-th non-stationary covariate, their position shifted of sum_i_0_to_i(Lnc_i))
    * @param qnc number of non-stationary covariates
    * @param Lnc total number of basis used for the non-stationary covariates functional regression coefficients
    * @param Lnc_j vector containing in element i-th the number of basis for the non-stationary covariate i-th functional regression coefficients
    * @param y_train functional matrix containing the response of the training set (n_train x 1)
    * @param Xc_train functional matrix containing the stationary covariates of the training set (n_train x qc)
    * @param Xnc_train functional matrix containing the non-stationary covariates of the training set (n_train x qnc)
    * @param Rnc penalization matrix of the non-stationary covariates (diagonal block matrix containing the the scalar product within the second order derivatives of the functional regression coefficients basis. LncxLnc)
    * @param a left extreme functional data domain 
    * @param b right extreme functional data domain 
    * @param n_intervals_integration number of intervals used by the midpoint quadrature rule
    * @param n_train number of training statistical units
    * @param number_threads number of threads for OMP
    * @note input dimensions check and transpose computation
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
                       std::size_t n_train, 
                       int number_threads)
            :   
                fwr_predictor<INPUT,OUTPUT>(a,b,n_intervals_integration,n_train,number_threads,false),
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
    * @brief Constructor if beta already tuned
    */
    template<typename FUNC_SPARSE_MATRIX_OBJ,
             typename SCALAR_MATRIX_OBJ_VEC_VEC>
    fwr_FGWR_predictor(SCALAR_MATRIX_OBJ_VEC_VEC &&Bnc_tuned,
                       FUNC_SPARSE_MATRIX_OBJ &&eta,
                       std::size_t qnc,
                       std::size_t Lnc,
                       const std::vector<std::size_t> &Lnc_j,
                       std::size_t n_pred,
                       INPUT a, 
                       INPUT b, 
                       int n_intervals_integration, 
                       std::size_t n_train, 
                       int number_threads)
                :
                    fwr_predictor<INPUT,OUTPUT>(a,b,n_intervals_integration,n_train,number_threads,false),
                    m_Bnc_pred{std::forward<SCALAR_MATRIX_OBJ_VEC_VEC>(Bnc_tuned)},
                    m_eta{std::forward<FUNC_SPARSE_MATRIX_OBJ>(eta)},
                    m_qnc(qnc),
                    m_Lnc(Lnc),
                    m_Lnc_j(Lnc_j)
            {
                //input coherency
                assert(m_Bnc_pred.size() == m_qnc);
                for(std::size_t j = 0; j < m_qnc; ++j){  assert(m_Bnc_pred[j].size() == n_pred);}
                assert((m_eta.rows() == m_qnc) && (m_eta.cols() == Lnc));

                //bnc_pred
                m_bnc_pred = this->operator_comp().dewrap_operator(m_Bnc_pred,m_Lnc_j,n_pred);
            }

    /*!
    * @brief Function to compute the partial residuals accordingly to the fitted model
    * @note no partial residuals to be computed
    */
    inline 
    void
    computePartialResiduals()
    override
    {}

    /*!
    * @brief Updating the non-stationary betas on the units to be predicted
    * @param W map containing the non-stationary covariates of the units to be predicted. Each key represents, accordingly to the fitted model, a specific type of covariates
    * @note keys are the one stored in the base class. Input coherency is checked
    */ 
    inline
    void
    computeBNew(const std::map<std::string,std::vector< functional_matrix_diagonal<INPUT,OUTPUT> >> &W)
    override
    {
        assert(W.size() == 1);
        auto Wnc_new = W.at(std::string{fwr_FGWR_predictor<INPUT,OUTPUT>::id_NC});
        for(std::size_t i = 0; i < Wnc_new.size(); ++i){
            assert((Wnc_new[i].rows() == this->n_train()) && (Wnc_new[i].cols() == this->n_train()));}
        //number of units to be predicted
        std::size_t n_pred = Wnc_new.size();

        //compute the non-stationary betas in the new locations
        //penalties in the new locations
        //(j_tilde + Rnc)^-1 (LncxLnc)
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_Rnc_inv = this->operator_comp().compute_penalty(m_eta_t,m_Xnc_train_t,Wnc_new,m_Xnc_train,m_eta,m_Rnc);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
        //COMPUTING all the m_bnc in the new locations, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE NON-STATIONARY BETAS (n_pred elements Lncx1)
        m_bnc_pred = this->operator_comp().compute_operator(m_eta_t,m_Xnc_train_t,Wnc_new,m_y_train,j_Rnc_inv);

        //
        //wrapping the b from the shape useful for the computation into a more useful format for reporting the results
        //
        //non-stationary covariates (qnc elements of n_pred elements 1xLnc_i)
        m_Bnc_pred = this->operator_comp().wrap_operator(m_bnc_pred,m_Lnc_j,m_qnc,n_pred);

    }

    /*!
    * @brief Compute stationary betas as functional matrices
    * @note no stationary covariates for the FGWR
    */
    inline
    void 
    computeStationaryBetas()
    override
    {}

    /*!
    * @brief Compute non-stationary betas on the to-be-predicted units as functional matrices
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
    * @param X_new map containig, as elements, non-stationary covariates of the units to be predicted. Each key represents, accordingly to the fitted model, a specific type of covariates
    * @return a functional matrix containing the prediction, n_predx1
    * @note keys are the one stored in the base class. Input coherency is checked
    */
    inline
    functional_matrix<INPUT,OUTPUT>
    predict(const std::map<std::string,functional_matrix<INPUT,OUTPUT>> &X_new)
    const
    override
    {
        //input coherence
        assert(X_new.size() == 1);

        auto Xnc_new = X_new.at(std::string{fwr_FGWR_predictor<INPUT,OUTPUT>::id_NC});

        std::size_t n_pred = Xnc_new.rows();
        assert(n_pred == m_BetaNC.size());
        assert(Xnc_new.cols() == m_qnc);


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
    * @brief Evaluating the functional betas along a grid
    * @param abscissa the grid over which evaluating the functional betas
    */
    inline 
    void 
    evalBetas(const std::vector<INPUT> &abscissa)
    override
    {
        m_BetaNC_ev = this->operator_comp().eval_func_betas(m_BetaNC,m_qnc,abscissa);
    }

    /*!
    * @brief Function to return the coefficients of the betas basis expansion
    * @return a tuple containing m_Bnc_pred
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
    * @brief Function to return the the betas evaluated, tuple of different dimension depending on the model fitted
    * @return a tuple containing m_BetaNC_ev
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

#endif  /*FWR_FGWR_PREDICT_HPP*/