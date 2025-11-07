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


#ifndef FWR_FMSGWR_ESC_PREDICT_HPP
#define FWR_FMSGWR_ESC_PREDICT_HPP

#include "fwr_predictor.hpp"


/*!
* @file fwr_FMSGWR_ESC_predictor.hpp
* @brief Contains the definition of the Functional Multi-Source Geographically Weighted Regression ESC predictor
* @author Andrea Enrico Franzoni
*/



/*!
* @class fwr_FMSGWR_ESC_predictor
* @brief Concrete class for the Functional Multi-Source Geographically Weighted Regression ESC predictor
* @tparam INPUT type of functional data abscissa
* @tparam OUTPUT type of functional data image
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_FMSGWR_ESC_predictor final : public fwr_predictor<INPUT,OUTPUT>
{
private:
    //
    //FITTED MODEL
    //
    //Basis expansion coefficients for the regression coefficients: fitted values
    /*!Coefficients of the basis expansion for stationary regressors coefficients: Lcx1*/
    FDAGWR_TRAITS::Dense_Matrix m_bc_fitted;
    /*!Coefficients of the basis expansion for stationary regressors coefficients: n_train elements Lc_jx1*/
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
    /*!y train (n_trainx1)*/
    functional_matrix<INPUT,OUTPUT> m_y_train;
    /*!Xc train (n_trainxqc)*/
    functional_matrix<INPUT,OUTPUT> m_Xc_train;
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

    //Parts to be computed to computed the new non-stationary betas
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
    /*!Discrete evaluation of all the beta_c: a vector of dimension qc, containing, for all the stationary covariates, the discrete ev of the respective beta*/
    std::vector< std::vector< OUTPUT >> m_BetaC_ev;
    //events-dependent: n_pred matrices of qe x 1
    std::vector< functional_matrix<INPUT,OUTPUT>> m_BetaE;
    /*!Discrete evaluation of all the beta_e: a vector of dimension qe, containing, for all the event-dependent covariates, the discrete ev of the respective beta, for each statistical unit*/
    std::vector< std::vector< std::vector< OUTPUT >>> m_BetaE_ev;
    //stations-dependent: n_pred matrices of qs x 1
    std::vector< functional_matrix<INPUT,OUTPUT>> m_BetaS;
    /*!Discrete evaluation of all the beta_s: a vector of dimension qs, containing, for all the station-dependent covariates, the discrete ev of the respective beta, for each statistical unit*/
    std::vector< std::vector< std::vector< OUTPUT >>> m_BetaS_ev;

public:
    /*!
    * @brief Constructor
    * @param Bc_fitted coefficients of the basis expansion for stationary regressors coefficients: n_train elements Lc_jx1
    * @param Bs_fitted Coefficients of the basis expansion for station-dependent regressors coefficients: every of the qe elements are n 1xLs_j matrices, one for each statistical unit
    * @param omega functional sparse matrix containing the basis of the functional regression coefficients of the stationary covariates (qcxLc, row i-th contains zeros and the basis of the i-th stationary covariate, their position shifted of sum_i_0_to_i(Lc_i))
    * @param qc number of stationary covariates
    * @param Lc total number of basis used for the stationary covariates functional regression coefficients
    * @param Lc_j vector containing in element i-th the number of basis for the stationary covariate i-th functional regression coefficients
    * @param theta functional sparse matrix containing the basis of the functional regression coefficients of the event-dependent covariates (qexLe, row i-th contains zeros and the basis of the i-th event-dependent covariate, their position shifted of sum_i_0_to_i(Le_i))
    * @param qe number of event-dependent covariates
    * @param Le total number of basis used for the event-dependent covariates functional regression coefficients
    * @param Le_j vector containing in element i-th the number of basis for the event-dependent covariate i-th functional regression coefficients
    * @param psi functional sparse matrix containing the basis of the functional regression coefficients of the station-dependent covariates (qsxLs, row i-th contains zeros and the basis of the i-th station-dependent covariate, their position shifted of sum_i_0_to_i(Ls_i))
    * @param qs number of station-dependent covariates
    * @param Ls total number of basis used for the station-dependent covariates functional regression coefficients
    * @param Ls_j vector containing in element i-th the number of basis for the station-dependent covariate i-th functional regression coefficients
    * @param phi functional sparse matrix containing the basis for the response (n_trainx(n_train*Ly)). Each row contains the basis, every time shifted by (row_i-1)+Ly, always the same basis
    * @param Ly number of basis for the response
    * @param c_tilde_hat basis expansion coefficients ((n_train*Ly)x1)
    * @param A_e vector of n_train elements, containing the Ae_i (Lex(n_train*Ly)) computed while fitting the model
    * @param B_e_for_K_e_s vector of n_train elements, containing B_E_i_for_K_e_s the ((n_train elements LexLs)) computed while fitting the model
    * @param y_train functional matrix containing the response of the training set (n_train x 1)
    * @param Xc_train functional matrix containing the stationary covariates of the training set (n_train x qc)
    * @param Xe_train functional matrix containing the event-dependent covariates of the training set (n_train x qe)
    * @param Re penalization matrix of the event-dependent covariates (diagonal block matrix containing the the scalar product within the second order derivatives of the functional regression coefficients basis. LexLe)
    * @param Xs_train functional matrix containing the station-dependent covariates of the training set (n_train x qs)
    * @param Rs penalization matrix of the station-dependent covariates (diagonal block matrix containing the the scalar product within the second order derivatives of the functional regression coefficients basis. LsxLs)
    * @param a left extreme functional data domain 
    * @param b right extreme functional data domain 
    * @param n_intervals_integration number of intervals used by the midpoint quadrature rule
    * @param n_train number of training statistical units
    * @param number_threads number of threads for OMP
    * @param in_cascade_estimation if true, for more than one source covariates, the estimation is made in cascade. If false, exact
    * @note input dimensions check and transpose computation
    */
    template<typename FUNC_MATRIX_OBJ, 
             typename FUNC_SPARSE_MATRIX_OBJ,
             typename SCALAR_MATRIX_OBJ, 
             typename SCALAR_MATRIX_OBJ_VEC,
             typename SCALAR_MATRIX_OBJ_VEC_VEC,
             typename SCALAR_SPARSE_MATRIX_OBJ>
    fwr_FMSGWR_ESC_predictor(SCALAR_MATRIX_OBJ_VEC &&Bc_fitted,
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
                           FUNC_MATRIX_OBJ &&y_train,
                           FUNC_MATRIX_OBJ &&Xc_train,
                           FUNC_MATRIX_OBJ &&Xe_train,
                           SCALAR_SPARSE_MATRIX_OBJ &&Re,
                           FUNC_MATRIX_OBJ &&Xs_train,
                           SCALAR_SPARSE_MATRIX_OBJ &&Rs,
                           INPUT a, 
                           INPUT b, 
                           int n_intervals_integration, 
                           std::size_t n_train, 
                           int number_threads,
                           bool in_cascade_estimation)
            :   
                    fwr_predictor<INPUT,OUTPUT>(a,b,n_intervals_integration,n_train,number_threads,in_cascade_estimation),
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
                    m_y_train{std::forward<FUNC_MATRIX_OBJ>(y_train)},
                    m_Xc_train{std::forward<FUNC_MATRIX_OBJ>(Xc_train)},
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
                m_bc_fitted = this->operator_comp().dewrap_operator(m_Bc_fitted,m_Lc_j);
                //bs_fitted
                m_bs_fitted = this->operator_comp().dewrap_operator(m_Bs_fitted,m_Ls_j,this->n_train());
            }

    /*!
    * @brief Constructor if beta already tuned
    */
    template<typename FUNC_SPARSE_MATRIX_OBJ,
             typename SCALAR_MATRIX_OBJ_VEC,
             typename SCALAR_MATRIX_OBJ_VEC_VEC>
    fwr_FMSGWR_ESC_predictor(SCALAR_MATRIX_OBJ_VEC &&Bc_fitted,
                             SCALAR_MATRIX_OBJ_VEC_VEC &&Be_tuned,
                             SCALAR_MATRIX_OBJ_VEC_VEC &&Bs_tuned,
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
                             std::size_t n_pred,
                             INPUT a, 
                             INPUT b, 
                             int n_intervals_integration, 
                             std::size_t n_train, 
                             int number_threads,
                             bool in_cascade_estimation)
                :
                    fwr_predictor<INPUT,OUTPUT>(a,b,n_intervals_integration,n_train,number_threads,in_cascade_estimation),
                    m_Bc_fitted{std::forward<SCALAR_MATRIX_OBJ_VEC>(Bc_fitted)},
                    m_Be_pred{std::forward<SCALAR_MATRIX_OBJ_VEC_VEC>(Be_tuned)},
                    m_Bs_pred{std::forward<SCALAR_MATRIX_OBJ_VEC_VEC>(Bs_tuned)},
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
                    m_Ls_j(Ls_j)

            {
                std::cout << "New constructor" << std::endl;
                //input coherency
                assert(m_Bc_fitted.size() == m_qc);
                assert(m_Be_pred.size() == m_qs);
                assert(m_Bs_pred.size() == m_qs);
                for(std::size_t j = 0; j < m_qe; ++j){  assert(m_Be_pred[j].size() == n_pred);}
                for(std::size_t j = 0; j < m_qs; ++j){  assert(m_Bs_pred[j].size() == n_pred);}
                assert((m_omega.rows() == m_qc) && (m_omega.cols() == Lc));
                assert((m_theta.rows() == m_qe) && (m_theta.cols() == Le));
                assert((m_psi.rows() == m_qs) && (m_psi.cols() == Ls));

                //dewrappare i b_train (li incolonna)
                m_bc_fitted = this->operator_comp().dewrap_operator(m_Bc_fitted,m_Lc_j);
                //be_pred
                m_be_pred = this->operator_comp().dewrap_operator(m_Be_pred,m_Ls_j,n_pred);
                //bs_fitted
                m_bs_pred = this->operator_comp().dewrap_operator(m_Bs_pred,m_Ls_j,n_pred);


                std::cout << "Bc: " << m_bc_fitted.rows() << "x" << m_bc_fitted.cols() << std::endl;
                std::cout << m_bc_fitted << std::endl;

                std::cout << "Be:" << std::endl;
                for(std::size_t i = 0; i < m_be_pred.size(); ++i){
                    std::cout << "unit " << i+1 << ": " << m_be_pred[i].rows() << "x" << m_be_pred[i].cols() << std::endl;
                    std::cout << m_be_pred[i] << std::endl;
                }

                std::cout << "Bs:" << std::endl;
                for(std::size_t i = 0; i < m_bs_pred.size(); ++i){
                    std::cout << "unit " << i+1 << ": " << m_bs_pred[i].rows() << "x" << m_bs_pred[i].cols() << std::endl;
                    std::cout << m_bs_pred[i] << std::endl;
                }
            }

    /*!
    * @brief Function to compute the partial residuals accordingly to the fitted model
    */
    inline 
    void
    computePartialResiduals()
    override
    {
        //exact estimation
        if(!this->in_cascade_estimation())
        {
            //retrieve the partial residuals from the fitted model
            //K_e_s(t) (n_train x Ls)
            m_K_e_s = this->operator_comp().compute_functional_operator(m_Xe_train,m_theta,m_B_e_for_K_e_s);
            //X_s_crossed(t) (n_train x Ls)
            m_X_s_train_crossed = fm_prod(m_Xs_train,m_psi) - m_K_e_s;
            m_X_s_train_crossed_t = m_X_s_train_crossed.transpose();
            //y_tilde_hat(t) (n_trainx1)
            m_y_tilde_hat = fm_prod(m_phi,m_c_tilde_hat);
            //He(t) (n_trainx(n_train*Ly))
            m_H_e = this->operator_comp().compute_functional_operator(m_Xe_train,m_theta,m_A_e);
            //y_tilde_new(t) (n_trainx1)
            m_y_tilde_new = fm_prod(functional_matrix<INPUT,OUTPUT>(m_phi - m_H_e),m_c_tilde_hat,this->number_threads());
            //y_tilde_tilde_hat(t) (n_trainx1)
            m_y_tilde_tilde_hat = m_y_tilde_hat - this->operator_comp().compute_functional_operator(m_Xs_train,m_psi,m_bs_fitted);
        }
        //in cascade estimation
        else
        {
            //y_tilde_hat(t) (n_trainx1)
            m_y_tilde_hat = m_y_train - fm_prod(fm_prod(m_Xc_train,m_omega),m_bc_fitted,this->number_threads());
            //y_tilde_tilde_hat(t) (n_trainx1)
            m_y_tilde_tilde_hat = m_y_train;

#ifdef _OPENMP
#pragma omp parallel for shared(m_y_tilde_tilde_hat,m_Xs_train,m_psi,m_y_tilde_hat) num_threads(this->number_threads())
#endif
            for(std::size_t i = 0; i < this->n_train(); ++i)
            {
                std::vector< FUNC_OBJ<INPUT,OUTPUT> > xs_i(m_Xs_train.row(i).cbegin(),m_Xs_train.row(i).cend()); //1xqs
                functional_matrix<INPUT,OUTPUT> Xs_i(xs_i,1,m_qs);
                functional_matrix<INPUT,OUTPUT> y_tilde_hat_i(1,1,m_y_tilde_hat(i,0));
                m_y_tilde_tilde_hat(i,0) = (y_tilde_hat_i - fm_prod(fm_prod(Xs_i,m_psi),m_bs_fitted[i],this->number_threads()))(0,0);
            }
        }
    }


    /*!
    * @brief Updating the non-stationary betas on the units to be predicted
    * @param W map containing the event-dependent and station-dependent covariates of the units to be predicted. Each key represents, accordingly to the fitted model, a specific type of covariates
    * @note keys are the one stored in the base class. Input coherency is checked
    */ 
    inline
    void
    computeBNew(const std::map<std::string,std::vector< functional_matrix_diagonal<INPUT,OUTPUT> >> &W)
    override
    {
        assert(W.size() == 2);
        auto We_new = W.at(std::string{fwr_FMSGWR_ESC_predictor<INPUT,OUTPUT>::id_E});
        auto Ws_new = W.at(std::string{fwr_FMSGWR_ESC_predictor<INPUT,OUTPUT>::id_S});
        //input coherency
        assert(We_new.size() == Ws_new.size());
        for(std::size_t i = 0; i < We_new.size(); ++i){
            assert((We_new[i].rows() == this->n_train()) && (We_new[i].cols() == this->n_train()));
            assert((Ws_new[i].rows() == this->n_train()) && (Ws_new[i].cols() == this->n_train()));}
        //number of units to be predicted
        std::size_t n_pred = We_new.size();

        //exact estimation
        if(!this->in_cascade_estimation())
        {
            //compute the non-stationary betas in the new locations
            //penalties in the new locations
            //(j_tilde_tilde + Re)^-1
            std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_double_tilde_Re_inv = this->operator_comp().compute_penalty(m_theta_t,m_Xe_train_t,We_new,m_Xe_train,m_theta,m_Re);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
            //(j_tilde + Rs)^-1
            std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_tilde_Rs_inv        = this->operator_comp().compute_penalty(m_X_s_train_crossed_t,Ws_new,m_X_s_train_crossed,m_Rs);
            //COMPUTING all the m_bs in the new locations, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATION-DEPENDENT BETAS (n_pred elements Lsx1)
            m_bs_pred = this->operator_comp().compute_operator(m_X_s_train_crossed_t,Ws_new,m_y_tilde_new,j_tilde_Rs_inv);
            //COMPUTING all the m_be in the new locations, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATION-DEPENDENT BETAS (n_pred elements Lex1)
            m_be_pred = this->operator_comp().compute_operator(m_theta_t,m_Xe_train_t,We_new,m_y_tilde_tilde_hat,j_double_tilde_Re_inv);
        }
        //in cascade estimation
        else
        {
            //bs
            std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_i_Rs_inv = this->operator_comp().compute_penalty(m_psi_t,m_Xs_train_t,Ws_new,m_Xs_train,m_psi,m_Rs);      //LsxLs
            m_bs_pred = this->operator_comp().compute_operator(m_psi_t,m_Xs_train_t,Ws_new,m_y_tilde_hat,j_i_Rs_inv);           //(n_pred elements Lsx1)
            //be
            std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_i_Re_inv = this->operator_comp().compute_penalty(m_theta_t,m_Xe_train_t,We_new,m_Xe_train,m_theta,m_Re);  //LexLe
            m_be_pred = this->operator_comp().compute_operator(m_theta_t,m_Xe_train_t,We_new,m_y_tilde_tilde_hat,j_i_Re_inv);   //(n_pred elements Lex1)
        }

        //
        //wrapping the b from the shape useful for the computation into a more useful format for reporting the results
        //
        //event-dependent covariates (qe elements of n_pred elements 1xLe_i)
        m_Be_pred = this->operator_comp().wrap_operator(m_be_pred,m_Le_j,m_qe,n_pred);
        //station-dependent covariates (qs elements of n_pred elements 1xLs_i)
        m_Bs_pred = this->operator_comp().wrap_operator(m_bs_pred,m_Ls_j,m_qs,n_pred);

    }

    /*!
    * @brief Compute stationary betas as functional matrices
    */
    inline
    void 
    computeStationaryBetas()
    override
    {
        m_BetaC = fm_prod(m_omega,m_bc_fitted); //qc x 1

        double loc = 0.3;
        for(std::size_t i = 0; i < m_BetaC.rows(); ++i){
            for(std::size_t j = 0; j < m_BetaC.cols(); ++j){
                std::cout << "betac(" << i << "," << j << "): " << m_BetaC(i,j)(loc) << std::endl;
            }
        }
    }

    /*!
    * @brief Compute non-stationary betas on the to-be-predicted units as functional matrices
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
            m_BetaE[i] = fm_prod(m_theta,m_be_pred[i]);     //n_pred elems qe x 1
            m_BetaS[i] = fm_prod(m_psi,m_bs_pred[i]);       //n_pred elems qs x 1
        }
    }

    /*!
    * @brief Compute prediction
    * @param X_new map containig, as elements, stationary, event-dependent and station-dependent covariates of the units to be predicted. Each key represents, accordingly to the fitted model, a specific type of covariates
    * @return a functional matrix containing the prediction, n_predx1
    * @note keys are the one stored in the base class. Input coherency is checked
    */
    inline
    functional_matrix<INPUT,OUTPUT>
    predict(const std::map<std::string,functional_matrix<INPUT,OUTPUT>> &X_new)
    const
    override
    {
        //input coherency
        assert(X_new.size() == 3);

        auto Xc_new = X_new.at(std::string{fwr_FMSGWR_ESC_predictor<INPUT,OUTPUT>::id_C});
        auto Xe_new = X_new.at(std::string{fwr_FMSGWR_ESC_predictor<INPUT,OUTPUT>::id_E});
        auto Xs_new = X_new.at(std::string{fwr_FMSGWR_ESC_predictor<INPUT,OUTPUT>::id_S});

        assert(Xc_new.rows() == Xe_new.rows());
        assert(Xe_new.rows() == Xs_new.rows());
        std::size_t n_pred = Xc_new.rows();
        assert((n_pred == m_BetaE.size()) && (n_pred == m_BetaS.size()));
        assert((Xc_new.cols() == m_qc) && (Xe_new.cols() == m_qe) && (Xs_new.cols() == m_qs));



        //y_new = X_new*beta = Xc_new*beta_c + Xe_new*beta_e + Xs_new*beta_s
        functional_matrix<INPUT,OUTPUT> y_new_C = fm_prod(Xc_new,m_BetaC,this->number_threads());    //n_pred x 1
        functional_matrix<INPUT,OUTPUT> y_new_E(n_pred,1);
        functional_matrix<INPUT,OUTPUT> y_new_S(n_pred,1);
        
#ifdef _OPENMP
#pragma omp parallel for shared(Xe_new,m_BetaE,y_new_E,Xs_new,m_BetaS,y_new_S,n_pred,m_qe,m_qs) num_threads(this->number_threads())
#endif
        for(std::size_t i = 0; i < n_pred; ++i)
        {
            std::vector< FUNC_OBJ<INPUT,OUTPUT> > xe_new_i(Xe_new.row(i).cbegin(),Xe_new.row(i).cend()); //1xqe
            functional_matrix<INPUT,OUTPUT> Xe_new_i(xe_new_i,1,m_qe);
            y_new_E(i,0) = fm_prod(Xe_new_i,m_BetaE[i],this->number_threads())(0,0);

            std::vector< FUNC_OBJ<INPUT,OUTPUT> > xs_new_i(Xs_new.row(i).cbegin(),Xs_new.row(i).cend()); //1xqs
            functional_matrix<INPUT,OUTPUT> Xs_new_i(xs_new_i,1,m_qs);
            y_new_S(i,0) = fm_prod(Xs_new_i,m_BetaS[i],this->number_threads())(0,0);
        }

        return y_new_C + y_new_E + y_new_S;
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
        m_BetaC_ev = this->operator_comp().eval_func_betas(m_BetaC,m_qc,abscissa);
        m_BetaE_ev = this->operator_comp().eval_func_betas(m_BetaE,m_qe,abscissa);
        m_BetaS_ev = this->operator_comp().eval_func_betas(m_BetaS,m_qs,abscissa);
    }

    /*!
    * @brief Function to return the coefficients of the betas basis expansion
    * @return a tuple containing m_Bc_fitted,m_Be_pred and m_Bs_pred
    */
    inline 
    BTuple 
    bCoefficients()
    const 
    override
    {
        return std::tuple{m_Bc_fitted,m_Be_pred,m_Bs_pred};
    }

    /*!
    * @brief Function to return the the betas evaluated, tuple of different dimension depending on the model fitted
    * @return a tuple containing m_BetaC_ev,m_BetaE_ev and m_BetaS_ev
    */
    inline 
    BetasTuple 
    betas() 
    const
    override
    {
        return std::tuple{m_BetaC_ev,m_BetaE_ev,m_BetaS_ev};
    }

};

#endif  /*FWR_FMSGWR_ESC_PREDICT_HPP*/