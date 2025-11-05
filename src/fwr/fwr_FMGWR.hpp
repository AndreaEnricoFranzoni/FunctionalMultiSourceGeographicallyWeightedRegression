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


#ifndef FWR_FMGWR_ALGO_HPP
#define FWR_FMGWR_ALGO_HPP

#include "fwr.hpp"


/*!
* @file fwr_FMGWR.hpp
* @brief Contains the definition of the Functional Mixed Geographically Weighted Regression model
* @author Andrea Enrico Franzoni
*/



/*!
* @class fwr_FMGWR
* @brief Concrete class for the Functional Mixed Geographically Weighted Regression model, estimating, in order, stationary and non-stationary functional regression coefficients
* @tparam INPUT type of functional data abscissa
* @tparam OUTPUT type of functional data image
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_FMGWR final : public fwr<INPUT,OUTPUT>
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
    /*!Coefficients of the basis expansion for stationary regressors coefficients: every element is Lc_jx1*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_Bc;
    /*!Discrete evaluation of all the beta_c: a vector of dimension qc, containing, for all the stationary covariates, the discrete ev of the respective beta*/
    std::vector< std::vector< OUTPUT >> m_beta_c;
    /*!Number of stationary covariates*/
    std::size_t m_qc;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the stationary regressors coefficients*/
    std::size_t m_Lc; 
    /*!Number of basis, for each stationary covariate, to perform the basis expansion of the regressors coefficients for the stationary regressors coefficients*/
    std::vector<std::size_t> m_Lc_j;

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


    //A operators
    /*!A_NC_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_A_nc;

    //B operators
    /*!B_NC_i*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_B_nc;

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
    * @param Xnc functional matrix containing the non-stationary covariates (nxqnc)
    * @param Wnc vector of functional diagonal matrix containing, as element i-th, the functional non-stationary weights (nxn) of unit i-th
    * @param Rnc penalization matrix of the non-stationary covariates (diagonal block matrix containing the the scalar product within the second order derivatives of the functional regression coefficients basis. LncxLnc)
    * @param eta functional sparse matrix containing the basis of the functional regression coefficients of the non-stationary covariates (qncxLnc, row i-th contains zeros and the basis of the i-th non-stationary covariate, their position shifted of sum_i_0_to_i(Lnc_i))
    * @param qnc number of non-stationary covariates
    * @param Lnc total number of basis used for the non-stationary covariates functional regression coefficients
    * @param Lnc_j vector containing in element i-th the number of basis for the non-stationary covariate i-th functional regression coefficients
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
    fwr_FMGWR(FUNC_MATRIX_OBJ &&y,
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
                assert((m_phi.rows() == this->n()) && (m_phi.cols() == m_Ly*(this->n())));
                assert((m_c.rows() == m_Ly*(this->n())) && (m_c.cols() == 1));
                //stationary covariates
                assert((m_Xc.rows() == this->n()) && (m_Xc.cols() == m_qc));
                assert((m_Wc.rows() == this->n()) && (m_Wc.cols() == this->n()));
                assert((m_Rc.rows() == m_Lc) && (m_Rc.cols() == m_Lc));
                assert((m_omega.rows() == m_qc) && (m_omega.cols() == m_Lc));
                assert((m_Lc_j.size() == m_qc) && (std::reduce(m_Lc_j.cbegin(),m_Lc_j.cend(),static_cast<std::size_t>(0)) == m_Lc));
                //non stationary covariates
                assert((m_Xnc.rows() == this->n()) && (m_Xnc.cols() == m_qnc));
                assert(m_Wnc.size() == this->n());
                for(std::size_t i = 0; i < m_Wnc.size(); ++i){   assert((m_Wnc[i].rows() == this->n()) && (m_Wnc[i].cols() == this->n()));}
                assert((m_Rnc.rows() == m_Lnc) && (m_Rnc.cols() == m_Lnc));
                assert((m_eta.rows() == m_qnc) && (m_eta.cols() == m_Lnc));
                assert((m_Lnc_j.size() == m_qnc) && (std::reduce(m_Lnc_j.cbegin(),m_Lnc_j.cend(),static_cast<std::size_t>(0)) == m_Lnc));

                //compute all the transpost necessary for the computations
                m_Xc_t = m_Xc.transpose();
                m_omega_t = m_omega.transpose();
                m_Xnc_t = m_Xnc.transpose();
                m_eta_t = m_eta.transpose();
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
        if(!this->in_cascade_estimation())
        {
            //(j_tilde + Rnc)^-1 (LncxLnc)
            std::cout << "Computing (j_tilde + Rnc)^-1" << std::endl;
            std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_tilde_Rnc_inv = this->operator_comp().compute_penalty(m_eta_t,m_Xnc_t,m_Wnc,m_Xnc,m_eta,m_Rnc);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
            //A_NC_i (n elements Lncx(n*Ly))
            std::cout << "Computing A_nc_i" << std::endl;
            m_A_nc = this->operator_comp().compute_operator(m_eta_t,m_Xnc_t,m_Wnc,m_phi,j_tilde_Rnc_inv);
            //H_nc(t) (nx(n*Ly))
            std::cout << "Computing H_nc(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> H_nc = this->operator_comp().compute_functional_operator(m_Xnc,m_eta,m_A_nc);
            //B_NC_i (n elements LncxLc)
            std::cout << "Computing B_nc_i" << std::endl;
            m_B_nc = this->operator_comp().compute_operator(m_eta_t,m_Xnc_t,m_Wnc,m_Xc,m_omega,j_tilde_Rnc_inv);
            //K_nc_c(t) (nxLc)
            std::cout << "Computing K_nc_c(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> K_nc_c = this->operator_comp().compute_functional_operator(m_Xnc,m_eta,m_B_nc);

            //y_new(t) (nx1)
            std::cout << "Computing y_new(t)" << std::endl;
            functional_matrix<INPUT,OUTPUT> y_new = fm_prod(functional_matrix<INPUT,OUTPUT>(m_phi - H_nc),m_c,this->number_threads());
            //Xc_crossed (nxqnc)
            functional_matrix<INPUT,OUTPUT> X_c_crossed = fm_prod(m_Xc,m_omega) - K_nc_c;
            functional_matrix<INPUT,OUTPUT> X_c_crossed_t = X_c_crossed.transpose();
            //[J + Rc]^-1 (LcxLc)
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
            functional_matrix<INPUT,OUTPUT> y_tilde_new = fm_prod(m_phi,m_c_tilde_hat);


            //COMPUTING all the m_bnc, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATION-DEPENDENT BETAS (n elements Lncx1)
            std::cout << "Computing bnc" << std::endl;
            m_bnc = this->operator_comp().compute_operator(m_eta_t,m_Xnc_t,m_Wnc,y_tilde_new,j_tilde_Rnc_inv);
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
            //[J_i + Rnc]^-1 (LncxLnc)
            std::cout << "Computing (j + Rnc)^-1" << std::endl;
            std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_i_Rnc_inv = this->operator_comp().compute_penalty(m_eta_t,m_Xnc_t,m_Wnc,m_Xnc,m_eta,m_Rnc);
            //COMPUTING m_bnc, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATIONARY BETAS (n elements Lncx1)
            std::cout << "Computing bnc" << std::endl;
            m_bnc = this->operator_comp().compute_operator(m_eta_t,m_Xnc_t,m_Wnc,y_tilde,j_i_Rnc_inv);
            //extra (default values of 0s)
            m_c_tilde_hat = Eigen::MatrixXd::Zero(m_Ly*this->n(),1);
        }


        //
        //wrapping the b from the shape useful for the computation into a more useful format
        //
        //stationary covariates (qc elements 1xLc_i)
        m_Bc  = this->operator_comp().wrap_operator(m_bc,m_Lc_j,m_qc);
        //non-stationary covariates (qnc elements of n elements 1xLnc_i)
        m_Bnc = this->operator_comp().wrap_operator(m_bnc,m_Lnc_j,m_qnc,this->n());
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
        m_beta_c  = this->operator_comp().eval_func_betas(m_Bc,m_omega,m_Lc_j,m_qc,this->abscissa_points());        
        //BETA_NC
        m_beta_nc = this->operator_comp().eval_func_betas(m_Bnc,m_eta,m_Lnc_j,m_qnc,this->n(),this->abscissa_points());
    }

    /*!
    * @brief Function to return the basis expansion coefficients of the functional regression coefficitens
    * @return a tuple containing m_Bc and m_Bnc
    */
    inline 
    BTuple 
    bCoefficients()
    const 
    override
    {
        return std::tuple{m_Bc,m_Bnc};
    }

    /*!
    * @brief Function to return the the functional regression coefficients evaluated
    * @return a tuple containing m_beta_c and m_beta_nc
    */
    inline 
    BetasTuple 
    betas() 
    const
    override
    {
        return std::tuple{m_beta_c,m_beta_nc};
    }

    /*!
    * @brief Function to return objects useful for reconstructing the functional partial residuals
    * @return a tuple containing m_c_tilde_hat 
    */
    inline
    PartialResidualTuple
    PRes()
    const
    override
    {
        return std::tuple{m_c_tilde_hat};
    }
};

#endif  /*FWR_FMGWR_ALGO_HPP*/