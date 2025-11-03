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


#ifndef FWR_FGWR_ALGO_HPP
#define FWR_FGWR_ALGO_HPP

#include "fwr.hpp"



/*!
* @file fwr_FGWR.hpp
* @brief Contains the definition of the Functional Geographically Weighted Regression model
* @author Andrea Enrico Franzoni
*/



/*!
* @class fwr_FGWR
* @brief Concrete class for the Functional Geographically Weighted Regression model, estimating non-stationary functional regression coefficients
* @tparam INPUT type of functional data abscissa
* @tparam OUTPUT type of functional data image
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_FGWR final : public fwr<INPUT,OUTPUT>
{
private:
    /*!Functional response (nx1)*/
    functional_matrix<INPUT,OUTPUT> m_y;

    /*!Functional non-stationary covariates (n x qnc)*/
    functional_matrix<INPUT,OUTPUT> m_Xnc;
    /*!Their transpost (qnc x n)*/
    functional_matrix<INPUT,OUTPUT> m_Xnc_t;
    /*!Functional weights for non-stationary covariates (n elements of diagonal n x n)*/
    std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > m_Wnc;
    /*!Scalar matrix with the penalization on the non-stationary covariates (sparse Lnc x Lnc, where Lnc is the sum of the basis of each NC covariate)*/
    FDAGWR_TRAITS::Sparse_Matrix m_Rnc;
    /*!Basis for non-stationary covariates regressors (sparse qnc x Lnc)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_eta;
    /*!Their transpost (sparse Lnc x qnc)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_eta_t;
    /*!Coefficients of the basis expansion for non-stationary regressors: Lncx1, every element of the vector is referring to a specific unit: TO BE COMPUTED*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_bnc;
    /*!Coefficients of the basis expansion for non-stationary regressors coefficients: every of the qnc elements are n 1xLnc_j matrices, one for each statistical unit*/
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >> m_Bnc;
    /*!Discrete evaluation of all the beta_nc: a vector of dimension qnc, containing, for all the non-stationary covariates, the discrete ev of the respective beta, for each statistical unit*/
    std::vector< std::vector< std::vector< OUTPUT >>> m_beta_nc;
    /*!Number of non-stationary covariates*/
    std::size_t m_qnc;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the non-stationary regressors coefficients*/
    std::size_t m_Lnc; 
    /*!Number of basis, for each non-stationary covariate, to perform the basis expansion of the regressors coefficients for the non-stationary regressors coefficients*/
    std::vector<std::size_t> m_Lnc_j;


public:
    /*!
    * @brief Constructor
    * @param y functional matrix containing the response (nx1)
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
    * @note input dimensions check and transpose computation
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
        //(j + Rnc)^-1 (LncxLnc)
        std::cout << "Computing (j_tilde + Rnc)^-1" << std::endl;
        std::vector< Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> > j_Rnc_inv = this->operator_comp().compute_penalty(m_eta_t,m_Xnc_t,m_Wnc,m_Xnc,m_eta,m_Rnc);     //per applicarlo: j_double_tilde_RE_inv[i].solve(M) equivale a ([J_i_tilde_tilde + Re]^-1)*M
        //COMPUTING all the m_bnc, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATION-DEPENDENT BETAS (n elements Lncx1)
        std::cout << "Computing bnc" << std::endl;
        m_bnc = this->operator_comp().compute_operator(m_eta_t,m_Xnc_t,m_Wnc,m_y,j_Rnc_inv);

        //
        //wrapping the b from the shape useful for the computation into a more useful format
        //
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
    * @brief Function to return the basis expansion coefficients of the functional regression coefficitens
    * @return a tuple containing m_Bnc
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
    * @brief Function to return the the functional regression coefficients evaluated
    * @return an empty tuple
    * @note no necessity to reconstruct partial residuals since there is only one type of covariates
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