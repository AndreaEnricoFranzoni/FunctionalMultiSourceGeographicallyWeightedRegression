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


#ifndef FWR_FWR_ALGO_HPP
#define FWR_FWR_ALGO_HPP

#include "fwr.hpp"


/*!
* @file fwr_FWR.hpp
* @brief Contains the definition of the Functional Weighted Regression model
* @author Andrea Enrico Franzoni
*/



/*!
* @class fwr_FWR
* @brief Concrete class for the Functional Weighted Regression model, estimating stationary functional regression coefficients
* @tparam INPUT type of functional data abscissa
* @tparam OUTPUT type of functional data image
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_FWR final : public fwr<INPUT,OUTPUT>
{
private:
    /*!Functional response (nx1)*/
    functional_matrix<INPUT,OUTPUT> m_y;

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



public:
    /*!
    * @brief Constructor
    * @param y functional matrix containing the response (nx1)
    * @param Xc functional matrix containing the stationary covariates (nxqc)
    * @param Wc functional diagonal matrix containing the functional stationary weights (nxn)
    * @param Rc penalization matrix of the stationary covariates (diagonal block matrix containing the the scalar product within the second order derivatives of the functional regression coefficients basis. LcxLc)
    * @param omega functional sparse matrix containing the basis of the functional regression coefficients of the stationary covariates (qcxLc, row i-th contains zeros and the basis of the i-th stationary covariate, their position shifted of sum_i_0_to_i(Lc_i))
    * @param qc number of stationary covariates
    * @param Lc total number of basis used for the stationary covariates functional regression coefficients
    * @param Lc_j vector containing in element i-th the number of basis for the stationary covariate i-th functional regression coefficients
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
             typename FUNC_DIAG_MATRIX_OBJ, 
             typename SCALAR_SPARSE_MATRIX_OBJ> 
    fwr_FWR(FUNC_MATRIX_OBJ &&y,
            FUNC_MATRIX_OBJ &&Xc,
            FUNC_DIAG_MATRIX_OBJ &&Wc,
            SCALAR_SPARSE_MATRIX_OBJ &&Rc,
            FUNC_SPARSE_MATRIX_OBJ &&omega,
            std::size_t qc,
            std::size_t Lc,
            const std::vector<std::size_t> & Lc_j,
            INPUT a,
            INPUT b,
            int n_intervals_integration,
            const std::vector<INPUT> & abscissa_points,
            std::size_t n,
            int number_threads)
        :
            fwr<INPUT,OUTPUT>(a,b,n_intervals_integration,abscissa_points,n,number_threads,false),
            m_y{std::forward<FUNC_MATRIX_OBJ>(y)},
            m_Xc{std::forward<FUNC_MATRIX_OBJ>(Xc)},
            m_Wc{std::forward<FUNC_DIAG_MATRIX_OBJ>(Wc)},
            m_Rc{std::forward<SCALAR_SPARSE_MATRIX_OBJ>(Rc)},
            m_omega{std::forward<FUNC_SPARSE_MATRIX_OBJ>(omega)},
            m_qc(qc),
            m_Lc(Lc),
            m_Lc_j(Lc_j)
            {
                //checking input consistency
                //response
                assert((m_y.rows() == this->n()) && (m_y.cols() == 1));
                //stationary covariates
                assert((m_Xc.rows() == this->n()) && (m_Xc.cols() == m_qc));
                assert((m_Wc.rows() == this->n()) && (m_Wc.cols() == this->n()));
                assert((m_Rc.rows() == m_Lc) && (m_Rc.cols() == m_Lc));
                assert((m_omega.rows() == m_qc) && (m_omega.cols() == m_Lc));
                assert((m_Lc_j.size() == m_qc) && (std::reduce(m_Lc_j.cbegin(),m_Lc_j.cend(),static_cast<std::size_t>(0)) == m_Lc));

                //compute all the transpost necessary for the computations
                m_Xc_t = m_Xc.transpose();
                m_omega_t = m_omega.transpose();
            }
    

    /*!
    * @brief Method to compute the Functional Weighted Regression basis expansion coefficients of the functional regression coefficients
    */
    inline 
    void 
    compute()  
    override
    {
        //[J + Rc]^-1 (LcxLc)
        std::cout << "Computing (j + Rc)^-1" << std::endl;
        Eigen::PartialPivLU<FDAGWR_TRAITS::Dense_Matrix> j_Rc_inv = this->operator_comp().compute_penalty(m_omega_t,m_Xc_t,m_Wc,m_Xc,m_omega,m_Rc);

        //COMPUTING m_bc, SO THE COEFFICIENTS FOR THE BASIS EXPANSION OF THE STATIONARY BETAS (Lcx1)
        std::cout << "Computing bc" << std::endl;
        m_bc = this->operator_comp().compute_operator(m_omega_t,m_Xc_t,m_Wc,m_y,j_Rc_inv);

        //
        //wrapping the b from the shape useful for the computation into a more useful format
        //
        //stationary covariates (qc elements 1xLc_i)
        m_Bc = this->operator_comp().wrap_operator(m_bc,m_Lc_j,m_qc);
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
    }

    /*!
    * @brief Function to return the basis expansion coefficients of the functional regression coefficitens
    * @return a tuple containing m_Bc
    */
    inline 
    BTuple 
    bCoefficients()
    const 
    override
    {
        return std::tuple{m_Bc};
    }

    /*!
    * @brief Function to return the the functional regression coefficients evaluated
    * @return a tuple containing m_beta_c 
    */
    inline 
    BetasTuple 
    betas() 
    const
    override
    {
        return std::tuple{m_beta_c};
    }

    /*!
    * @brief Function to return objects useful for reconstructing the functional partial residuals
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

#endif  /*FWR_FWR_ALGO_HPP*/