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


#ifndef FGWR_PREDICTOR_HPP
#define FGWR_PREDICTOR_HPP


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"

#include "../functional_matrix/functional_matrix.hpp"
#include "../functional_matrix/functional_matrix_sparse.hpp"
#include "../functional_matrix/functional_matrix_diagonal.hpp"
#include "../functional_matrix/functional_matrix_product.hpp"
#include "../functional_matrix/functional_matrix_operators.hpp"
#include "../functional_matrix/functional_matrix_smoothing.hpp"

#include "../integration/functional_data_integration.hpp"
#include "../basis/basis_include.hpp"
#include "../utility/parameters_wrapper_fdagwr.hpp"

#include <cassert>

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fgwr_predictor
{


private:
    /*!Object to perform the integration using trapezoidal quadrature rule*/
    fd_integration m_integrating;
    /*!Number of statistical units used to train the model*/
    std::size_t m_n_train;
    /*!Number of threads for OMP*/
    int m_number_threads;

public:
    /*!
    * @brief Constructor
    * @param number_threads number of threads for OMP
    */
    fgwr_predictor(INPUT a, INPUT b, int n_intervals_integration, double target_error, int max_iterations, std::size_t n_train, int number_threads)
                   : m_integrating(a,b,n_intervals_integration,target_error,max_iterations), m_n_train(n_train), m_number_threads(number_threads) {}

    /*!
    * @brief Virtual destructor
    */
    virtual ~fgwr_predictor() = default;

    /*!
    * @brief Getter for the number of statistical units
    * @return the private m_n
    */
    inline std::size_t n_train() const {return m_n_train;}

    /*!
    * @brief Getter for the number of threads for OMP
    * @return the private m_number_threads
    */
    inline int number_threads() const {return m_number_threads;}

    /*!
    * @brief Id
    */
    static constexpr std::string id_C = COVARIATES_NAMES::Stationary;

    /*!
    * @brief Integrating element-wise a functional matrix
    */
    inline
    FDAGWR_TRAITS::Dense_Matrix
    fm_integration(const functional_matrix<INPUT,OUTPUT> &integrand)
    const
    {
        std::vector<OUTPUT> result_integrand;
        result_integrand.resize(integrand.size());

#ifdef _OPENMP
#pragma omp parallel for shared(integrand,m_integrating,result_integrand) num_threads(m_number_threads)
#endif
        for(std::size_t i = 0; i < integrand.size(); ++i){
            result_integrand[i] = m_integrating.integrate(integrand.as_vector()[i]);}

        FDAGWR_TRAITS::Dense_Matrix result_integration = Eigen::Map< FDAGWR_TRAITS::Dense_Matrix >(result_integrand.data(), integrand.rows(), integrand.cols());

        return result_integration;
    }

    /*!
    * @brief Compute all the [J_2_tilde_i + R]^(-1): 
    * @note FATTO
    */
    std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > >
    compute_penalty(const functional_matrix_sparse<INPUT,OUTPUT> &base_t,
                    const functional_matrix<INPUT,OUTPUT> &X_t,
                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                    const functional_matrix<INPUT,OUTPUT> &X,
                    const functional_matrix_sparse<INPUT,OUTPUT> &base,
                    const FDAGWR_TRAITS::Sparse_Matrix &R) const;

    /*!
    * @brief Compute [J_tilde_i + R]^(-1)
    * @note FATTO
    */
    inline
    std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > >
    compute_penalty(const functional_matrix<INPUT,OUTPUT> &X_crossed_t,
                    const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                    const functional_matrix<INPUT,OUTPUT> &X_crossed,
                    const FDAGWR_TRAITS::Sparse_Matrix &R) const;


    /*!
    * @brief Compute an operator
    * @note FATTO
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;

    /*!
    * @brief Compute an operator
    * @note FATTO
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    compute_operator(const functional_matrix_sparse<INPUT,OUTPUT> &base_lhs,
                     const functional_matrix<INPUT,OUTPUT> &X_lhs,
                     const std::vector< functional_matrix_diagonal<INPUT,OUTPUT> > &W,
                     const functional_matrix<INPUT,OUTPUT> &X_rhs,
                     const std::vector< Eigen::PartialPivLU< FDAGWR_TRAITS::Dense_Matrix > > &penalty) const;


    /*!
    * @brief Compute a functional operator
    * @note FATTO
    */
    functional_matrix<INPUT,OUTPUT> 
    compute_functional_operator(const functional_matrix<INPUT,OUTPUT> &X,
                                const functional_matrix_sparse<INPUT,OUTPUT> &base,
                                const std::vector< FDAGWR_TRAITS::Dense_Matrix > &operator_) const;

    /*!
    * @brief Wrap b, for non-stationary covariates: me li sistema tutti non in colonna, uno per covariata 
    */
    std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>
    wrap_b(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
           const std::vector<std::size_t>& L_j,
           std::size_t q,
           std::size_t n_pred) const;

    /*!
    * @brief Dewrap b, for stationary covariates: me li incolonna tutti 
    */
    FDAGWR_TRAITS::Dense_Matrix 
    dewrap_b(const std::vector< FDAGWR_TRAITS::Dense_Matrix >& b,
             const std::vector<std::size_t>& L_j) const;

    /*!
    * @brief Dewrap b, for non-stationary covariates: me li incolonna tutti
    */
    std::vector< FDAGWR_TRAITS::Dense_Matrix >
    dewrap_b(const std::vector< std::vector< FDAGWR_TRAITS::Dense_Matrix >>& b,
             const std::vector<std::size_t>& L_j,
             std::size_t n) const;

    /*!
    * @brief Eval the stationary betas on a grid
    */
    std::vector< std::vector<OUTPUT> >
    eval_betas(const functional_matrix<INPUT,OUTPUT> &beta,
               std::size_t q,
               std::vector<INPUT> abscissa) const;

    /*!
    * @brief Eval the non-stationary betas on a grid
    */
    std::vector< std::vector< std::vector<OUTPUT>>>
    eval_betas(const std::vector< functional_matrix<INPUT,OUTPUT>> &beta,
               std::size_t q,
               std::vector<INPUT> abscissa) const;

    /*!
    * @brief Compute partial residuals
    */
    virtual inline void computePartialResiduals() = 0;

    /*!
    * @brief Updating the non-stationary betas
    */                            
    virtual inline void computeBNew(const std::map<std::string,std::vector< functional_matrix_diagonal<INPUT,OUTPUT> >> &W) = 0;

    /*!
    * @brief Compute stationary beta
    */
    virtual inline void computeStationaryBetas() = 0;

    /*!
    * @brief Compute non-stationary betas
    */
    virtual inline void computeNonStationaryBetas() = 0;

    /*!
    * @brief Compute prediction
    */
    virtual inline functional_matrix<INPUT,OUTPUT> predict(const std::map<std::string,functional_matrix<INPUT,OUTPUT>>& X_new) const = 0;

    /*!
    * @brief Virtual method to compute the betas
    */
    virtual inline void evalBetas(const std::vector<INPUT> &abscissa) = 0;

    /*!
    * Function to evaluate the prediction
    */
    inline
    std::vector< std::vector<OUTPUT>>
    evalPred(const functional_matrix<INPUT,OUTPUT> &pred,
             const std::vector<INPUT> &abscissa)
    const
    {
        assert(pred.cols() == 1);
        std::size_t n_pred = pred.rows();
        std::size_t n_abs  = abscissa.size();

        std::vector< std::vector<OUTPUT>> evaluations_pred;
        evaluations_pred.resize(n_pred);

#ifdef _OPENMP
#pragma omp parallel for shared(pred,n_pred,abscissa,n_abs) num_threads(this->number_threads())
#endif
        for(std::size_t i = 0; i < n_pred; ++i)
        {
            std::vector<OUTPUT> evaluations_pred_i;
            evaluations_pred_i.resize(n_abs);

            std::transform(abscissa.cbegin(),
                           abscissa.cend(),
                           evaluations_pred_i.begin(),
                           [&pred,i](fm_utils::input_param_t<FUNC_OBJ<INPUT,OUTPUT>> x){return pred(i,0)(x);});

            evaluations_pred[i] = evaluations_pred_i;
        }

        return evaluations_pred;
    }

    /*!
    * @brief Function to return the coefficients of the betas basis expansion, tuple of different dimension depending on the algo used
    */
    virtual inline BTuple bCoefficients() const = 0;

    /*!
    * @brief Function to return the the betas evaluated, tuple of different dimension depending on the algo used
    */
    virtual inline BetasTuple betas() const = 0;
};

#include "fgwr_predictor_imp.hpp"

#endif  /*FGWR_PREDICTOR_HPP*/