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


#ifndef FWR_PREDICTOR_HPP
#define FWR_PREDICTOR_HPP


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"

#include "../integration/fwr_operator_computing.hpp"
#include "../functional_matrix/functional_matrix_smoothing.hpp"
#include "../basis/basis_include.hpp"
#include "../utility/parameters_wrapper_fdagwr.hpp"

#include <cassert>
#include <iostream>

/*!
* @file fwr_predictor.hpp
* @brief Contains the definition of a the virtual base class for the functional weighted regression predictor
* @author Andrea Enrico Franzoni
*/


/*!
* @class fwr_predictor
* @brief Base class for the functional weighted regression predictor, virtual interface
* @tparam INPUT type of functional data abscissa
* @tparam OUTPUT type of functional data image
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_predictor
{
private:
    /*!Object to perform the integration using midpoint quadrature rule*/
    fwr_operator_computing<INPUT,OUTPUT> m_operator_comp;
    /*!Number of statistical units used to train the model*/
    std::size_t m_n_train;
    /*!Number of threads for OMP*/
    int m_number_threads;
    /*!If false, performing fwr with more than one source in an exact way. If true, estimating the coefficients in cascade*/
    bool m_in_cascade_estimation;

public:
    /*!
    * @brief Constructor
    * @param a left extreme functional data domain 
    * @param a right extreme functional data domain 
    * @param n_intervals_integration number of intervals used by the midpoint quadrature rule
    * @param n_train number of training statistical units
    * @param number_threads number of threads for OMP
    * @param in_cascade_estimation if true, for more than one source covariates, the estimation is made in cascade. If false, exact
    */
    fwr_predictor(INPUT a, INPUT b, int n_intervals_integration, std::size_t n_train, int number_threads, bool in_cascade_estimation)
                   : m_operator_comp(a,b,n_intervals_integration,number_threads), m_n_train(n_train), m_number_threads(number_threads), m_in_cascade_estimation(in_cascade_estimation) {}

    /*!
    * @brief Virtual destructor
    */
    virtual ~fwr_predictor() = default;

    /*!
    * @brief ID for the key of stationary covariates in the maps used as input in the predictor functions
    */
    inline static constexpr std::string_view id_C  = COVARIATES_NAMES::Stationary;

    /*!
    * @brief ID for the key of non-stationary covariates in the maps used as input in the predictor functions
    */
    inline static constexpr std::string_view id_NC = COVARIATES_NAMES::Nonstationary;

    /*!
    * @brief ID for the key of event-dependent covariates in the maps used as input in the predictor functions
    */
    inline static constexpr std::string_view id_E  = COVARIATES_NAMES::Event;

    /*!
    * @brief ID for the key of station-dependent covariates in the maps used as input in the predictor functions
    */
    inline static constexpr std::string_view id_S  = COVARIATES_NAMES::Station;

    /*!
    * @brief Getter for operator that computes operators, scalar and functional
    * @return the private m_operator_comp
    */
    const fwr_operator_computing<INPUT,OUTPUT>& operator_comp() const {return m_operator_comp;}

    /*!
    * @brief Getter for the number of statistical units used in the training set
    * @return the private m_n_train
    */
    inline std::size_t n_train() const {return m_n_train;}

    /*!
    * @brief Getter for the number of threads for OMP
    * @return the private m_number_threads
    */
    inline int number_threads() const {return m_number_threads;}

    /*!
    * @brief Getter for how the estimation is performed
    * @return the private m_in_cascade_estimation
    */
    inline bool in_cascade_estimation() const {return m_in_cascade_estimation;}

    /*!
    * @brief Function to evaluate the prediction on a grid
    * @param pred functional matrix containing the prediction on the prediction set
    * @param abscissa the grid over which evaluating the prediction
    * @return a vector containing, for each unit to be predicted, the predicted response evaluation
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

            //evaluating the i-th predicted unit
            std::transform(abscissa.cbegin(),
                           abscissa.cend(),
                           evaluations_pred_i.begin(),
                           [&pred,i](fm_utils::input_param_t<FUNC_OBJ<INPUT,OUTPUT>> x){return pred(i,0)(x);});

            evaluations_pred[i] = evaluations_pred_i;
        }

        return evaluations_pred;
    }

    /*!
    * @brief Function to perform the smoothing of the prediction
    * @param pred functional matrix containing the prediction on the prediction set, n predicted units
    * @param basis basis for performing the smoothing, n_basis
    * @param knots knots for performing the smoothing
    * @return a Eigen::MatrixXd of dimension n_basis x n, containing the coefficient of the basis expansion, where el(i,j) is the coefficient of basis i-th of the basis expansion of the function j-th
    */
    template< class domain_type = FDAGWR_TRAITS::basis_geometry > 
    FDAGWR_TRAITS::Dense_Matrix
    smoothPred(const functional_matrix<INPUT,OUTPUT> &pred,
               const basis_base_class<domain_type> &basis,
               const FDAGWR_TRAITS::Dense_Matrix &knots)
    const
    {
        return fm_smoothing<INPUT,OUTPUT,domain_type>(pred,basis,knots); 
    }

    /*!
    * @brief Function to compute the partial residuals accordingly to the fitted model
    */
    virtual inline void computePartialResiduals() = 0;

    /*!
    * @brief Updating the non-stationary betas on the units to be predicted
    * @param W map containing the non-stationary covariates of the units to be predicted. Each key represents, accordingly to the fitted model, a specific type of covariates
    * @note keys are the one stored in the base class. Input coherency is checked
    */                            
    virtual inline void computeBNew(const std::map<std::string,std::vector< functional_matrix_diagonal<INPUT,OUTPUT> >> &W) = 0;

    /*!
    * @brief Compute stationary betas as functional matrices
    */
    virtual inline void computeStationaryBetas() = 0;

    /*!
    * @brief Compute non-stationary betas on the to-be-predicted units as functional matrices
    */
    virtual inline void computeNonStationaryBetas() = 0;

    /*!
    * @brief Compute prediction
    * @param X_new map containig the covariates of the units to be predicted. Each key represents, accordingly to the fitted model, a specific type of covariates
    * @return a functional matrix containing the prediction, n_predx1
    * @note keys are the one stored in the base class. Input coherency is checked
    */
    virtual inline functional_matrix<INPUT,OUTPUT> predict(const std::map<std::string,functional_matrix<INPUT,OUTPUT>>& X_new) const = 0;

    /*!
    * @brief Virtual method to evaluate the functional betas along a grid
    * @param abscissa the grid over which evaluating the functional betas
    */
    virtual inline void evalBetas(const std::vector<INPUT> &abscissa) = 0;

    /*!
    * @brief Function to return the coefficients of the betas basis expansion
    * @return a tuple of different dimension depending on the model fitted, containing the coefficients
    */
    virtual inline BTuple bCoefficients() const = 0;

    /*!
    * @brief Function to return the the betas evaluated, tuple of different dimension depending on the model fitted
    * @return a tuple of different dimension depending on the model fitted, containing the betas evaluated
    */
    virtual inline BetasTuple betas() const = 0;
};

#endif  /*FWR_PREDICTOR_HPP*/