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


#ifndef FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_NON_STATIONARY_HPP
#define FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_NON_STATIONARY_HPP

#include "functional_weight_matrix.hpp"
#include "distance_matrix.hpp"



/*!
* @file functional_weight_matrix_no_stat.hpp
* @brief Construct the non stationary weight matrix for performing the geographically weighted regression. Weights consist of functional reconstruction weights and spatial weights
* @author Andrea Enrico Franzoni
*/



/*!
* Doing tag dispatching for the correct way of evaluating the non stationary weights (kernel function for the distances)
* @tparam err_eval: template parameter for the error evaluation strategy
*/
template <KERNEL_FUNC kernel_func>
using KERNEL_FUNC_T = std::integral_constant<KERNEL_FUNC, kernel_func>;


template< class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis, FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::NON_STATIONARY, KERNEL_FUNC kernel_func = KERNEL_FUNC::GAUSSIAN, DISTANCE_MEASURE dist_meas = DISTANCE_MEASURE::EUCLIDEAN >  
    requires fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
class functional_weight_matrix_non_stationary : public functional_weight_matrix_base< functional_weight_matrix_non_stationary<domain_type,basis_type,stationarity_t,kernel_func,dist_meas>, domain_type, basis_type >
{

private:
    /*!Vector of diagonal matrices storing the weights*/
    WeightMatrixType<stationarity_t> m_weights;
    /*!Distance matrix*/
    distance_matrix<dist_meas> m_distance_matrix;
    /*!Kernel bandwith*/
    double m_kernel_bandwith;
    /*!
    * @brief Evaluation of the kernel function for the non stationary weights
    * @param distance distance between two locations
    * @param bandwith kernel bandwith
    * @return the evaluation of the kernel function
    */
    double kernel_eval(double distance, double bandwith, KERNEL_FUNC_T<KERNEL_FUNC::GAUSSIAN>) const;


public:

    /*!
    * @brief Constructor for the non stationary weight matrix: each weight consists of the reconstruction functional weight and spatial weight
    * @param weight_stat stationary weight, for each statistical unit
    * @param n number of statistical units
    * @param number_threads number of threads for OMP
    */
    template< typename DIST_MATRIX_OBJ >
    functional_weight_matrix_non_stationary(const functional_data<domain_type,basis_type> &y_recostruction_weights,
                                            DIST_MATRIX_OBJ&& distance_matrix,
                                            double kernel_bwt,
                                            int number_threads)
                                : 
                                  functional_weight_matrix_base<functional_weight_matrix_non_stationary>(y_recostruction_weights,
                                                                                                         number_threads),
                                  m_distance_matrix{std::forward<DIST_MATRIX_OBJ>(distance_matrix)},
                                  m_kernel_bandwith(kernel_bwt) 
                                {                                       
                                    static_assert(stationarity_t == FDAGWR_COVARIATES_TYPES::NON_STATIONARY   ||
                                                  stationarity_t == FDAGWR_COVARIATES_TYPES::EVENT            ||
                                                  stationarity_t == FDAGWR_COVARIATES_TYPES::STATION,
                                                  "Functional weight matrix for non stationary covariates needs FDAGWR_COVARIATES_TYPES::NON_STATIONARY or FDAGWR_COVARIATES_TYPES::EVENT or FDAGWR_COVARIATES_TYPES::STATION as template parameter");
                                }

    /*!
    * @brief Getter for the functional non-stationary weight matrix
    * @return the private m_weights
    */
    const WeightMatrixType<stationarity_t>& weights() const {return m_weights;}

    /*!
    * @brief Evaluation of kernel function for the non-stationary weights. Tag-dispacther.
    * @param distance distance between two locations
    * @param bandwith kernel bandwith
    * @return the evaluation of the kernel function
    */
    double kernel_eval(double distance, double bandwith) const { return kernel_eval(distance,bandwith,KERNEL_FUNC_T<kernel_func>{});};

    /*!
    * @brief Function to compute non stationary weights
    * @details Semantic strongly influeneced by Eigen
    */
    inline
    void
    computing_weights()
    {
      //to shared the values with OMP
      auto n_stat_units = this->n();
      //preparing the container for the functional non-stationary weight matrix
      m_weights.resize(n_stat_units);

#ifdef _OPENMP
#pragma omp parallel for shared(m_distance_matrix,n_stat_units) num_threads(this->number_threads())
#endif
      for(std::size_t i = 0; i < n_stat_units; ++i)
      {
        //non stationary weights: applying the kernel to the distances within statistical units
        auto weights_non_stat_unit_i = m_distance_matrix[i]; //Eigen vector with the distances with respect to unit i-th
        
        //applying the kernel function to correctly smoothing the distances
        std::transform(weights_non_stat_unit_i.data(),
                       weights_non_stat_unit_i.data() + weights_non_stat_unit_i.size(),
                       weights_non_stat_unit_i.data(),
                       [this](double dist){return this->kernel_eval(dist,this->m_kernel_bandwith);});

        //preparing the container for the functional non-stationary matrix of unit i-th (corresponding to index unit_index)
        std::vector< FDAGWR_TRAITS::f_type > weights_unit_i;
        weights_unit_i.reserve(n_stat_units);

        //computing the functional non-stationary matrix for unit i-th (corresponding to index unit_index),
        //  stationary and non-stationary weights interacting
        for (std::size_t j = 0; j < n_stat_units; ++j)
        {
          //the product element-wise between stationary weights at abscissa abscissa_index-th and non-stationary weight unit_index-th
          //is performed in the most efficient possible way, but it returns an Eigen::ArrayXd: the .matrix() cast it to an Eigen::VectorXd,
          //allowing to call the constructor for an Eigen::DiagonalMatrix
          //weights_unit_i.emplace_back((weights_non_stat_unit_i.array() * this->coeff_stat_weights_abscissa_i(abscissa_index).array()).matrix());
          
          //fdagwr_traits::Diag_Matrix w((weights_non_stat_unit_i.array() * this->coeff_stat_weights_abscissa_i(abscissa_index).array()).matrix());
          //weights_unit_i.push_back(w);
          
          
          
          
          double alpha_i_j = weights_non_stat_unit_i[j];
          FDAGWR_TRAITS::f_type w_i_j = [=i,=alpha_i_j](const double & loc){return alpha_i_j * this->y_recostruction_weights().eval(loc,i);};
          weights_unit_i.push_back(w_i_j);
        }
        
        //storing the functional non-stationary matrix for unit i-th (corresponding to index unit_index)
        m_weights[i] = weights_unit_i;
      }
    }
};

#include "functional_weight_matrix_kernel_functions_eval.hpp"

#endif  /*FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_NON_STATIONARY_HPP*/