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

#ifndef FDAGWR_DISTANCE_MATRIX_PRED_HPP
#define FDAGWR_DISTANCE_MATRIX_PRED_HPP


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"
#include <cassert>


/*!
* @file distance_matrix_pred.hpp
* @brief Class for computing the distances within the statistical units of the fitted model and the one to be predicted
* @author Andrea Enrico Franzoni
*/



/*!
* @brief Tag dispatching for how to compute the distance measure
* @tparam distance_measure: template parameter for the distance measure employed
*/
template <DISTANCE_MEASURE distance_measure>
using DISTANCE_MEASURE_T = std::integral_constant<DISTANCE_MEASURE, distance_measure>;



/*!
* @class distance_matrix_pred
* @tparam distance_measure how to compute the distances within different units
* @brief Class for constructing the distance matrix within statistical units used in the training stage (n_train) and the one to be predicted (n_pred): n_pred distance matrices of dim n_train x n_train
*/
template< DISTANCE_MEASURE distance_measure >
class distance_matrix_pred
{
private:

    /*!Every element of the outer vector represents a units to be predicted: every inner vector represent the distance matrix within that unit to be predicted and all the one in the training set*/
    std::vector< std::vector< double >> m_distances;
    /*!Matrix containing the trainig units coordinates, m_n_train x 2*/
    FDAGWR_TRAITS::Dense_Matrix m_coordinates_train;        
    /*!Number of units in the training set*/
    std::size_t m_n_train;
    /*!Matrix containing the to be predicted units coordinates, m_n_pred x 2*/
    FDAGWR_TRAITS::Dense_Matrix m_coordinates_pred;        
    /*!Number of units in the prediction set*/
    std::size_t m_n_pred;

    /*!
    * @brief Evaluation of the Euclidean distance between two units
    * @param loc_i_pred the location in the prediction set (row of coordinates matrix of the to be predicted units)
    * @param loc_j_train the location in the training set (row of coordinates matrix of the training units)
    * @return the pointwise distance within two locations
    * @details a tag dispatcher for the Euclidean distance is used
    */
    double pointwise_distance(std::size_t loc_i_pred, std::size_t loc_j_train, DISTANCE_MEASURE_T<DISTANCE_MEASURE::EUCLIDEAN>) const;

public:
    /*!
    * @brief Default constructor
    */
    distance_matrix_pred() = default;

    /*!
    * @brief Constructor
    * @param coordinates_train the coordinates of the training set units, n_train x 2
    * @param coordinates_pred the coordinates of the prediction set units, n_pred x 2
    */
    template<typename COORDINATES_OBJ>
    distance_matrix_pred(COORDINATES_OBJ &&coordinates_train,
                         COORDINATES_OBJ &&coordinates_pred)
                :
                         m_coordinates_train{std::forward<COORDINATES_OBJ>(coordinates_train)},      //pass the coordinates
                         m_n_train(m_coordinates_train.rows()),
                         m_coordinates_pred{std::forward<COORDINATES_OBJ>(coordinates_pred)},                       //if there are locations
                         m_n_pred(m_coordinates_pred.rows())
                         {
                            assert((m_coordinates_train.cols() == 2) && (m_coordinates_pred.cols() == 2));
                         }
    
    /*!
    * @brief Evaluation of the distance between two statistical units
    * @param loc_i_pred the location in the prediction set (row of coordinates matrix of the to be predicted units)
    * @param loc_j_train the location in the training set (row of coordinates matrix of the training units)
    * @return the pointwise distance within two locations
    * @details a tag dispatcher for the desired distance computation is used
    */
    double pointwise_distance(std::size_t loc_i_pred, std::size_t loc_j_train) const { return pointwise_distance(loc_i_pred,loc_j_train,DISTANCE_MEASURE_T<distance_measure>{});};

    /*!
    * @brief Function that computes the distances within the prediction set units and all the one in the training set
    */
    void compute_distances();

    /*!
    * @brief Getter for the distance matrix
    * @return the private m_distances
    */
    std::vector<std::vector<double>> distances() const {return m_distances;}

    /*!
    * @brief Getter for the number of statistical units in the training set
    * @return the private m_n_train
    */
    std::size_t n_train() const{ return m_n_train;}

    /*!
    * @brief Getter for the number of statistical units in the prediction set
    * @return the private m_n_pred
    */
    std::size_t n_pred() const{ return m_n_pred;}
};

#include "distance_matrix_pred_imp.hpp"

#endif  /*FDAGWR_DISTANCE_MATRIX_PRED_HPP*/