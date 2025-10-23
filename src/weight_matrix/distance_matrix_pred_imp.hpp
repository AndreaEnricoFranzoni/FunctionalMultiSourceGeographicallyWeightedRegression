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


#include "distance_matrix_pred.hpp"



/*!
* @file distance_matrix_pred_imp.hpp
* @brief Implementation of the class for computing the distances within the statistical units of the fitted model and the one to be predicted
* @author Andrea Enrico Franzoni
*/


/*!
* @brief Evaluation of the Euclidean distance between two units
* @param loc_i_pred the location in the prediction set (row of coordinates matrix of the to be predicted units)
* @param loc_j_train the location in the training set (row of coordinates matrix of the training units)
* @return the pointwise distance within two locations
* @details a tag dispatcher for the Euclidean distance is used
*/
template< DISTANCE_MEASURE distance_measure >
double
distance_matrix_pred<distance_measure>::pointwise_distance(std::size_t loc_i_pred, 
                                                           std::size_t loc_j_train, 
                                                           DISTANCE_MEASURE_T<DISTANCE_MEASURE::EUCLIDEAN>)
const
{
    // given coordinates of unit loc_i and loc_j, doing, for each coordinate, difference and squared.
    // Square root of the sum of the previous quantities
    return std::sqrt((m_coordinates_pred.row(loc_i_pred).array() - m_coordinates_train.row(loc_j_train).array()).square().sum());
}


/*!
* @brief Function that computes the distances within the prediction set units and all the one in the training set
*/
template< DISTANCE_MEASURE distance_measure >
void
distance_matrix_pred<distance_measure>::compute_distances()
{
    m_distances.reserve(m_n_pred);

    for(std::size_t i_pred = 0; i_pred < m_coordinates_pred.rows(); ++i_pred){

        //calcolare tutte le distance tra la loc i_pred-th del pred set
        std::vector<double> distances_with_loc_i_pred;
        distances_with_loc_i_pred.reserve(m_n_train);
        for(std::size_t j_train = 0; j_train < m_n_train; ++j_train){
            distances_with_loc_i_pred.push_back(this->pointwise_distance(i_pred,j_train));}

        m_distances.push_back(distances_with_loc_i_pred);
    }
}