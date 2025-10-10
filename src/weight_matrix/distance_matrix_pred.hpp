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

#ifndef FDAGWR_DISTANCE_MATRIX_PRED_HPP
#define FDAGWR_DISTANCE_MATRIX_PRED_HPP


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"
#include <cassert>



/*!
* Doing tag dispatching for the distance measure
* @tparam err_eval: template parameter for the error evaluation strategy
*/
template <DISTANCE_MEASURE distance_measure>
using DISTANCE_MEASURE_T = std::integral_constant<DISTANCE_MEASURE, distance_measure>;




template< DISTANCE_MEASURE distance_measure >
class distance_matrix_pred
{
private:

    //ogni vettore esterno rappresenta un'unità su cui fare pred: ogni elemento dell'interno sono distanze tra train e pred
    std::vector< std::vector< double >> m_distances;

    FDAGWR_TRAITS::Dense_Matrix m_coordinates_train;        //matrice numero unità nel train set x 2

    std::size_t m_n_train;

    FDAGWR_TRAITS::Dense_Matrix m_coordinates_pred;         //matrice numero unità nel pred set x 2

    std::size_t m_n_pred;

    /*!
    * @brief Evaluation of the Euclidean distance between two statistical units
    * @param loc_i the first location (row of coordinates matrix)
    * @param loc_j the second location (row of coordinates matrix)
    * @return the pointwise distance within two locations
    * @details a tag dispatcher for the Euclidean distance is used
    */
    double pointwise_distance(std::size_t loc_i_pred, std::size_t loc_j_train, DISTANCE_MEASURE_T<DISTANCE_MEASURE::EUCLIDEAN>) const;



public:
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
    * @param loc_i the index of the first location
    * @param loc_j the index of the second location
    * @return the pointwise distance within two locations
    * @details a tag dispatcher for the desired distance computation is used
    */
    double pointwise_distance(std::size_t loc_i_pred, std::size_t loc_j_train) const { return pointwise_distance(loc_i_pred,loc_j_train,DISTANCE_MEASURE_T<distance_measure>{});};

    /*!
    * Function that computes the distances within each pair of statistical units
    */
    void compute_distances();

    /*!
    * @brief Getter for the distance matrix
    * @return the private m_distances
    */
    std::vector<std::vector<double>> distances() const {return m_distances;}
};

#include "distance_matrix_pred_imp.hpp"

#endif  /*FDAGWR_DISTANCE_MATRIX_PRED_HPP*/