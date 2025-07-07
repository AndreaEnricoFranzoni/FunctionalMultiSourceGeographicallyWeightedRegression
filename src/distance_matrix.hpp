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

#ifndef FDAGWR_DISTANCE_MATRIX_HPP
#define FDAGWR_DISTANCE_MATRIX_HPP

#include "traits_fdagwr.hpp"
#include <cassert>


/*!
* @file distance_matrix.hpp
* @brief Class for computing the distances within the points of interest of GWR model
* @author Andrea Enrico Franzoni
*/

/*!
* Doing tag dispatching for the distance measure
* @tparam err_eval: template parameter for the error evaluation strategy
*/
template <DISTANCE_MEASURE distance_measure>
using DISTANCE_MEASURE_T = std::integral_constant<DISTANCE_MEASURE, distance_measure>;


/*!
* @class distance_matrix
* @brief Template class for constructing the distance matrix: a squared symmetric matrix containing the distance within each pair of units
* @tparam distance_measure how to compute the distances within different units (enumerator)
*/
template< DISTANCE_MEASURE distance_measure >
class distance_matrix
{
private:

    /*!
    * The distance matrix is, for efficiency reasons, stored in vector, 
    * column-wise (first col, second col, third col, ... of the original distance matrix)
    */
    std::vector<double> m_distances;

    /*!
    * The number of statistical units. For each stastical unit, there is a location
    */
    std::size_t m_number_locations;

    /*!
    * The number of distances to be computed (m*(m+1)/2, where m is the number of statistical units)
    */
    std::size_t m_number_dist_comp;

    /*!
    * A number of statistical units x 2 matrix with the (UTM) coordinates of each statistical unit. The class supports only
    * locations on a two dimensional mainfold
    */
    fdagwr_traits::Dense_Matrix m_coordinates;

    /*!
    * Flag that tracks if at least two statistical units are passed
    */
    bool m_flag_comp_dist;

    /*!
    * Number of threads for paralelization
    */
    int m_num_threads;

    /*!
    * @brief Evaluation of the Euclidean distance between two statistical units
    * @param loc_i the first location (row of coordinates matrix)
    * @param loc_j the second location (row of coordinates matrix)
    * @return the pointwise distance within two locations
    * @details a tag dispatcher for the Euclidean distance is used
    */
    double pointwise_distance(std::size_t loc_i, std::size_t loc_j, DISTANCE_MEASURE_T<DISTANCE_MEASURE::EUCLIDEAN>) const;
    

public:

    /*!
    * @brief Constructor for the distance matrix (square symmetric matrix containing the distances within each pair of units).
    *        Locations are intended over a two dimensional domain. Dimensionality check into the constructor
    * @param coordinates coordinates of each statistical unit. It is an Eigen dynamic matrix within as much rows as the number of statistical
    *                    units, and two columns (for each column, one coordinate). The coordinates are intended as UTM coordinates.
    *                    The distance matrix will be a number of statistical units x number of statistical units
    * @details Universal constructor: move semantic used to optimazing handling big size objects
    */
    template<typename COORDINATES_OBJ>
    distance_matrix(COORDINATES_OBJ&& coordinates,
                    int number_threads)
        :   
            m_coordinates{std::forward<COORDINATES_OBJ>(coordinates)},      //pass the coordinates
            m_number_locations(coordinates.rows()),                         //pass the number of statistical units
            m_flag_comp_dist(m_number_locations > 0),                       //if there are locations
            m_number_threads(number_threads)                                //number of threads for paralelization
        {       
            //cheack the correct dimension of the coordinates matrix
            assert((void("Coordinates matrix has to have 2 columns"), coordinates.cols() == 2));
            //the number of distances to be computed is m*(m+1)/2
            if (m_flag_comp_dist)   m_number_dist_comp =  (m_number_locations*(m_number_locations + static_cast<std::size_t>(1)))/static_cast<std::size_t>(2);
        }


    /*!
    * @brief Evaluation of the distance between two statistical units
    * @param loc_i the index of the first location
    * @param loc_j the index of the second location
    * @return the pointwise distance within two locations
    * @details a tag dispatcher for the desired distance computation is used
    */
    double pointwise_distance(double distance, double bandwith) const { return pointwise_distance(distance,bandwith,DISTANCE_MEASURE_T<distance_measure>{});};

    /*!
    * Function that computes the distances within each pair of statistical units
    */
    void compute_distances();

    /*!
    * @brief Getter for the distance matrix
    * @return the private m_distances
    */
    std::vector<double> distances() const {return m_distances;}


    inline fdagwr_traits::Dense_Matrix distances_view() const
    {
        fdagwr_traits::Dense_Matrix distances_symm(m_number_locations,m_number_locations);

        for (std::size_t i = 0; i < m_number_locations; ++i){
            for (std::size_t j = 0; j < m_number_locations; ++j){
                
                if (i==j)
                {
                    std::size_t index_k = (i*(1+i))/2 + j;
                    distances_symm(i,i) = m_distances[index_k];
                }
                if (i<j)
                {
                    std::size_t index_k = (j*(1+j))/2 + i;
                    distances_symm(i,j)=m_distances[index_k];
                    distances_symm(j,i) = m_distances[index_k];
                }
                else
                {
                    std::size_t index_k = (i*(1+i))/2 + j;
                    distances_symm(i,i) = m_distances[index_k];
                    distances_symm(j,i) = m_distances[index_k];
                }
            }
        }
        
        return distances_symm;
    };

};


#include "distance_matrix_imp.hpp"

#endif  /*FDAGWR_DISTANCE_MATRIX_HPP*/