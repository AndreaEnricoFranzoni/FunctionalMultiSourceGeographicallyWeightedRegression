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


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"
#include <cassert>


#ifdef _OPENMP
#include <omp.h>
#endif

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
    FDAGWR_TRAITS::Dense_Matrix m_coordinates;

    /*!
    * Flag that tracks if at least two statistical units are passed
    */
    bool m_flag_comp_dist;

    /*!
    * Number of threads for parallelization
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
    
    /*!
    * @brief Return the partial sum of integers up until up_until
    * @param up_until the last integer up until the partial sum is performed
    * @return the partial sum
    */
    static std::size_t partial_sum(std::size_t up_until){
        //partial sum
        std::size_t result(0);
        for (std::size_t i = 0; i <= up_until; ++i){    result += i;}
        return result;}


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
            m_number_locations(coordinates.rows()),                         //pass the number of statistical units
            m_coordinates{std::forward<COORDINATES_OBJ>(coordinates)},      //pass the coordinates
            m_flag_comp_dist(m_number_locations > 0),                       //if there are locations
            m_num_threads(number_threads)                                   //number of threads for paralelization
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
    double pointwise_distance(std::size_t loc_i, std::size_t loc_j) const { return pointwise_distance(loc_i,loc_j,DISTANCE_MEASURE_T<distance_measure>{});};

    /*!
    * Function that computes the distances within each pair of statistical units
    */
    void compute_distances();

    /*!
    * @brief Getter for the distance matrix
    * @return the private m_distances
    */
    std::vector<double> distances() const {return m_distances;}

    /*!
    * @brief Get element A(i,j). Const version, read-only.
    * @param i: number of row
    * @param j: number of col
    * @return the value in position (i,j) 
    */
    inline double operator()(std::size_t i, std::size_t j) const
    {    
        if (i < j) std::swap(i, j);
        return m_distances[i*(i+1)/2 + j];
    }

    /*!
    * @brief Return the col_i-th column (all the distances with respect to unit i-th)
    * @param col_i: the column desired. The first unit correspond to column 0
    * @return column col_i-th (the distances with respect to unit i-th), in an Eigen::VectorXd
    * @note Elements are stored column-wise
    */
   inline FDAGWR_TRAITS::Dense_Vector operator[](std::size_t col_i) const
   {    
        //cheack the correct dimension of the coordinates matrix
        assert((void("The column index has to be in {0,1,...,number-of-statistical-units - 1}"), 
               0<=col_i && col_i < m_number_locations));

        //container for the column
        FDAGWR_TRAITS::Dense_Vector column(m_number_locations);
        
        //vector to store the indeces of the elements of m_distances that refers to a distance computed with respect to the i-th statistical unit
        std::vector<std::size_t> access_indeces;
        //there is a total of number-of-statistical-units distances computed for each statistical unit
        access_indeces.reserve(m_number_locations);     

        //the first index is the index of start of the column (elem of the column in the first row)
        access_indeces.emplace_back(distance_matrix::partial_sum(col_i));  

        //then, the following indeces are the colomn_number integers following the first index (elems of the column up until diagonal)
        for (size_t i = 1; i <= col_i; ++i){    access_indeces.emplace_back(access_indeces.front() + i);}
        
        //last set of indeces: elements of row col_i from the diagonal up until the end of the row
        //(index of the previous element plus the column index of the next one in the same row)
        for (std::size_t i = col_i + 1; i < m_number_locations; ++i){   access_indeces.emplace_back(access_indeces.back()+i);}
        
        //filling the column
#ifdef _OPENMP
#pragma omp parallel for shared(m_number_locations,access_indeces) num_threads(m_num_threads)
#endif
        for(std::size_t i = 0; i < m_number_locations; ++i){    column(i) = m_distances[access_indeces[i]];}

        return column;
   }

};


#include "distance_matrix_imp.hpp"

#endif  /*FDAGWR_DISTANCE_MATRIX_HPP*/