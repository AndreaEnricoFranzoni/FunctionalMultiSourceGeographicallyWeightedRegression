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


#include "distance_matrix.hpp"

/*!
* @file distance_matrix_imp.hpp
* @brief Implementation of distance matrix class
* @author Andrea Enrico Franzoni
*/


/*!
* @brief Euaclidean distance within two statistical units
* @details 'DISTANCE_MEASURE::EUCLIDEAN' dispatch
*/
template< DISTANCE_MEASURE distance_measure >
double
distance_matrix<distance_measure>::pointwise_distance(std::size_t loc_i, 
                                                      std::size_t loc_j, 
                                                      DISTANCE_MEASURE_T<DISTANCE_MEASURE::EUCLIDEAN>)
const
{
    // given coordinates of unit loc_i and loc_j, doing, for each coordinate, difference and squared.
    // Square root of the sum of the previous quantities
    return std::sqrt((m_coordinates.row(loc_i).array() - m_coordinates.row(loc_j).array()).square().sum());
}


/*!
* @brief Compute the distance matrix within the different locations, modifying the private member m_distances
* @tparam distance_measure indicates which distance is used in the computations
*/
template< DISTANCE_MEASURE distance_measure >
void
distance_matrix<distance_measure>::compute_distances()
{

    /*
    Distances are stored column-wise, up until the diagonal. Logic of access is the following
    double& get(int i, int j) {
        if (i < j) std::swap(i, j);
        return m_distances[i * (i + 1) / 2 + j];}
    */

   //prearing the container for storing
    m_distances.resize(m_number_dist_comp);

#ifdef _OPENMP
#pragma omp parallel for shared(m_number_locations) num_threads(m_num_threads)
#endif
    for(std::size_t j = 0; j < m_number_locations; ++j){
        for (std::size_t i = 0; i <= j; ++i){    
            
            //the index for the new element in the storing vector
            std::size_t k = i>=j ? (i*(i+1))/2 + j : (j*(j+1))/2 + i;
            
            m_distances[k]=this->pointwise_distance(i,j);}}


    
    /*
    //Version for non parallel
    m_distances.reserve(m_number_dist_comp);

    for(std::size_t j = 0; j < m_number_locations; ++j){
        for (std::size_t i = 0; i <= j; ++i){    

            m_distances.push_back(this->pointwise_distance(i,j));}}
    */


}