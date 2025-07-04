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


template< DISTANCE_MEASURE distance_measure >
class distance_matrix
{
private:

    /*!*/
    std::vector<double> m_distances;

    /*!
    * the number of locations
    */
    std::size_t m_number_locations;

    /*!
    * the number of distances to be computed
    */
    std::size_t m_number_dist_comp;

    /*!*/
    fdagwr_traits::Dense_Matrix m_coordinates;

    /*!*/
    bool m_flag_comp_dist;

    /*!*/
    //int m_num_threads;

    /*!
    * @brief Evaluation of distance between two points
    * @param distance distance between two locations
    * @param bandwith kernel bandwith
    * @return the evaluation of the kernel function
    */
    double pointwise_distance(std::size_t loc_i, std::size_t loc_j, DISTANCE_MEASURE_T<DISTANCE_MEASURE::EUCLIDEAN>) const;
    
public:

    //le coordinate sono passate come una matrice del tipo: NUMERO DI UNITà x 2: OGNI RIGA è LA LOCATION DI UN EVENTO
    template<typename COORDINATES_OBJ>
    distance_matrix(COORDINATES_OBJ&& coordinates)
        :   
            m_coordinates{std::forward<COORDINATES_OBJ>(coordinates)},      //pass the coordinates
            m_number_locations(coordinates.rows()),                         //pass the number of locations
            m_flag_comp_dist(m_number_locations > 0)                       //if there are locations
        {       
            assert((void("Coordinates matrix has to have 2 columns"), coordinates.cols() == 2));
            if (m_flag_comp_dist)   m_number_dist_comp =  (m_number_locations*(m_number_locations + static_cast<std::size_t>(1)))/static_cast<std::size_t>(2);
            //the number of distances to be computed is m*(m+1)/2
            std::cout << m_coordinates << std::endl;
        }


    /*!
    * @brief Evaluation of kernel function for the non-stationary weights. Tag-dispacther.
    * @param distance distance between two locations
    * @param bandwith kernel bandwith
    * @return the evaluation of the kernel function
    */
    double pointwise_distance(double distance, double bandwith) const { return pointwise_distance(distance,bandwith,DISTANCE_MEASURE_T<distance_measure>{});};


    void compute_distances();

    std::vector<double> distances() const {return m_distances;}
    std::size_t number_dist_comp() const {m_number_dist_comp};

    inline fdagwr_traits::Dense_Matrix distances_view() const
    {
        fdagwr_traits::Dense_Matrix distances_symm(m_number_locations,m_number_locations);

        for (std::size_t i = 0; i < m_number_locations; ++i)
        {
            for (std::size_t j = 0; j < m_number_locations; ++j)
            {
                auto elem = m_distances[(i*(1+i))/2 + j];

                if (i == j)
                {
                    distances_symm(i,i) = elem;
                }
                else
                {
                    distances_symm(i,j)=elem;
                    distances_symm(j,i) = elem;
                }
                
            }
            
        }
        
        return distances_symm;
    };

};


#include "distance_matrix_imp.hpp"

#endif  /*FDAGWR_DISTANCE_MATRIX_HPP*/