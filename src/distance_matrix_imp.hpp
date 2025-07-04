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
* @brief Implementation of distance function
* @author Andrea Enrico Franzoni
*/



template< DISTANCE_MEASURE distance_measure >
double
distance_matrix<distance_measure>::pointwise_distance(std::size_t loc_i, std::size_t loc_j, DISTANCE_MEASURE_T<DISTANCE_MEASURE::EUCLIDEAN>)
const
{
    //each row contains the coordinates of a location
    //std::cout << "Row loc_i" << loc_i << std::endl;
    //std::cout << m_coordinates.row(loc_i).array() << std::endl;
    //std::cout << "Row loc_j" << loc_j << std::endl << std::endl;
    //std::cout << m_coordinates.row(loc_j).array() << std::endl; 
    return std::sqrt( (m_coordinates.row(loc_i).array() - m_coordinates.row(loc_j).array()).square().sum() );
}


template< DISTANCE_MEASURE distance_measure >
void
distance_matrix<distance_measure>::compute_distances()
{

    /*
    Salvare le distanza col_major
    */


    //L'IDEA E':
    // HO UNA MATRICE m x 2, dove m indica il numero di unità statistiche disponibili
    // faccio le distanze tra tutte queste righe (infatti l'idea è che mi calcolo le distanze solo tra le medesime distanze)
    // la matrice che ottengo è m x m, MA SIMMETRICA: l'elemento (i,j) indica la distanza tra unità i e unità j, il (j,i) viceversa
    // salvo in un vettore, PER COLONNE, con questa logica qui
    //  double& get(int i, int j) {
    //          if (i < j) std::swap(i, j);
    //          return m_distances[i * (i + 1) / 2 + j];}
    // una colonna alla volta, fino all'elemento sulla diagonale (dall'alto) viene inserito nel vettore sequenzialmente

    
    m_distances.reserve(m_number_dist_comp);

    for(std::size_t j = 0; j < m_number_locations; ++j){
        for (std::size_t i = 0; i <= j; ++i){    
                if (i>=j)
                    {
                        std::cout << "Elem number " << (i*(i+1))/2 + j << std::endl;
                    }   
                else{
                        std::cout << "Elem number " << (j*(j+1))/2 + i << std::endl;
                    }
            m_distances.push_back(this->pointwise_distance(i,j));}}
}