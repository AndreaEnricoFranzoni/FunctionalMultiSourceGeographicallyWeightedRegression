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


#ifndef FDAGWR_PENALIZATION_MATRIX_HPP
#define FDAGWR_PENALIZATION_MATRIX_HPP


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include "basis_bspline_systems.hpp"
#include "penalty_matrix_penalties_comp.hpp"

/*!
* @file penalization_matrix.hpp
* @brief Contains the class to define the penalization matrix used in the fmsgwr algoroithm
* @author Andrea Enrico Franzoni
*/

template <PENALIZED_DERIVATIVE der_pen>
using PenaltyOrderDerivativeType = std::conditional<der_pen == PENALIZED_DERIVATIVE::SECOND,
                                                    SecondDerivativePenalty,        
                                                    typename std::conditional<der_pen == PENALIZED_DERIVATIVE::FIRST,
                                                                     FirstDerivativePenalty,
                                                                     ZeroDerivativePenalty>::type>::type;

template< PENALIZED_DERIVATIVE der_pen = PENALIZED_DERIVATIVE::SECOND >
class penalization_matrix
{

//order of the penalization for the policy to compute the penalization itself
using PenaltyPolicy = PenaltyOrderDerivativeType<der_pen>;

private:
    /*!Penalization matrix*/
    FDAGWR_TRAITS::Sparse_Matrix m_PenalizationMatrix;

    /*!Number of functional covariates described by a basis expansion (total number of blocks in the penalization matrix)*/
    std::size_t m_q;

    /*!Number of basis for each covariate*/
    std::vector<std::size_t> m_Lj;

    /*!Number of total basis (penalization matrix is a m_L x m_L)*/;
    std::size_t m_L;


public:
    /*!
    * @brief Constructor: PER AVERE LE PENALIZZAZIONI CON LA DERIVATA SECONDA SERVE UN ORDINE DELLE BASI >= 2
    * @param bs is an object storing systems of basis
    * @param lambdas is an std::vector<double> storing the penalization for each regressor
    * @note  PENALIZATION COMPUTATION IS IMPLEMENTED ONLY FOR 1D DOMAINS AND BSPLINES BASIS
    */
    template< typename BASIS_SPACE >
    penalization_matrix(BASIS_SPACE&& bs,
                        const std::vector<double>& lambdas)
        :   
        m_Lj(bs.numbers_of_basis()),
        m_L(std::reduce(bs.numbers_of_basis().cbegin(),bs.numbers_of_basis().cend(),static_cast<std::size_t>(0))),
        m_q(bs.q())
            {   
                //storing the penalty for each covariate in an Eigen::Triplet
                std::vector<Eigen::Triplet<double>> penalty_matrices_triplets;
                //the unlikely scenario in which all the L2 scalar products are not-null
                penalty_matrices_triplets.reserve(std::transform_reduce(m_Lj.cbegin(),
                                                                        m_Lj.cend(),
                                                                        static_cast<std::size_t>(0),
                                                                        std::plus{},
                                                                        [](const double &nb){return std::pow(nb,2);}));

                //constructing the penalty matrices
                for(std::size_t i = 0; i < m_q; ++i){

                    //penalties, for each basis system: PenaltyPolicy indicates the order
                    penalty_computation<PenaltyPolicy> penalty_comp;
                    FDAGWR_TRAITS::Sparse_Matrix PenaltyBasis_i = penalty_comp(bs,i);
                    PenaltyBasis_i *= lambdas[i];

                    //all the penalty matrix are squared matrices: therse are the index at which each block starts
                    std::size_t start_of_block = std::reduce(bs.numbers_of_basis().cbegin(),bs.numbers_of_basis().cbegin()+i,static_cast<std::size_t>(0));
                    //storing the matrix in the a vector of Eigen::Triplets
                    for (std::size_t k = 0; k < PenaltyBasis_i.outerSize(); ++k){
                        for (FDAGWR_TRAITS::Sparse_Matrix::InnerIterator it(PenaltyBasis_i,k); it; ++it){
                            penalty_matrices_triplets.emplace_back(it.row() + start_of_block, 
                                                                   it.col() + start_of_block,
                                                                   it.value());}}}

                //shrinking
                penalty_matrices_triplets.shrink_to_fit();

                //size for the penalization matrix
                m_PenalizationMatrix.resize(m_L,m_L);

                //constructing the penalization matrix as a sparse block matrix
                m_PenalizationMatrix.setFromTriplets(penalty_matrices_triplets.begin(),penalty_matrices_triplets.end());
            }
    
    /*!
    * @brief Getter for the penalization matrix
    */
    const FDAGWR_TRAITS::Sparse_Matrix& PenalizationMatrix() const {return m_PenalizationMatrix;}

    /*!
    * @brief Getter for the dimension of the penalization matrix
    */
   std::size_t L() const {return m_L;}
};

#endif  /*FDAGWR_PENALIZATION_MATRIX_HPP*/