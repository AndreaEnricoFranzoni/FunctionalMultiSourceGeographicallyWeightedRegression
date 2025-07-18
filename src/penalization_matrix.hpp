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


#include "traits_fdagwr.hpp"
#include "basis_systems.hpp"

/*!
* @file penalization_matrix.hpp
* @brief Contains the class to define the penalization matrix used in the fmsgwr algoroithm
* @author Andrea Enrico Franzoni
*/

template< PENALIZED_DERIVATIVE der_pen = PENALIZED_DERIVATIVE::SECOND >
class penalization_matrix
{

private:
    /*!Penalization matrix*/
    fdagwr_traits::Sparse_Matrix m_PenalizationMatrix;

    /*!Number of functional covariates described by a basis expansion (total number of blocks in the penalization matrix)*/
    std::size_t m_q;

    /*!Number of basis for each covariate*/
    std::vector<std::size_t> m_Lj;

    /*!Number of total basis*/;
    std::size_t m_L;

    /*!
    PER AVERE LE PENALIZZAZIONI CON LA DERIVATA SECONDA SERVE UN ORDINE DELLE BASI >= 2
    */
public:
    penalization_matrix(const basis_systems< fdagwr_traits::Domain, BASIS_TYPE::BSPLINES > & bs,
                        const std::vector<double>& lambdas)
        :   
        //m_Lj(bs.number_of_basis()),
        //m_L(std::reduce(bs.number_of_basis().cbegin(),bs.number_of_basis().cend(),static_cast<std::size_t>(0))),
        m_q(bs.q())
        //m_PenalizationMatrix(m_L,m_L)       //initializing the penalization matrix
            {   
                m_Lj.reserve(bs.number_of_basis().size());
                std::copy(bs.number_of_basis().cbegin(),bs.number_of_basis().cend(),std::back_inserter(m_Lj));
                std::cout << "Tot basis= " << std::reduce(bs.number_of_basis().cbegin(),bs.number_of_basis().cend(),static_cast<std::size_t>(0)) << std::endl;
                m_L = std::reduce(bs.number_of_basis().cbegin(),bs.number_of_basis().cend(),static_cast<std::size_t>(0));
                //storing the penalty for each covariate in an Eigen::Triplet
                std::vector<Eigen::Triplet<double>> stiff_matrices_triplets;
                stiff_matrices_triplets.reserve(std::transform_reduce(m_Lj.cbegin(),
                                                                      m_Lj.cend(),
                                                                      static_cast<std::size_t>(0),
                                                                      std::plus{},
                                                                      [](const double &nb){return std::pow(nb,2);}));


                std::cout << "Triplets size: " << stiff_matrices_triplets.size() << ", triplets capacity: " << stiff_matrices_triplets.capacity() << std::endl;
                std::cout << "m_L" << m_L << std::endl;
                std::cout << "basi divise in" << std::endl;
                for (std::size_t i = 0; i < m_Lj.size(); ++i)
                {
                    std::cout << "Cov " << i+1 << m_Lj[i] << std::endl;
                }
                std::cout << "Cov num: " << m_q << std::endl;
                



                //constructing the penalty matrices
                for(std::size_t i = 0; i < m_q; ++i){
                    // integration
                    fdapde::TrialFunction u(bs.systems_of_basis()[i]); 
                    fdapde::TestFunction  v(bs.systems_of_basis()[i]);
                    // stiff matrix: penalizing the second derivaive
                    auto stiff = integral(bs.interval())(dxx(u) * dxx(v));
                    Eigen::SparseMatrix<double> M = stiff.assemble();
                    std::cout << "Pen " << i+1 << " pre" << std::endl;
                    std::cout << Eigen::MatrixXd(M) << std::endl;
                    //penalties, for each basis system
                    M *= lambdas[i];
                    std::cout << "Pen " << i+1 << " pro, with lambda= " << lambdas[i] << std::endl;
                    std::cout << Eigen::MatrixXd(M) << std::endl;

                    //all the penalty matrix are squared matrices: therse are the index at which each block starts
                    std::size_t start_of_block = std::reduce(bs.number_of_basis().cbegin(),bs.number_of_basis().cbegin()+i,static_cast<std::size_t>(0));
                    std::cout << "Start of block " << i+1 << ": " << start_of_block << std::endl;
                    //storing the matrix in the a vector of Eigen::Triplets
                    for (std::size_t k = 0; k < M.outerSize(); ++k){
                        for (fdagwr_traits::Sparse_Matrix::InnerIterator it(M,k); it; ++it){
                            stiff_matrices_triplets.emplace_back(it.row() + start_of_block, 
                                                                 it.col() + start_of_block,
                                                                 it.value());}}}
                
                std::cout << "Starting init the penalization matrix" << std::endl;
                m_PenalizationMatrix.resize(m_L,m_L);
                m_PenalizationMatrix(0,0) = 1;
                //constructing the penalization matrix as a sparse block matrix
                //m_PenalizationMatrix.setFromTriplets(stiff_matrices_triplets.begin(),stiff_matrices_triplets.end());
            }
    
    /*!
    * @brief Getter for the penalization matrix
    */
    const fdagwr_traits::Sparse_Matrix& PenalizationMatrix() const {return m_PenalizationMatrix;}
};

#endif  /*FDAGWR_PENALIZATION_MATRIX_HPP*/