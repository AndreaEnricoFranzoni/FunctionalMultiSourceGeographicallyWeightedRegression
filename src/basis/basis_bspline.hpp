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


#ifndef FDAGWR_BSPLINES_BASIS_HPP
#define FDAGWR_BSPLINES_BASIS_HPP


#include "basis.hpp"



// do not use this if you have other namespaces loaded
//using namespace fdapde;

// evaluates a basis system \{ \phi_1(t), \phi_2(t), ..., \phi_N(t) \} at a set of locations \{ t_1, t_2, ..., t_n \}
template <typename Triangulation_, typename CoordsMatrix_>
    //requires(internals::is_eigen_dense_xpr_v<CoordsMatrix_>)
    requires(fdagwr_concepts::as_interval<Triangulation_> && fdapde::internals::is_eigen_dense_xpr_v<CoordsMatrix_>)
inline 
Eigen::SparseMatrix<double> 
bsplines_basis_evaluation(const fdapde::BsSpace<Triangulation_>& bs_space, 
                          CoordsMatrix_&& coords) 
{
    static constexpr int embed_dim = Triangulation_::embed_dim;
    fdapde::fdapde_assert(coords.rows() > 0 && coords.cols() == embed_dim);

    int n_shape_functions = bs_space.n_shape_functions();
    int n_dofs = bs_space.n_dofs();
    int n_locs = coords.rows();
    Eigen::SparseMatrix<double> psi_(n_locs, n_dofs);    
    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(n_locs * n_shape_functions);

    Eigen::Matrix<int, Dynamic, 1> cell_id = bs_space.triangulation().locate(coords);
    const auto& dof_handler = bs_space.dof_handler();
    // build basis evaluation matrix
    for (int i = 0; i < n_locs; ++i) {
        if (cell_id[i] != -1) {   // point falls inside domain
            Eigen::Matrix<double, embed_dim, 1> p_i(coords.row(i));
            auto cell = dof_handler.cell(cell_id[i]);
            // update matrix
            for (std::size_t h = 0; h < cell.dofs().size(); ++h) {
                int active_dof = cell.dofs()[h];
                triplet_list.emplace_back(i, active_dof, bs_space.eval_cell_value(active_dof, p_i));   // \psi_j(p_i)
            }
        }
    }
    // finalize construction
    psi_.setFromTriplets(triplet_list.begin(), triplet_list.end());
    psi_.makeCompressed();
    return psi_;
}





/*!
* @brief class for bsplines basis
*/
template< typename domain_type = FDAGWR_TRAITS::basis_geometry >
    requires fdagwr_concepts::as_interval<domain_type>
class bsplines_basis :  public basis_base_class<domain_type>
{

/*!
* @brief Alias for the basis space
* @note calling the constructor of BsSpace in the constructor of the class
*/
using BasisSpace = fdapde::BsSpace<domain_type>;

private:
    /*!Bslines*/
    BasisSpace m_basis;

public:
    /*!Default(cubic bsplines)*/
    static constexpr std::size_t bsplines_degree_default = static_cast<std::size_t>(3); 

    /*!Constructor*/
    bsplines_basis(const FDAGWR_TRAITS::Dense_Vector & knots,
                   std::size_t degree,
                   std::size_t number_of_basis)    
            :   
                basis_base_class<domain_type>(knots,degree,number_of_basis),
                m_basis(this->knots(),this->degree())
                {
                    //cheack input consistency
                    assert((void("Number of knots = number of basis - degree + 1"), this->number_knots() == (m_number_of_basis - m_degree + static_cast<std::size_t>(1))));
                }
    
    //Move and copy constructor 
    bsplines_basis(const bsplines_basis&) = default;
    bsplines_basis(bsplines_basis&&) noexcept = default;
    bsplines_basis& operator=(const bsplines_basis&) = default;
    bsplines_basis& operator=(bsplines_basis&&) noexcept = default;
                
    /*!
    * @brief Getter for the basis
    * @return the private m_basis
    */
    const BasisSpace& basis() const {return m_basis;}

    /*!
    * @brief Giving the basis type
    * @return std::string
    */
    inline
    std::string 
    type()
    const 
    override
    {
        return "bsplines";
    }

    /*!
    * @brief evaluating the system of basis basis_i-th in location location. Overriding the method
    */
    inline 
    FDAGWR_TRAITS::Dense_Matrix 
    eval_base(const double &location) 
    const
    override
    {
        //wrap the input into a coherent object for the spline evaluation
        FDAGWR_TRAITS::Dense_Matrix loc = FDAGWR_TRAITS::Dense_Matrix::Constant(1, 1, location);
        //wrap the output into a dense matrix:      HA UNA RIGA, N_BASIS COLONNE
        return FDAGWR_TRAITS::Dense_Matrix(bsplines_basis_evaluation<domain_type>(m_basis, loc));
    }

    /*!
    * @brief evaluating the basis basis_i over a set of locations. Overriding the method
    * @note locations è una FDAGWR_TRAITS::Dense_Matrix of dimensions n_locs x 1
    */
    inline 
    FDAGWR_TRAITS::Sparse_Matrix 
    eval_base_on_locs(const FDAGWR_TRAITS::Dense_Matrix &locations) 
    const
    override
    {
        //ritorna una matrice n_locs x n_basis
        return bsplines_basis_evaluation<domain_type>(m_basis, locations);
    }

    /*!
    * @brief Performing the smoothing of fdata, discrete evaluations, relatively to the abscissa in knots, given by f_ev
    * @param f_ev an n_locs x 1 matrix with the evaluations of the fdata in correspondence of the knots
    * @param knots knots over which evaluating the basis, and for which it is available the evaluation of the functional datum
    * @return a dense matrix of dimension n_basis x 1, with the coefficient of the basis expansion
    */
    inline
    FDAGWR_TRAITS::Dense_Matrix
    smoothing(const FDAGWR_TRAITS::Dense_Matrix & f_ev, 
              const FDAGWR_TRAITS::Dense_Matrix & knots) 
    const
    override
    {
        assert((f_ev.rows() == knots.rows()) && (f_ev.cols() == 1) && (knots.cols() == 1));

        //psi: an knots.size() x number of basis matrix: each row represents a knot, every column a basis: contains its evaluation
        Eigen::SparseMatrix<double> psi = this->eval_base_on_locs(knots);
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        //performs (t(Psi)*Psi)^(-1) * t(Psi)
        solver.compute(psi);

        return solver.solve(f_ev);
    }
};

#endif  /*FDAGWR_BSPLINES_BASIS_HPP*/