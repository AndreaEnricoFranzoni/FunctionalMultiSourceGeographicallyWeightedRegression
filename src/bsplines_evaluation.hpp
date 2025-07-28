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


#ifndef FDAGWR_BSPLINES_EVAL_HPP
#define FDAGWR_BSPLINES_EVAL_HPP


#include "traits_fdagwr.hpp"
#include "concepts_fdagwr.hpp"

// do not use this if you have other namespaces loaded
using namespace fdapde;

// evaluates a basis system \{ \phi_1(t), \phi_2(t), ..., \phi_N(t) \} at a set of locations \{ t_1, t_2, ..., t_n \}
template <typename Triangulation_, typename CoordsMatrix_>
    //requires(internals::is_eigen_dense_xpr_v<CoordsMatrix_>)
    requires(as_interval<Triangulation_> && internals::is_eigen_dense_xpr_v<CoordsMatrix_>)
inline Eigen::SparseMatrix<double> bsplines_basis_evaluation(const BsSpace<Triangulation_>& bs_space, CoordsMatrix_&& coords) {
    static constexpr int embed_dim = Triangulation_::embed_dim;
    fdapde_assert(coords.rows() > 0 && coords.cols() == embed_dim);

    int n_shape_functions = bs_space.n_shape_functions();
    int n_dofs = bs_space.n_dofs();
    int n_locs = coords.rows();
    Eigen::SparseMatrix<double> psi_(n_locs, n_dofs);    
    std::vector<Triplet<double>> triplet_list;
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

#endif  /*FDAGWR_BSPLINES_EVAL_HPP*/