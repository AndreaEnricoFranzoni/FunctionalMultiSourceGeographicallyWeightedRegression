// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_WALK_SEARCH_H__
#define __FDAPDE_WALK_SEARCH_H__

#include "header_check.h"

namespace fdapde {

// barycentric walk strategy for point location problem, works only for 2D and 3D *convex* triangualtions
template <typename MeshType> class WalkSearch {
   private:
    static constexpr int embed_dim = MeshType::embed_dim;
    static constexpr int local_dim = MeshType::local_dim;
    const MeshType* mesh_;
   public:
    WalkSearch() = default;
    WalkSearch(const MeshType* mesh) : mesh_(mesh) {
        static_assert(MeshType::local_dim == MeshType::embed_dim);
    };
    // finds element containing p, returns nullptr if element not found
    int locate(const Eigen::Matrix<double, embed_dim, 1>& p) const {
        // start search from random element
        std::random_device rng {};
        std::uniform_int_distribution<std::size_t> uniform_int(0, mesh_->n_cells() - 1);
	std::size_t next = uniform_int(rng);

        std::unordered_set<std::size_t> visited_;
        while (!mesh_->cell(next).contains(p) || visited_.find(next) != visited_.end()) {
            visited_.insert(next);
            // compute barycantric coordinates
            Eigen::Matrix<double, embed_dim + 1, 1> bary_coord = mesh_->cell(next).barycentric_coords(p);
            // find minimum baricentric coordinate and move to element insisting of opposite face
            std::size_t min_bary_coord_index;
            bary_coord.minCoeff(&min_bary_coord_index);
            // the i-th value in neighbors refers to the element adjacent to the face oppsite the i-th vertex
            next = mesh_->cell(next).neighbors()[min_bary_coord_index];
        }
        return mesh_->cell(next).contains(p) ? mesh_->cell(next).id() : -1;
    }
    template <typename CoordsMatrix>
        requires(internals::is_eigen_dense_xpr_v<CoordsMatrix>)
    Eigen::Matrix<int, Dynamic, 1> locate(const CoordsMatrix& locs) const {
        fdapde_assert(locs.cols() == embed_dim);
        Eigen::Matrix<int, Dynamic, 1> ids(locs.rows());
        for (int i = 0; i < locs.rows(); ++i) { ids[i] = locate(Eigen::Matrix<double, embed_dim, 1>(locs.row(i))); }
        return ids;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_WALK_SEARCH_H__
