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

#ifndef __FDAPDE_TRIANGLE_H__
#define __FDAPDE_TRIANGLE_H__

#include "header_check.h"

namespace fdapde {

// an element is a geometric object bounded to a mesh. It carries both geometrical and connectivity informations
template <typename Triangulation> class Triangle : public Simplex<Triangulation::local_dim, Triangulation::embed_dim> {
    fdapde_static_assert(Triangulation::local_dim == 2, THIS_CLASS_IS_FOR_TRIANGULAR_MESHES_ONLY);
    using Base = Simplex<Triangulation::local_dim, Triangulation::embed_dim>;
   public:
    // constructor
    Triangle() = default;
    Triangle(int id, const Triangulation* mesh) : id_(id), mesh_(mesh), boundary_(false) {
        int b_matches_ = 0;   // element is on boundary if has at least (n_nodes - 1) nodes on boundary
        for (int j = 0; j < this->n_nodes; ++j) {
            this->coords_.col(j) = mesh_->node(mesh_->cells()(id_, j));
	    if (mesh_->is_node_on_boundary(mesh_->cells()(id_, j))) b_matches_++;
        }
	if (b_matches_ >= this->n_nodes - 1) boundary_ = true;
	this->initialize();
    }
    // a triangulation-aware view of a triangle edge
    class EdgeType : public Simplex<Triangulation::local_dim, Triangulation::embed_dim>::BoundaryCellType {
       private:
        int edge_id_;
        const Triangulation* mesh_;
       public:
        using CoordsType = Eigen::Matrix<double, Triangulation::embed_dim, Triangulation::local_dim>;
        EdgeType() = default;
        EdgeType(int edge_id, const Triangulation* mesh) : edge_id_(edge_id), mesh_(mesh) {
            for (int i = 0; i < this->n_nodes; ++i) { this->coords_.col(i) = mesh_->node(mesh_->edges()(edge_id_, i)); }
            this->initialize();
        }
        bool on_boundary() const { return mesh_->is_edge_on_boundary(edge_id_); }
        Eigen::Matrix<int, Dynamic, 1> node_ids() const { return mesh_->edges().row(edge_id_); }
        int id() const { return edge_id_; }
        Eigen::Matrix<int, Dynamic, 1> adjacent_cells() const { return mesh_->edge_to_cells().row(edge_id_); }
        int marker() const {   // mesh edge's marker
            return std::cmp_greater(mesh_->edges_markers().size(), edge_id_) ? mesh_->edges_markers()[edge_id_] :
                                                                               Unmarked;
        }
    };

    // getters
    int id() const { return id_; }
    Eigen::Matrix<int, Dynamic, 1> neighbors() const { return mesh_->neighbors().row(id_); }
    Eigen::Matrix<int, Dynamic, 1> node_ids() const { return mesh_->cells().row(id_); }
    Eigen::Matrix<int, Dynamic, 1> edge_ids() const { return mesh_->cell_to_edges().row(id_); }
    bool on_boundary() const { return boundary_; }
    operator bool() const { return mesh_ != nullptr; }
    EdgeType edge(int n) const {
        fdapde_assert(n < this->n_edges);
        return EdgeType(mesh_->cell_to_edges()(id_, n), mesh_);
    }
    // cell marker
    int marker() const { return mesh_->cells_markers().size() > id_ ? mesh_->cells_markers()[id_] : Unmarked; }
    // given the global id of an edge, returns its local numbering on this cell, or -1 if the edge is not part of it
    int local_edge_id(int edge_id) const {
        int local_id = -1;
        for (int i = 0; i < Base::n_edges; ++i) {
            if (mesh_->cell_to_edges()(id_, i) == edge_id) {
                local_id = i;
                break;
            }
        }
        return local_id;
    }
    int local_facet_id(int edge_id) const { return local_edge_id(edge_id); }

    // iterator over triangle edge
    class edge_iterator : public internals::index_iterator<edge_iterator, EdgeType> {
        using Base = internals::index_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        friend Base;
        const Triangle* t_;
        // access to i-th triangle edge
        edge_iterator& operator()(int i) {
            Base::val_ = t_->edge(i);
            return *this;
        }
       public:
        edge_iterator(int index, const Triangle* t) : Base(index, 0, t->n_edges), t_(t) {
            if (index_ < t_->n_edges) operator()(index_);
        }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(this->n_edges, this); }
   private:
    int id_ = 0;   // triangle ID in the physical mesh
    const Triangulation* mesh_ = nullptr;
    bool boundary_ = false;   // true if the element has at least one vertex on the boundary
};

}   // namespace fdapde

#endif   // __FDAPDE_TRIANGLE_H__
