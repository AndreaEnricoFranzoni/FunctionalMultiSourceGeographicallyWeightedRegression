#include "traits_fdagwr.hpp"

// do not use this if you have other namespaces loaded
using namespace fdapde;

// evaluates a basis system \{ \phi_1(t), \phi_2(t), ..., \phi_N(t) \} at a set of locations \{ t_1, t_2, ..., t_n \}
template <typename Triangulation_, typename CoordsMatrix_>
    requires(internals::is_eigen_dense_xpr_v<CoordsMatrix_>)
Eigen::SparseMatrix<double> spline_basis_eval(const BsSpace<Triangulation_>& bs_space, CoordsMatrix_&& coords) {
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



int test_fda_PDE(double input_test){


  // build equispaced knots vector
  double a = 0;
  double b = 2;
  int n_nodes = 10;
  // divides [a, b] using 10 equispaced nodes
  Triangulation<1, 1> interval = Triangulation<1, 1>::Interval(a, b, n_nodes);

  // in the case you have NOT equispaced knots, or knots are provided externally, use the following constructor
  // Eigen::Matrix<double, Dynamic, 1> nodes; // fill this vector with your specific knots
  // Triangulation<1, 1> interval(nodes);
  
  // build BSpline basis
  int order = 3; // cubic splines:          NUMERO DI BASI E' ORDER + KNOTS (n_nodes) - 1
  BsSpace Vh(interval, order);          

  // evaluate basis at set of locations
  int n_locs = 10;
  Eigen::Matrix<double, Dynamic, Dynamic> locs(n_locs + 1, 1);
  for(int i = 0; i <= n_locs; ++i) { locs(i, 0) = (b - a)/n_locs * i; }

  Eigen::SparseMatrix<double> Psi = spline_basis_eval(Vh, locs);

  std::cout << "basis evaluation at location" << std::endl;
  std::cout << Eigen::Matrix<double, Dynamic, Dynamic>(Psi) << std::endl;   // cast to dense matrix just for printing

  // integration
  TrialFunction u(Vh);
  TestFunction  v(Vh);

  // mass matrix
  auto mass = integral(interval)(u * v);
  Eigen::SparseMatrix<double> M = mass.assemble();

  std::cout << "\n\nmass matrix:  [M]_{ij} = int_I (psi_i * psi_j)" << std::endl;
  std::cout << Eigen::Matrix<double, Dynamic, Dynamic>(M) << std::endl;

  // stiff matrix
  auto stiff = integral(interval)(dxx(u) * dxx(v));
  Eigen::SparseMatrix<double> A = stiff.assemble();

  std::cout << "\n\nstiff matrix: [A]_{ij} = int_I (dxx(psi_i) * dxx(psi_j))" << std::endl;
  std::cout << Eigen::Matrix<double, Dynamic, Dynamic>(A) << std::endl;

  return 0;

}




void testing_function(const std::vector<double> & fd_points,
                      const std::vector<int> & basis_order,
                      const std::vector<double> & knots){

    std::cout << "Nella testing_function" << std::endl;
    basis_systems< FDAGWR_FEATS::FDAGWR_DOMAIN, BASIS_TYPE::BSPLINES > bs(knots,basis_order,3);

    std::cout << "creato basis_systems" << std::endl;
    

    int n_locs = fd_points.size();
    Eigen::Matrix<double, Dynamic, Dynamic> locs(n_locs, 1);
    for(int i = 0; i < n_locs; ++i) { locs(i, 0) = fd_points[i]; }


    for(std::size_t i = 0; i < bs.q(); ++i){
        Eigen::SparseMatrix<double> Psi = spline_basis_eval(bs.systems_of_basis()[i], locs);

        std::cout << i+1 << "basis evaluation at location" << std::endl;
        std::cout << Eigen::Matrix<double, Dynamic, Dynamic>(Psi) << std::endl; 
    }
}