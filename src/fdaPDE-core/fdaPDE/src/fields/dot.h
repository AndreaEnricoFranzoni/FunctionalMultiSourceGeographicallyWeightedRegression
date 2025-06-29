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

#ifndef __FDAPDE_DOT_H__
#define __FDAPDE_DOT_H__

#include "header_check.h"

namespace fdapde {

template <int StaticInputSize, typename Derived> struct MatrixFieldBase;

template <typename Lhs, typename Rhs>
class DotProduct : public ScalarFieldBase<Lhs::StaticInputSize, DotProduct<Lhs, Rhs>> {
    fdapde_static_assert(
      (Lhs::StaticInputSize == Dynamic || Rhs::StaticInputSize == Dynamic ||
       Lhs::StaticInputSize == Rhs::StaticInputSize) &&
        ((internals::is_scalar_field_v<Lhs> && internals::is_scalar_field_v<Rhs>) ||
         (internals::is_matrix_field_v<Lhs> && internals::is_matrix_field_v<Rhs> &&
          ((Lhs::Cols == Rhs::Cols &&
            (Lhs::Rows == Rhs::Rows || (Lhs::Cols == 1 && Rhs::Rows == 1 && Lhs::Rows == Rhs::Cols) ||
             (Lhs::Rows == 1 && Rhs::Cols == 1 && Lhs::Cols == Rhs::Rows)))))),
      INVALID_OPERANDS_FOR_DOT_PRODUCT);
    fdapde_static_assert(
      std::is_convertible_v<typename Lhs::Scalar FDAPDE_COMMA typename Rhs::Scalar>,
      YOU_MIXED_FIELDS_WITH_NON_CONVERTIBLE_SCALAR_OUTPUT_TYPES);
   public:
    using LhsDerived = Lhs;
    using RhsDerived = Rhs;
    template <typename T1, typename T2> using Meta = DotProduct<T1, T2>;
    using Base = ScalarFieldBase<LhsDerived::StaticInputSize, DotProduct<Lhs, Rhs>>;
    using LhsInputType = typename LhsDerived::InputType;
    using RhsInputType = typename RhsDerived::InputType;
    using InputType = internals::prefer_most_derived_t<LhsInputType, RhsInputType>;
    using Scalar = decltype(std::declval<typename LhsDerived::Scalar>() * std::declval<typename RhsDerived::Scalar>());
    static constexpr int StaticInputSize = LhsDerived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = LhsDerived::XprBits | RhsDerived::XprBits;

    constexpr DotProduct(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) {
        if constexpr (LhsDerived::Cols == Dynamic || RhsDerived::Cols == Dynamic) {
            fdapde_assert(
              lhs.cols() == rhs.cols() &&
              (rhs.rows() == rhs.rows() || (lhs.cols() == 1 && rhs.rows() == 1 && lhs.rows() == rhs.cols()) ||
               (lhs.rows() == 1 && rhs.cols() == 1 && lhs.cols() == rhs.rows())));
        }
        if constexpr (LhsDerived::StaticInputSize == Dynamic || RhsDerived::StaticInputSize == Dynamic) {
            fdapde_assert(lhs.input_size() == rhs.input_size());
        }
    }
    constexpr Scalar operator()(const InputType& p) const {
        fdapde_static_assert(
          std::is_same_v<LhsInputType FDAPDE_COMMA RhsInputType> ||
            internals::are_related_by_inheritance_v<LhsInputType FDAPDE_COMMA RhsInputType>,
          YOU_MIXED_FIELDS_WITH_INCOMPATIBLE_INPUT_VECTOR_TYPES);
        if constexpr (internals::is_scalar_field_v<Lhs> && internals::is_scalar_field_v<Rhs>) {
            // scalar-scalar case, just plain multiplication
            return lhs_(p) * rhs_(p);
        } else {
            if constexpr (Lhs::Cols == Rhs::Cols && Lhs::Rows == Rhs::Rows && Lhs::Cols > 1 && Lhs::Rows > 1) {
                // matrix-matrix dot product, dot(A, B) = Tr[A * B^\top] = \sum_{i=1}^n (\sum_{j=1}^m (a_{ij} * b_{ji}))
                Scalar dot_ = 0;
                for (int i = 0; i < Lhs::Rows; ++i) {
                    for (int j = 0; j < Lhs::Cols; ++j) { dot_ += lhs_.eval(i, j, p) * rhs_.eval(i, j, p); }
                }
                return dot_;
            } else {   // vector-vector case
                Scalar dot_ = 0;
                int n = lhs_.cols() == 1 ? lhs_.rows() : lhs_.cols();
                for (int i = 0; i < n; ++i) { dot_ += lhs_.eval(i, p) * rhs_.eval(i, p); }
                return dot_;
            }
        }
    }
    constexpr int input_size() const { return lhs_.input_size(); }
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }
   private:
    typename internals::ref_select<const LhsDerived>::type lhs_;
    typename internals::ref_select<const RhsDerived>::type rhs_;
};

template <typename Lhs, typename Rhs>
constexpr DotProduct<Lhs, Rhs>
dot(const ScalarFieldBase<Lhs::StaticInputSize, Lhs>& lhs, const ScalarFieldBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return DotProduct<Lhs, Rhs>(lhs.derived(), rhs.derived());
}
template <int Size, typename Derived>
template <typename Rhs>
constexpr auto ScalarFieldBase<Size, Derived>::dot(const ScalarFieldBase<Size, Rhs>& rhs) const {
    return DotProduct<Derived, Rhs>(derived(), rhs.derived());
}
template <typename Lhs, typename Rhs>
constexpr DotProduct<Lhs, Rhs> dot(
  const MatrixFieldBase<Lhs::StaticInputSize, Lhs>& lhs, const MatrixFieldBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return DotProduct<Lhs, Rhs>(lhs.derived(), rhs.derived());
}

// integration with Eigen types
  
namespace internals {

template <
  typename Lhs, typename Rhs, typename FieldType_ = std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, Rhs, Lhs>>
class dot_product_eigen_impl : public ScalarFieldBase<FieldType_::StaticInputSize, dot_product_eigen_impl<Lhs, Rhs>> {
    using FieldType = std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, Rhs, Lhs>;
    using EigenType = std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, Lhs, Rhs>;
    static constexpr bool is_field_lhs = std::is_same_v<FieldType, Lhs>;
    fdapde_static_assert(
      (FieldType::Cols == 1 &&
       ((EigenType::ColsAtCompileTime == 1 && FieldType::Rows == EigenType::RowsAtCompileTime) ||
        (EigenType::RowsAtCompileTime == 1 && FieldType::Rows == EigenType::ColsAtCompileTime))) ||
      (FieldType::Rows == 1 &&
       ((EigenType::RowsAtCompileTime == 1 && FieldType::Cols == EigenType::ColsAtCompileTime) ||
        (EigenType::ColsAtCompileTime == 1 && FieldType::Cols == EigenType::RowsAtCompileTime))),
      INVALID_OPERAND_SIZES_FOR_DOT_PRODUCT);
   public:
    using LhsDerived = Lhs;
    using RhsDerived = Rhs;
    template <typename T1, typename T2> using Meta = dot_product_eigen_impl<T1, T2>;
    using Base = ScalarFieldBase<FieldType::StaticInputSize, dot_product_eigen_impl<Lhs, Rhs>>;
    using InputType = typename FieldType::InputType;
    using Scalar = decltype(std::declval<typename FieldType::Scalar>() * std::declval<typename EigenType::Scalar>());
    static constexpr int StaticInputSize = FieldType::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = FieldType::XprBits;

    dot_product_eigen_impl(const Lhs& lhs, const Rhs& rhs) : Base(), lhs_(lhs), rhs_(rhs) {
        if constexpr (
          FieldType::Cols == Dynamic || EigenType::ColsAtCompileTime == Dynamic || FieldType::Rows == Dynamic ||
          EigenType::RowsAtCompileTime == Dynamic) {
            fdapde_assert(
              (lhs.cols() == 1 &&
               ((rhs.cols() == 1 && lhs.rows() == rhs.rows()) || (rhs.rows() == 1 && lhs.rows() == rhs.cols()))) ||
              (lhs.rows() == 1 &&
               ((rhs.rows() == 1 && lhs.cols() == rhs.cols()) || (rhs.cols() == 1 && lhs.cols() == rhs.rows()))));
        }
    }
    Scalar operator()(const InputType& p) const {
        Scalar dot_ = 0;
        int n = lhs_.cols() == 1 ? lhs_.rows() : lhs_.cols();
        for (int i = 0; i < n; ++i) {
	    if constexpr (is_field_lhs) dot_ += lhs_.eval(i, p) * rhs_[i];
	    else dot_ += lhs_[i] * rhs_.eval(i, p);
	}
        return dot_;
    }
    constexpr int input_size() const {
        if constexpr (is_field_lhs)  return lhs_.input_size();
        else return rhs_.input_size();
    }
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }
   protected:
    std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, const Lhs, internals::ref_select_t<const Lhs>> lhs_;
    std::conditional_t<internals::is_eigen_dense_xpr_v<Rhs>, const Rhs, internals::ref_select_t<const Rhs>> rhs_;
};

}   // namespace internals

template <typename Lhs, typename Rhs>
struct DotProduct<Lhs, Eigen::MatrixBase<Rhs>> : public internals::dot_product_eigen_impl<Lhs, Rhs> {
    DotProduct(const Lhs& lhs, const Rhs& rhs) : internals::dot_product_eigen_impl<Lhs, Rhs>(lhs, rhs) { }
};
template <typename Lhs, typename Rhs>
struct DotProduct<Eigen::MatrixBase<Lhs>, Rhs> : public internals::dot_product_eigen_impl<Lhs, Rhs> {
    DotProduct(const Lhs& lhs, const Rhs& rhs) : internals::dot_product_eigen_impl<Lhs, Rhs>(lhs, rhs) { }
};

template <typename Lhs, typename Rhs>
constexpr DotProduct<Lhs, Eigen::MatrixBase<Rhs>>
dot(const MatrixFieldBase<Lhs::StaticInputSize, Lhs>& lhs, const Eigen::MatrixBase<Rhs>& rhs) {
    return DotProduct<Lhs, Eigen::MatrixBase<Rhs>>(lhs.derived(), rhs.derived());
}
template <typename Lhs, typename Rhs>
constexpr DotProduct<Eigen::MatrixBase<Lhs>, Rhs>
dot(const Eigen::MatrixBase<Lhs>& lhs, const MatrixFieldBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return DotProduct<Eigen::MatrixBase<Lhs>, Rhs>(lhs.derived(), rhs.derived());
}

}   // namespace fdapde

#endif   // __FDAPDE_DOT_H__
