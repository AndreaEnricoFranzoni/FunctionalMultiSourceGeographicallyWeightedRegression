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

#ifndef __FDAPDE_MATRIX_FIELD_H__
#define __FDAPDE_MATRIX_FIELD_H__

#include "header_check.h"

namespace fdapde {

#define MATRIX_FIELD_SAME_INPUT_TYPE(Lhs, Rhs)                                                                         \
    fdapde_static_assert(                                                                                              \
      std::is_same_v<typename Lhs::InputType FDAPDE_COMMA typename Rhs::InputType>,                                    \
      YOU_MIXED_MATRIX_FIELDS_WITH_DIFFERENT_INPUT_VECTOR_TYPES);

template <int Size, typename Derived> struct MatrixFieldBase;

template <typename Lhs, typename Rhs>
class MatrixFieldProduct : public MatrixFieldBase<Lhs::StaticInputSize, MatrixFieldProduct<Lhs, Rhs>> {
    fdapde_static_assert(
      Lhs::StaticInputSize == Dynamic || Rhs::StaticInputSize == Dynamic ||
        (Lhs::StaticInputSize != Dynamic && Rhs::StaticInputSize != Dynamic &&
         Lhs::StaticInputSize == Rhs::StaticInputSize),
      YOU_MIXED_MATRICES_WITH_DIFFERENT_INPUT_SIZES);
    fdapde_static_assert(
      std::is_convertible_v<typename Lhs::Scalar FDAPDE_COMMA typename Rhs::Scalar>,
      YOU_MIXED_MATRIX_FIELDS_WITH_NON_CONVERTIBLE_SCALAR_OUTPUT_TYPES);
   public:
    using LhsDerived = Lhs;
    using RhsDerived = Rhs;
    template <typename T1, typename T2> using Meta = MatrixFieldProduct<T1, T2>;
    using Base = MatrixFieldBase<Lhs::StaticInputSize, MatrixFieldProduct<Lhs, Rhs>>;
    using LhsInputType = typename LhsDerived::InputType;
    using RhsInputType = typename RhsDerived::InputType;
    using InputType = internals::prefer_most_derived_t<LhsInputType, RhsInputType>;
    using Scalar = decltype(std::declval<typename Lhs::Scalar>() * std::declval<typename Rhs::Scalar>());
    static constexpr int StaticInputSize = Lhs::StaticInputSize;
    static constexpr int Rows = Lhs::Rows;
    static constexpr int Cols = Rhs::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Lhs::XprBits | Rhs::XprBits;

    constexpr MatrixFieldProduct(const Lhs& lhs, const Rhs& rhs) requires(Rows != Dynamic && Cols != Dynamic)
        : Base(), lhs_(lhs), rhs_(rhs) {
        fdapde_static_assert(Lhs::Cols == Rhs::Rows, INVALID_OPERAND_SIZES_FOR_MATRIX_PRODUCT);
    }
    MatrixFieldProduct(const Lhs& lhs, const Rhs& rhs) requires(Rows == Dynamic || Cols == Dynamic)
        : Base(), lhs_(lhs), rhs_(rhs) {
        fdapde_assert(lhs_.cols() == rhs_.rows() && lhs_.input_size() == rhs_.input_size());
    }
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return rhs_.cols(); }
    constexpr int input_size() const { return lhs_.input_size(); }
    constexpr int size() const { return rows() * cols(); }
    constexpr const Lhs& lhs() const { return lhs_; }
    constexpr const Rhs& rhs() const { return rhs_; }
  
    // for matrix multiplication, it is more convenient to evaluate the two operands at p, and take the product of the
    // evaluations
    template <typename Dest> constexpr void eval_at(const InputType& p, Dest& dest) const {
        MATRIX_FIELD_SAME_INPUT_TYPE(Lhs, Rhs)
        fdapde_static_assert(
          std::is_invocable_v<Dest FDAPDE_COMMA int FDAPDE_COMMA int> ||
            internals::is_subscriptable<Dest FDAPDE_COMMA int>,
          DESTINATION_TYPE_MUST_EITHER_EXPOSE_A_MATRIX_LIKE_ACCESS_OPERATOR_OR_A_SUBSCRIPT_OPERATOR);

        // store evaluations in temporaries (O(n) function calls)
        using LhsStorageType = std::conditional_t<
          Lhs::Rows == Dynamic || Lhs::Cols == Dynamic, std::vector<Scalar>,
          std::array<Scalar, int(Lhs::Rows) * int(Lhs::Cols)>>;
        using RhsStorageType = std::conditional_t<
          Rhs::Rows == Dynamic || Rhs::Cols == Dynamic, std::vector<Scalar>,
          std::array<Scalar, int(Rhs::Rows) * int(Rhs::Cols)>>;
        LhsStorageType lhs_temp;
        RhsStorageType rhs_temp;
        if constexpr (Lhs::Rows == Dynamic || Lhs::Cols == Dynamic) lhs_temp.resize(lhs_.size());
        if constexpr (Rhs::Rows == Dynamic || Rhs::Cols == Dynamic) rhs_temp.resize(rhs_.size());
        for (int i = 0; i < lhs_.rows(); ++i) {
            for (int j = 0; j < lhs_.cols(); ++j) { lhs_temp[i * lhs_.cols() + j] = lhs_.eval(i, j, p); }
        }
        for (int i = 0; i < rhs_.cols(); ++i) {   // store col major to exploit cache locality in matrix product
            for (int j = 0; j < rhs_.rows(); ++j) { rhs_temp[i * rhs_.rows() + j] = rhs_.eval(j, i, p); }
        }
        // perform standard matrix-matrix product
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) {
                Scalar res = 0;
                for (int k = 0; k < lhs_.cols(); ++k) {
                    res += lhs_temp[i * lhs_.cols() + k] * rhs_temp[j * rhs_.rows() + k];
		}
                if constexpr (std::is_invocable_v<Dest, int, int>) {
                    dest(i, j) = res;
                } else {
                    dest[i * cols() + j] = res;
                }
            }
	}
        return;
    }  
    constexpr auto operator()(int i, int j) const {
        return [i, j, this](const InputType& p) {
            MATRIX_FIELD_SAME_INPUT_TYPE(Lhs, Rhs)
            Scalar res = 0;
            for (int k = 0; k < lhs_.cols(); ++k) { res += lhs_.eval(i, k, p) * rhs_.eval(k, j, p); }
            return res;
        };
    }
    constexpr Scalar eval(int i, int j, const InputType& p) const {
        MATRIX_FIELD_SAME_INPUT_TYPE(Lhs, Rhs)
        Scalar res = 0;
        for (int k = 0; k < lhs_.cols(); ++k) { res += lhs_.eval(i, k, p) * rhs_.eval(k, j, p); }
        return res;
    }
    constexpr Scalar eval(int i, const InputType& p) const {   // matrix-vector product
        MATRIX_FIELD_SAME_INPUT_TYPE(Lhs, Rhs);
        fdapde_static_assert(Rhs::Cols == 1 || Lhs::Rows == 1, INVALID_MATRIX_VECTOR_PRODUCT_DIMENSIONS);
        Scalar res = 0;
        if constexpr (Rhs::Cols == 1) {
            for (int k = 0; k < lhs_.cols(); ++k) { res += lhs_.eval(i, k, p) * rhs_.eval(k, p); }
        } else {
            for (int k = 0; k < rhs_.rows(); ++k) { res += lhs_.eval(k, p) * rhs_.eval(k, i, p); }
        }
	return res;
    }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   protected:
    internals::ref_select_t<const Lhs> lhs_;
    internals::ref_select_t<const Rhs> rhs_;
};
template <typename Lhs, typename Rhs>
constexpr MatrixFieldProduct<Lhs, Rhs> operator*(
  const MatrixFieldBase<Lhs::StaticInputSize, Lhs>& lhs, const MatrixFieldBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return MatrixFieldProduct<Lhs, Rhs> {lhs.derived(), rhs.derived()};
}

template <int BlockRows_, int BlockCols_, typename Derived_>
class MatrixFieldBlock :
    public MatrixFieldBase<Derived_::StaticInputSize, MatrixFieldBlock<BlockRows_, BlockCols_, Derived_>> {
    fdapde_static_assert(
      (BlockRows_ == Dynamic || (Derived_::Rows == Dynamic || (BlockRows_ > 0 && BlockRows_ <= Derived_::Rows))) &&
        (BlockCols_ == Dynamic || (Derived_::Cols == Dynamic || (BlockCols_ > 0 && BlockCols_ <= Derived_::Cols))),
      INVALID_BLOCK_SIZES);
   public:
    using Derived = Derived_;
    template <typename T> using Meta = MatrixFieldBlock<BlockRows_, BlockCols_, T>;
    using Base = MatrixFieldBase<Derived::StaticInputSize, MatrixFieldBlock<BlockRows_, BlockCols_, Derived>>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;
    static constexpr int ReadOnly = Derived::ReadOnly;

    // row/column constructor
    constexpr MatrixFieldBlock(const Derived& xpr, int i) :
        Base(),
        xpr_(xpr),
        start_row_(BlockRows_ == 1 ? i : 0),
        start_col_(BlockCols_ == 1 ? i : 0),
        block_rows_(BlockRows_ == 1 ? 1 : xpr.rows()),
        block_cols_(BlockCols_ == 1 ? 1 : xpr.cols()) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_AND_COLUMN_BLOCKS);
        fdapde_constexpr_assert(
          i >= 0 && ((BlockRows_ == 1 && i < xpr_.rows()) || (BlockCols_ == 1 && i < xpr_.cols())));
    }
    constexpr MatrixFieldBlock(const Derived& xpr, int start_row, int start_col) :
        Base(), xpr_(xpr), start_row_(start_row), start_col_(start_col), block_rows_(BlockRows_),
        block_cols_(BlockCols_) {
        fdapde_static_assert(
          BlockRows_ != Dynamic && BlockCols_ != Dynamic, THIS_METHOD_IS_ONLY_FOR_STATIC_SIZED_BLOCKS);
        fdapde_constexpr_assert(
          start_row_ >= 0 && start_row_ + BlockRows_ <= xpr_.rows() && start_col_ >= 0 &&
          start_col_ + BlockCols_ <= xpr_.cols());
    }
    MatrixFieldBlock(const Derived& xpr, int start_row, int start_col, int block_rows, int block_cols) :
        Base(), xpr_(xpr), start_row_(start_row), start_col_(start_col), block_rows_(block_rows),
        block_cols_(block_cols) {
        fdapde_static_assert(
          BlockRows_ == Dynamic || BlockCols_ == Dynamic, THIS_METHOD_IS_ONLY_FOR_DYNAMIC_SIZED_BLOCKS);
        fdapde_assert(
          start_row_ >= 0 && start_row_ + block_rows <= xpr_.rows() && start_col_ >= 0 &&
          start_col_ + block_cols <= xpr_.cols());
    }

    constexpr int rows() const { return block_rows_; }
    constexpr int cols() const { return block_cols_; }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return rows() * cols(); }
    constexpr const Derived& derived() const { return xpr_; }
    constexpr auto operator()(int i, int j) const { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr auto operator[](int i) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_BLOCKS);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    constexpr Scalar eval(int i, int j, const InputType& p) const {
        return xpr_.eval(start_row_ + i, start_col_ + j, p);
    }
    constexpr Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_BLOCKS);
        if constexpr (Rows == 1) return xpr_.eval(start_row_, start_col_ + i, p);
        if constexpr (Cols == 1) return xpr_.eval(start_row_ + i, start_col_, p);
    }
    // block assignment
    template <int Size_, typename RhsDerived>
    constexpr MatrixFieldBlock<BlockRows_, BlockCols_, Derived>&
    operator=(const MatrixFieldBase<Size_, RhsDerived>& rhs)
        requires(BlockRows_ != Dynamic && BlockCols_ != Dynamic) {
        fdapde_static_assert(Derived::ReadOnly != 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          RhsDerived::Rows == BlockRows_ && RhsDerived::Cols == BlockCols_ &&
            std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA typename Derived::FunctorType>,
          INVALID_BLOCK_SIZE_OR_YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_FUNCTOR_TYPE);
        for (int i = 0; i < xpr_.rows(); ++i) {
            for (int j = 0; j < xpr_.cols(); ++j) { xpr_(start_row_ + i, start_col_ + j) = rhs(i, j); }
        }
	return *this;
    }
    template <int Size_, typename RhsDerived>
    MatrixFieldBlock<BlockRows_, BlockCols_, Derived>& operator=(const MatrixFieldBase<Size_, RhsDerived>& rhs)
        requires(BlockRows_ == Dynamic || BlockCols_ == Dynamic) {
        fdapde_static_assert(Derived::ReadOnly != 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA typename Derived::FunctorType>,
          YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_COEFFICIENT_TYPE);
        fdapde_assert(rhs.rows() == xpr_.rows() && rhs.cols() == xpr_.cols());
        for (int i = 0; i < xpr_.rows(); ++i) {
            for (int j = 0; j < xpr_.cols(); ++j) { xpr_(start_row_ + i, start_col_ + j) = rhs(i, j); }
        }
	return *this;
    }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   private:
    int start_row_ = 0, start_col_ = 0;
    int block_rows_ = 0, block_cols_ = 0;
    internals::ref_select_t<Derived> xpr_;
};

template <typename Lhs, typename Rhs, typename BinaryOperation>
class MatrixFieldBinOp : public MatrixFieldBase<Lhs::StaticInputSize, MatrixFieldBinOp<Lhs, Rhs, BinaryOperation>> {
    fdapde_static_assert(
      (Lhs::StaticInputSize == Dynamic || Rhs::StaticInputSize == Dynamic ||
       Lhs::StaticInputSize == Rhs::StaticInputSize) &&
        (Lhs::Rows == Dynamic || Rhs::Rows == Dynamic || Lhs::Rows == Rhs::Rows) &&
        (Lhs::Cols == Dynamic || Rhs::Cols == Dynamic || Lhs::Cols == Rhs::Cols),
      YOU_MIXED_MATRIX_FIELDS_OF_DIFFERENT_SIZES);
    fdapde_static_assert(
      std::is_convertible_v<typename Lhs::Scalar FDAPDE_COMMA typename Rhs::Scalar>,
      YOU_MIXED_MATRIX_FIELDS_WITH_NON_CONVERTIBLE_SCALAR_OUTPUT_TYPES);
    internals::ref_select_t<const Lhs> lhs_;
    internals::ref_select_t<const Rhs> rhs_;
    BinaryOperation op_;
   public:
    using LhsDerived = Lhs;
    using RhsDerived = Rhs;
    template <typename T1, typename T2> using Meta = MatrixFieldBinOp<T1, T2, BinaryOperation>;
    using Base = MatrixFieldBase<Lhs::StaticInputSize, MatrixFieldBinOp<Lhs, Rhs, BinaryOperation>>;
    using LhsInputType = typename LhsDerived::InputType;
    using RhsInputType = typename RhsDerived::InputType;
    using InputType = internals::prefer_most_derived_t<LhsInputType, RhsInputType>;
    using Scalar = decltype(std::declval<BinaryOperation>().operator()(
      std::declval<typename Lhs::Scalar>(), std::declval<typename Rhs::Scalar>()));
    static constexpr int StaticInputSize = Lhs::StaticInputSize;
    static constexpr int Rows = Lhs::Rows;
    static constexpr int Cols = Lhs::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Lhs::XprBits | Rhs::XprBits;
    static constexpr int ReadOnly = 1;

    constexpr MatrixFieldBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op)
        requires(StaticInputSize != Dynamic && Rows != Dynamic && Cols != Dynamic)
        : Base(), lhs_(lhs), rhs_(rhs), op_(op) { }
    MatrixFieldBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op)
        requires(StaticInputSize == Dynamic || Rows == Dynamic || Cols == Dynamic)
        : lhs_(lhs), rhs_(rhs), op_(op) {
        fdapde_assert(
          (lhs.input_size() == rhs.input_size()) && (Rows != Dynamic || lhs.rows() == rhs.rows()) &&
          (Cols != Dynamic || lhs.cols() == rhs.cols()));
    }
    MatrixFieldBinOp(const Lhs& lhs, const Rhs& rhs) : MatrixFieldBinOp(lhs, rhs, BinaryOperation {}) { }

    constexpr Scalar eval(int i, int j, const InputType& p) const {
        MATRIX_FIELD_SAME_INPUT_TYPE(Lhs, Rhs)
        return op_(lhs_.eval(i, j, p), rhs_.eval(i, j, p));
    }
    constexpr Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_MATRICES);
        MATRIX_FIELD_SAME_INPUT_TYPE(Lhs, Rhs)
        return op_(lhs_.eval(i, p), rhs_.eval(i, p));
    }
    constexpr auto operator()(int i, int j) const {
        return [i, j, this](const InputType& p) {
            MATRIX_FIELD_SAME_INPUT_TYPE(Lhs, Rhs)
            return op_(lhs_.eval(i, j, p), rhs_.eval(i, j, p));
        };
    }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_MATRICES);
        return [i, this](const InputType& p) {
            MATRIX_FIELD_SAME_INPUT_TYPE(Lhs, Rhs)
            return op_(lhs_.eval(i, p), rhs_.eval(i, p));
        };
    }  
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return lhs_.cols(); }
    constexpr int input_size() const { return lhs_.input_size(); }
    constexpr int size() const { return lhs_.size(); }
    constexpr const Lhs& lhs() const { return lhs_; }
    constexpr const Rhs& rhs() const { return rhs_; }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
};
template <typename Lhs, typename Rhs>
constexpr MatrixFieldBinOp<Lhs, Rhs, std::plus<>> operator+(
  const MatrixFieldBase<Lhs::StaticInputSize, Lhs>& lhs, const MatrixFieldBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return MatrixFieldBinOp<Lhs, Rhs, std::plus<>>{lhs.derived(), rhs.derived(), std::plus<>()};
}
template <typename Lhs, typename Rhs>
constexpr MatrixFieldBinOp<Lhs, Rhs, std::minus<>> operator-(
  const MatrixFieldBase<Lhs::StaticInputSize, Lhs>& lhs, const MatrixFieldBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return MatrixFieldBinOp<Lhs, Rhs, std::minus<>>{lhs.derived(), rhs.derived(), std::minus<>()};
}

template <typename Lhs, typename Rhs, typename BinaryOperation>
class MatrixFieldCoeffWiseOp :
    public MatrixFieldBase<
      std::conditional_t<std::is_arithmetic_v<Lhs> || internals::is_scalar_field_v<Lhs>, Rhs, Lhs>::StaticInputSize,
      MatrixFieldCoeffWiseOp<Lhs, Rhs, BinaryOperation>> {
   public:
    using LhsDerived = Lhs;
    using RhsDerived = Rhs;
    template <typename T1, typename T2> using Meta = MatrixFieldCoeffWiseOp<T1, T2, BinaryOperation>;
    static constexpr bool is_coeff_lhs = std::is_arithmetic_v<Lhs> || internals::is_scalar_field_v<Lhs>;
    using CoeffType = std::conditional_t<is_coeff_lhs, Lhs, Rhs>;
   private:
    // keep this private to avoid to consider ScalarCoeffOp as a unary node
    using Derived = std::conditional_t<is_coeff_lhs, Rhs, Lhs>;
    constexpr const Derived& derived() const { if constexpr(is_coeff_lhs) return rhs_; else return lhs_; }
    constexpr const CoeffType& coeff() const { if constexpr(is_coeff_lhs) return lhs_; else return rhs_; }
   public:
    static constexpr bool is_coeff_scalar_field = internals::is_scalar_field_v<CoeffType>;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    using Base = MatrixFieldBase<StaticInputSize, MatrixFieldCoeffWiseOp<Lhs, Rhs, BinaryOperation>>;
    using Scalar = decltype(std::declval<BinaryOperation>().operator()(
      std::declval<typename Derived::Scalar>(), std::declval<decltype([]() {
          if constexpr (internals::is_scalar_field_v<CoeffType>) {
              return typename CoeffType::Scalar {};
          } else {
              return CoeffType {};
          }
      }())>()));
    using InputType = typename Derived::InputType;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = []() {
        if constexpr (is_coeff_scalar_field) {
            return Lhs::XprBits | Rhs::XprBits;
        } else {
            return Derived::XprBits;
        }
    }();
    static constexpr int ReadOnly = 1;

    constexpr MatrixFieldCoeffWiseOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) :
        Base(), lhs_(lhs), rhs_(rhs), op_(op) { }
    constexpr MatrixFieldCoeffWiseOp(const Lhs& lhs, const Rhs& rhs) :
        MatrixFieldCoeffWiseOp(lhs, rhs, BinaryOperation {}) { }

    template <typename Dest>
    constexpr void eval_at(const InputType& p, Dest& dest) const
        requires(is_coeff_scalar_field) {
        fdapde_static_assert(
          std::is_invocable_v<Dest FDAPDE_COMMA int FDAPDE_COMMA int> ||
            internals::is_subscriptable<Dest FDAPDE_COMMA int>,
          DESTINATION_TYPE_MUST_EITHER_EXPOSE_A_MATRIX_LIKE_ACCESS_OPERATOR_OR_A_SUBSCRIPT_OPERATOR);
        // store evaluation of scalar field in temporary (just one invocation)
        Scalar tmp = coeff()(p);
        // perform standard scalar-matrix product
        Scalar res = 0;
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) {
                if constexpr (is_coeff_lhs)  { res = op_(tmp, derived().eval(i, j, p)); }
                if constexpr (!is_coeff_lhs) { res = op_(derived().eval(i, j, p), tmp); }
                if constexpr (std::is_invocable_v<Dest, int, int>) {
                    dest(i, j) = res;
                } else {
                    dest[i * cols() + j] = res;
                }
            }
        }
        return;
    }
    template <typename InputType_> constexpr Scalar eval(int i, int j, const InputType_& p) const {
        fdapde_static_assert(
          std::is_arithmetic_v<CoeffType> || std::is_invocable_v<CoeffType FDAPDE_COMMA InputType>,
          COEFFICIENT_IS_NOT_AN_ARITHMETIC_TYPE_AND_IS_NOT_INVOCABLE_AT_INPUT_TYPE);
        if constexpr (is_coeff_lhs) {
            if constexpr (internals::is_scalar_field_v<LhsDerived>) {
                return op_(lhs_(p), rhs_.eval(i, j, p));
            } else {
                return op_(lhs_, rhs_.eval(i, j, p));
            }
        } else {
            if constexpr (internals::is_scalar_field_v<RhsDerived>) {
                return op_(lhs_.eval(i, j, p), rhs_(p));
            } else {
                return op_(lhs_.eval(i, j, p), rhs_);
            }	  
	}
    }
    template <typename InputType_> constexpr Scalar eval(int i, const InputType_& p) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_MATRICES);
        fdapde_static_assert(
          std::is_arithmetic_v<CoeffType> || std::is_invocable_v<CoeffType FDAPDE_COMMA InputType>,
          COEFFICIENT_IS_NOT_AN_ARITHMETIC_TYPE_AND_IS_NOT_INVOCABLE_AT_INPUT_TYPE);
        if constexpr (is_coeff_lhs) {
            if constexpr (internals::is_scalar_field_v<LhsDerived>) {
                return op_(lhs_(p), rhs_.eval(i, p));
            } else {
                return op_(lhs_, rhs_.eval(i, p));
            }
        } else {
            if constexpr (internals::is_scalar_field_v<RhsDerived>) {
                return op_(lhs_.eval(i, p), rhs_(p));
            } else {
                return op_(lhs_.eval(i, p), rhs_);
            }
        }
    }
    constexpr auto operator()(int i, int j) const {
        return [i, j, this](const InputType& p) { this->eval(i, j, p); };
    }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_MATRICES);
        return [i, this](const InputType& p) { this->eval(i, p); };
    }
    constexpr int rows() const { return derived().rows(); }
    constexpr int cols() const { return derived().cols(); }
    constexpr int input_size() const { return derived().input_size(); }
    constexpr int size() const { return derived().size(); }
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   protected:
    internals::ref_select_t<const LhsDerived> lhs_;
    internals::ref_select_t<const RhsDerived> rhs_;
    BinaryOperation op_;
};

template <int Size, typename Lhs, typename Rhs>
constexpr MatrixFieldCoeffWiseOp<Lhs, Rhs, std::multiplies<>>
operator*(const MatrixFieldBase<Size, Lhs>& lhs, const Rhs& rhs)
    requires(std::is_arithmetic_v<Rhs> || internals::is_scalar_field_v<Rhs>) {
    if constexpr (internals::is_scalar_field_v<Rhs>) {
        return MatrixFieldCoeffWiseOp<Lhs, Rhs, std::multiplies<>>(lhs.derived(), rhs.derived(), std::multiplies<>());
    } else {
        return MatrixFieldCoeffWiseOp<Lhs, Rhs, std::multiplies<>>(lhs.derived(), rhs, std::multiplies<>());
    }
}
template <int Size, typename Lhs, typename Rhs>
constexpr MatrixFieldCoeffWiseOp<Lhs, Rhs, std::multiplies<>>
operator*(const Lhs& lhs, const MatrixFieldBase<Size, Rhs>& rhs)
    requires(std::is_arithmetic_v<Lhs> || internals::is_scalar_field_v<Lhs>) {
    if constexpr (internals::is_scalar_field_v<Lhs>) {
        return MatrixFieldCoeffWiseOp<Lhs, Rhs, std::multiplies<>>(lhs.derived(), rhs.derived(), std::multiplies<>());
    } else {
        return MatrixFieldCoeffWiseOp<Lhs, Rhs, std::multiplies<>>(lhs, rhs.derived(), std::multiplies<>());
    }
}
template <int Size, typename Lhs, typename Rhs>
constexpr MatrixFieldCoeffWiseOp<Lhs, Rhs, std::divides<>>
operator*(const MatrixFieldBase<Size, Lhs>& lhs, const Rhs& rhs)
    requires(std::is_arithmetic_v<Rhs> || internals::is_scalar_field_v<Rhs>) {
    if constexpr (internals::is_scalar_field_v<Rhs>) {
        return MatrixFieldCoeffWiseOp<Lhs, Rhs, std::divides<>>(lhs.derived(), rhs.derived(), std::divides<>());
    } else {
        return MatrixFieldCoeffWiseOp<Lhs, Rhs, std::divides<>>(lhs.derived(), rhs, std::divides<>());
    }
}

template <
  int StaticInputSize_, int Rows_, int Cols_,
  typename FunctorType_ = std::function<double(internals::static_dynamic_eigen_vector_selector_t<StaticInputSize_>)>>
class MatrixField :
    public MatrixFieldBase<StaticInputSize_, MatrixField<StaticInputSize_, Rows_, Cols_, FunctorType_>> {
    template <typename T> struct is_dynamic_sized {
        static constexpr bool value = (StaticInputSize_ == Dynamic || Rows_ == Dynamic || Cols_ == Dynamic);
    };
    using This = MatrixField<StaticInputSize_, Rows_, Cols_, FunctorType_>;
    using StorageType = typename std::conditional<
      is_dynamic_sized<This>::value, std::vector<FunctorType_>, std::array<FunctorType_, Rows_ * Cols_>>::type;
    using Base = MatrixFieldBase<StaticInputSize_, MatrixField<StaticInputSize_, Rows_, Cols_, FunctorType_>>;
    using traits = internals::fn_ptr_traits<&FunctorType_::operator()>;
    fdapde_static_assert(traits::n_args == 1, PROVIDED_FUNCTOR_MUST_ACCEPT_ONLY_ONE_ARGUMENT);

    StorageType data_;
    int inner_size_ = 0;
    int n_rows_ = 0, n_cols_ = 0;
   public:
    using FunctorType = std::decay_t<FunctorType_>;
    using InputType = std::decay_t<std::tuple_element_t<0, typename traits::ArgsType>>;
    using Scalar = typename std::invoke_result<FunctorType, InputType>::type;
    static constexpr int StaticInputSize = StaticInputSize_;   // dimensionality of base space (can be Dynamic)
    static constexpr int NestAsRef = 0;                        // whether to store the node by reference of by copy
    static constexpr int XprBits = 0;
    static constexpr int ReadOnly = 0;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
  
    // static sized constructor
    constexpr MatrixField() requires(!is_dynamic_sized<This>::value)
        : Base(), data_(), inner_size_(StaticInputSize), n_rows_(Rows), n_cols_(Cols) { }
    // dynamic sized constructor
    MatrixField() requires(is_dynamic_sized<This>::value)
        : Base(), data_(), inner_size_(0), n_rows_(0), n_cols_(0) { }
    MatrixField(int inner_size, int rows, int cols) requires(is_dynamic_sized<This>::value)
        : Base(), inner_size_(inner_size), n_rows_(rows), n_cols_(cols) {
        fdapde_assert(rows > 0 && cols > 0);
        data_.resize(rows * cols);
    }
    // vector constructor
    explicit MatrixField(int rows) requires(is_dynamic_sized<This>::value) : Base(rows, 1) {
        fdapde_static_assert(Rows == Dynamic && Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
    }
    template <int Size_, typename RhsDerived>
    explicit constexpr MatrixField(const MatrixFieldBase<Size_, RhsDerived>& other)
        requires(!is_dynamic_sized<This>::value)
        : Base(), data_() {
        fdapde_static_assert(
          StaticInputSize == Size_ && Rows == RhsDerived::Row && Cols == RhsDerived::Cols &&
            std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA FunctorType>,
          INVALID_RHS_SIZE_OR_NON_CONVERTIBLE_FUNCTOR_TYPE);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = other(i, j); }
        }
    }
    template <int Size_, typename RhsDerived>
    explicit MatrixField(const MatrixFieldBase<Size_, RhsDerived>& other)
        requires(is_dynamic_sized<This>::value)
        : Base(other.rows(), other.cols()) {
        fdapde_static_assert(
          StaticInputSize == Size_ && std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA FunctorType>,
          INVALID_INPUT_SIZE_OR_NON_CONVERTIBLE_FUNCTOR_TYPE);
	fdapde_assert(rows() == other.rows() && cols() == other.cols());
        for (int i = 0; i < n_rows_; ++i) {
            for (int j = 0; j < n_cols_; ++j) { operator()(i, j) = other(i, j); }
        }
    }

    // assignment
    template <int Size_, typename RhsDerived>
    constexpr MatrixField<StaticInputSize, Rows, Cols, FunctorType>&
    operator=(const MatrixFieldBase<Size_, RhsDerived>& rhs)
        requires(Rows != Dynamic && Cols != Dynamic)
    {
        fdapde_static_assert(
          StaticInputSize_ == Size_ && Rows == RhsDerived::Rows && Cols == RhsDerived::Cols &&
            std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA FunctorType>,
          INVALID_RHS_SIZE_OR_NON_CONVERTIBLE_RHS_FUNCTOR_TYPE_IN_ASSIGNMENT);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = rhs.derived()(i, j); }
        }
        return *this;
    }
    template <int Size_, typename RhsDerived>
    MatrixField<StaticInputSize, Rows, Cols, FunctorType>& operator=(const MatrixFieldBase<Size_, RhsDerived>& rhs)
        requires(StaticInputSize == Dynamic || Rows == Dynamic || Cols == Dynamic) {
        using RhsFunctorType =
          decltype(std::declval<RhsDerived>().operator()(std::declval<int>(), std::declval<int>()));
        fdapde_static_assert(
          std::is_convertible_v<RhsFunctorType FDAPDE_COMMA FunctorType>,
          NON_CONVERTIBLE_RHS_FUNCTOR_TYPE_IN_ASSIGNMENT);
        if constexpr (Rows == Dynamic) n_rows_ = rhs.derived().rows();
        if constexpr (Cols == Dynamic) n_cols_ = rhs.derived().cols();
        if constexpr (StaticInputSize == Dynamic) inner_size_ = rhs.derived().input_size();
        data_.resize(n_rows_ * n_cols_);
        for (int i = 0; i < n_cols_; ++i) {
            for (int j = 0; j < n_rows_; ++j) { operator()(i, j) = rhs.derived()(i, j); }
        }
        return *this;
    }

    void resize(int inner_size, int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_ONLY_FOR_DYNAMIC_SIZED_MATRICES);
        fdapde_assert(inner_size > 0 && rows > 0 && cols > 0);
        if constexpr (Rows == Dynamic) n_rows_ = rows;
        if constexpr (Cols == Dynamic) n_cols_ = cols;
        data_ = std::vector<FunctorType>(n_rows_ * n_cols_);
        if constexpr (StaticInputSize == Dynamic) inner_size_ = inner_size;
        return;
    }
    void resize(int inner_size, int size) {
        fdapde_static_assert(
          (Rows == Dynamic && Cols == 1) || (Cols == Dynamic && Rows == 1),
          THIS_METHOD_IS_ONLY_FOR_DYNAMIC_SIZED_VECTORS);
        fdapde_assert(inner_size > 0 && size > 0);
        if constexpr (Rows == Dynamic) n_rows_ = size;
        n_cols_ = 1;
        data_ = std::vector<FunctorType>(n_rows_ * n_cols_);
        if constexpr (StaticInputSize == Dynamic) inner_size_ = inner_size;
        return;
    }
    // getters
    constexpr Scalar eval(int i, int j, const InputType& p) const {
        if constexpr (is_dynamic_sized<This>::value) { fdapde_assert(p.size() == inner_size_); }
        return data_[i * cols() + j](p);
    }
    constexpr Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        if constexpr (is_dynamic_sized<This>::value) { fdapde_assert(p.size() == inner_size_); }
        return data_[i](p);
    }
    constexpr const FunctorType& operator()(int i, int j) const { return data_[i * cols() + j]; }
    constexpr FunctorType& operator()(int i, int j) { return data_[i * cols() + j]; }
    constexpr const FunctorType& operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        return data_[i];
    }
    constexpr FunctorType& operator[](int i) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        return data_[i];
    }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
    constexpr int rows() const { return n_rows_; }
    constexpr int cols() const { return n_cols_; }
    constexpr int input_size() const { return inner_size_; }
    constexpr int size() const { return n_rows_ * n_cols_; }
};

template <int StaticInputSize, int Rows, typename Derived>
using VectorField = MatrixField<StaticInputSize, Rows, 1, Derived>;

template <typename Derived_>
struct MatrixFieldTranspose : public MatrixFieldBase<Derived_::StaticInputSize, MatrixFieldTranspose<Derived_>> {
    using Derived = Derived_;
    template <typename T> using Meta = MatrixFieldTranspose<T>;
    using Base = MatrixFieldBase<Derived::StaticInputSize, MatrixFieldTranspose<Derived>>;
    using Scalar = typename Derived::Scalar;
    using InputType = typename Derived::InputType;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::Cols;
    static constexpr int Cols = Derived::Rows;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;
    static constexpr int ReadOnly = 1;

    constexpr explicit MatrixFieldTranspose(const Derived& xpr) : Base(), xpr_(xpr) { }
    constexpr Scalar eval(int i, int j, const InputType& p) const { return xpr_.eval(j, i, p); }
    constexpr Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        return xpr_.eval(i, p);
    }
    constexpr const auto& operator()(int i, int j) const { return xpr_(j, i); }
    constexpr const auto& operator[](int i) const { return xpr_[i]; }
    constexpr int rows() const { return xpr_.cols(); }
    constexpr int cols() const { return xpr_.rows(); }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return xpr_.size(); }
    constexpr const Derived& derived() const { return xpr_; }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   protected:
    internals::ref_select_t<const Derived> xpr_;
};

template <typename Derived_>
struct MatrixFieldDiagonalBlock :
    public MatrixFieldBase<Derived_::StaticInputSize, MatrixFieldDiagonalBlock<Derived_>> {
    fdapde_static_assert(Derived_::Rows == Derived_::Cols, DIAGONAL_BLOCK_DEFINED_ONLY_FOR_SQUARED_MATRICES);
    using Derived = Derived_;
    template <typename T> using Meta = MatrixFieldDiagonalBlock<T>;  
    using Base = MatrixFieldBase<Derived::StaticInputSize, MatrixFieldDiagonalBlock<Derived>>;
    using Scalar = typename Derived::Scalar;
    using InputType = typename Derived::InputType;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;
    static constexpr int ReadOnly = 1;

    constexpr explicit MatrixFieldDiagonalBlock(const Derived& xpr) : Base(), xpr_(xpr) { }
    constexpr Scalar eval(int i, int j, const InputType& p) const { return i != j ? Scalar(0) : xpr_.eval(i, j, p); }
    constexpr const auto& operator[](int i) const { return xpr_(i, i); }
    constexpr auto& operator[](int i) { return xpr_(i, i); }
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return xpr_.size(); }
    constexpr const Derived& derived() const { return xpr_; }
    // diagonal assignment
    template <int Size_, typename RhsDerived>
    constexpr MatrixFieldDiagonalBlock<Derived>& operator=(const MatrixFieldBase<Size_, RhsDerived>& rhs)
        requires(Rows != Dynamic && Cols != Dynamic) {
        fdapde_static_assert(Derived::ReadOnly != 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          RhsDerived::Cols == 1 && RhsDerived::Rows == Rows &&
            std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA typename Derived::FunctorType>,
          VECTOR_FIELD_REQUIRED_OR_YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_FUNCTOR_TYPE);
        for (int i = 0; i < xpr_.rows(); ++i) { xpr_(i, i) = rhs[i]; }
        return *this;
    }
    template <int Size_, typename RhsDerived>
    MatrixFieldDiagonalBlock<Derived>& operator=(const MatrixFieldBase<Size_, RhsDerived>& rhs)
        requires(Rows == Dynamic || Cols == Dynamic) {
        fdapde_static_assert(Derived::ReadOnly != 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA typename Derived::FunctorType>,
          YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_COEFFICIENT_TYPE);
        fdapde_assert(rhs.rows() == xpr_.rows() && rhs.cols() == 1);
        for (int i = 0; i < xpr_.rows(); ++i) { xpr_(i, i) = rhs[i]; }
        return *this;
    }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   protected:
    internals::ref_select_t<Derived> xpr_;
};

template <typename Derived_, int ViewMode>
struct MatrixFieldSymmetricView :
    public MatrixFieldBase<Derived_::StaticInputSize, MatrixFieldSymmetricView<Derived_, ViewMode>> {
    fdapde_static_assert(
      (Derived_::Rows == Dynamic || Derived_::Cols == Dynamic) || Derived_::Rows == Derived_::Cols,
      SYMMETRIC_MATRIX_CONCEPT_DEFINED_ONLY_FOR_SQUARED_MATRICES);
    fdapde_static_assert(
      ViewMode == Upper || ViewMode == Lower, SYMMETRIC_VIEWS_MUST_BE_EITHER_LOWER_OR_UPPER);
    using Derived = Derived_;
    template <typename T> using Meta = MatrixFieldSymmetricView<T, ViewMode>;
    using Base = MatrixFieldBase<Derived::StaticInputSize, MatrixFieldSymmetricView<Derived, ViewMode>>;
    using Scalar = typename Derived::Scalar;
    using InputType = typename Derived::InputType;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;
    static constexpr int ReadOnly = 1;

    constexpr MatrixFieldSymmetricView() = default;
    constexpr MatrixFieldSymmetricView(const Derived& xpr) requires(Rows != Dynamic && Cols != Dynamic)
        : Base(), xpr_(xpr) { }
    MatrixFieldSymmetricView(const Derived& xpr) requires(Rows == Dynamic || Cols == Dynamic) : Base(), xpr_(xpr) {
        fdapde_assert(xpr_.rows() == xpr_.cols());
    }

    template <typename Dest> constexpr void eval_at(const InputType& p, Dest& dest) const {
        fdapde_static_assert(
          std::is_invocable_v<Dest FDAPDE_COMMA int FDAPDE_COMMA int> ||
            internals::is_subscriptable<Dest FDAPDE_COMMA int>,
          DESTINATION_TYPE_MUST_EITHER_EXPOSE_A_MATRIX_LIKE_ACCESS_OPERATOR_OR_A_SUBSCRIPT_OPERATOR);
        // just evaluate half of the coefficients
        Scalar tmp = 0;
        int row = 0, col = 0;
        for (int i = 0; i < xpr_.rows(); ++i) {
            for (int j = 0; j < i; ++j) {
                if constexpr (ViewMode == Lower) { row = i; col = j; }
                if constexpr (ViewMode == Upper) { row = j; col = i; }
		tmp = xpr_.eval(row, col, p);
                if constexpr (std::is_invocable_v<Dest, int, int>) {
                    dest(row, col) = tmp;
                    dest(col, row) = tmp;
                } else {
                    dest[row * xpr_.cols() + col] = tmp;
                    dest[col * xpr_.cols() + row] = tmp;
                }
            }
        }
        // evaluate coefficients on the diagonal
        for (int i = 0; i < xpr_.rows(); ++i) {
            if constexpr (std::is_invocable_v<Dest, int, int>) {
                dest(i, i) = xpr_.eval(i, i, p);
            } else {
                dest[i * xpr_.cols() + i] = xpr_.eval(i, i, p);
            }
        }
        return;
    }
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return xpr_.size(); }
    constexpr const Derived& derived() const { return xpr_; }
    constexpr decltype(auto) operator()(int i, int j) const {
        if constexpr (ViewMode == Lower) return i < j ? xpr_(j, i) : xpr_(i, j);
        if constexpr (ViewMode == Upper) return i > j ? xpr_(j, i) : xpr_(i, j);
    }
    constexpr Scalar eval(int i, int j, const InputType& p) const {
        if constexpr (ViewMode == Lower) return i < j ? xpr_.eval(j, i, p) : xpr_.eval(i, j, p);
        if constexpr (ViewMode == Upper) return i > j ? xpr_.eval(j, i, p) : xpr_.eval(i, j, p);
    }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   protected:
    internals::ref_select_t<Derived> xpr_;
};

// base class for matrix expressions
template <int StaticInputSize, typename Derived> struct MatrixFieldBase {
   private:
    template <typename InputType_> constexpr auto call_(const InputType_& p) const {
        using OutputType = std::conditional_t<
          internals::is_eigen_dense_xpr_v<InputType_>,
          Eigen::Matrix<
            typename Derived::Scalar, Derived::Rows == Dynamic ? Dynamic : InputType_::RowsAtCompileTime,
            Derived::Cols == Dynamic ? Dynamic : InputType_::ColsAtCompileTime>,
          Matrix<typename Derived::Scalar, Derived::Rows, Derived::Cols>>;
        OutputType out;
        if constexpr (Derived::StaticInputSize == Dynamic) { fdapde_assert(p.size() == derived().input_size()); }
        if constexpr (
          Derived::Rows == Dynamic || Derived::Cols == Dynamic || InputType_::RowsAtCompileTime == Dynamic ||
          InputType_::ColsAtCompileTime == Dynamic) {
            fdapde_assert(p.rows() == derived().rows() && p.cols() == derived().cols());
            out.resize(derived().rows(), derived().cols());
        }
        eval_at(p, out);
        return out;
    }
   public:
    friend Derived;
    constexpr MatrixFieldBase() = default;

    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }
    // evaluate the expression at point p storing result in dest
    template <typename InputType, typename Dest> constexpr void eval_at(const InputType& p, Dest& dest) const {
        fdapde_static_assert(
          std::is_invocable_v<Dest FDAPDE_COMMA int FDAPDE_COMMA int> ||
            internals::is_subscriptable<Dest FDAPDE_COMMA int>,
          DESTINATION_TYPE_MUST_EITHER_EXPOSE_A_MATRIX_LIKE_ACCESS_OPERATOR_OR_A_SUBSCRIPT_OPERATOR);
        for (int i = 0; i < derived().rows(); ++i) {
 	    for (int j = 0; j < derived().cols(); ++j) {
                if constexpr (std::is_invocable_v<Dest, int, int>) {
                    dest(i, j) = derived().eval(i, j, p);
                } else {
                    dest[i * derived().cols() + j] = derived().eval(i, j, p);
                }
            }
        }
        return;
    }
    // transpose
    constexpr MatrixFieldTranspose<Derived> transpose() const { return MatrixFieldTranspose<Derived>(derived()); }
    // block operations
    template <int BlockRows, int BlockCols>   // static sized block
    constexpr MatrixFieldBlock<BlockRows, BlockCols, Derived> block(int i, int j) {
        return MatrixFieldBlock<BlockRows, BlockCols, Derived>(derived(), i, j);
    }
    MatrixFieldBlock<Dynamic, Dynamic, Derived>   // dynamic sized block
    block(int start_row, int start_col, int block_rows, int block_cols) {
        return MatrixFieldBlock<Dynamic, Dynamic, Derived>(derived(), start_row, start_col, block_rows, block_cols);
    }
    constexpr auto col(int i) { return MatrixFieldBlock<Derived::Rows, 1, Derived>(derived(), i); }
    constexpr auto col(int i) const { return MatrixFieldBlock<Derived::Rows, 1, const Derived>(derived(), i); }
    constexpr auto row(int i) { return MatrixFieldBlock<1, Derived::Cols, Derived>(derived(), i); }
    constexpr auto row(int i) const { return MatrixFieldBlock<1, Derived::Cols, const Derived>(derived(), i); }
    // other block-type accessors
    MatrixFieldBlock<Dynamic, Dynamic, Derived> top_rows(int n) { return block(0, 0, n, derived().cols()); }
    MatrixFieldBlock<Dynamic, Dynamic, Derived> bottom_rows(int n) {
        return block(derived().rows() - n, 0, n, derived().cols());
    }
    MatrixFieldBlock<Dynamic, Dynamic, Derived> left_cols(int n) { return block(0, 0, derived().rows(), n); }
    MatrixFieldBlock<Dynamic, Dynamic, Derived> right_cols(int n) {
        return block(0, derived().cols() - n, derived().rows(), n);
    }
    // unary minus
    constexpr MatrixFieldCoeffWiseOp<Derived, int, std::multiplies<>> operator-() const { return -1 * derived(); }
    // matrix norm
    constexpr MatrixFieldNorm<2, Derived, 0> norm() const { return MatrixFieldNorm<2, Derived, 0>(derived()); }
    constexpr MatrixFieldNorm<2, Derived, 1> squared_norm() const { return MatrixFieldNorm<2, Derived, 1>(derived()); }
    template <int Order> constexpr MatrixFieldNorm<Order, Derived, 0> lp_norm() const {
        return MatrixFieldNorm<Order, Derived, 0>(derived());
    }
    // vector field divergence
    constexpr Divergence<Derived> divergence() const {
        fdapde_static_assert(Derived::Cols == 1, THIS_METHOD_IS_FOR_VECTOR_FIELDS_ONLY);
        return Divergence<Derived>(derived());
    }
    // vector field gradient (i.e., its jacobian matrix field)
    constexpr Jacobian<Derived> gradient() const {
        fdapde_static_assert(Derived::Cols == 1, THIS_METHOD_IS_FOR_VECTOR_FIELDS_ONLY);
        return Jacobian<Derived>(derived());
    }
    // dot product
    template <int RhsStaticInputSize, typename Rhs>
    constexpr DotProduct<Derived, Rhs> dot(const MatrixFieldBase<RhsStaticInputSize, Rhs>& rhs) const {
        return DotProduct<Derived, Rhs>(derived(), rhs.derived());
    }
    template <typename Rhs> DotProduct<Derived, Eigen::MatrixBase<Rhs>> dot(const Eigen::MatrixBase<Rhs>& rhs) const {
        return DotProduct<Derived, Eigen::MatrixBase<Rhs>>(derived(), rhs.derived());
    }
    // diagonal
    constexpr MatrixFieldDiagonalBlock<Derived> diagonal() const {
        return MatrixFieldDiagonalBlock<Derived>(derived());
    }
    // symmetric view
    template <int ViewMode> constexpr MatrixFieldSymmetricView<Derived, ViewMode> symmetric_view() const {
        return MatrixFieldSymmetricView<Derived, ViewMode>(derived());
    }
};
  
// integration with Eigen types (these expressions are never constexpr-enabled, since Eigen types are not)

namespace internals {

template <
  typename Lhs, typename Rhs, typename FieldType_ = std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, Rhs, Lhs>>
class matrix_field_eigen_product_impl :
    public MatrixFieldBase<FieldType_::StaticInputSize, matrix_field_eigen_product_impl<Lhs, Rhs>> {
    using FieldType = std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, Rhs, Lhs>;
    using EigenType = std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, Lhs, Rhs>;
    static constexpr bool is_field_lhs = std::is_same_v<FieldType, Lhs>;
    fdapde_static_assert(
      FieldType::Rows == Dynamic || FieldType::Cols == Dynamic || EigenType::RowsAtCompileTime == Dynamic ||
        EigenType::ColsAtCompileTime == Dynamic || (is_field_lhs && FieldType::Cols == EigenType::RowsAtCompileTime) ||
        (!is_field_lhs && EigenType::ColsAtCompileTime == FieldType::Rows),
      INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_PRODUCT);
   public:
    using LhsDerived = Lhs;
    using RhsDerived = Rhs;
    template <typename T1, typename T2> using Meta = matrix_field_eigen_product_impl<T1, T2>;
    using Base = MatrixFieldBase<FieldType::StaticInputSize, matrix_field_eigen_product_impl<Lhs, Rhs>>;
    using InputType = typename FieldType::InputType;
    using Scalar = decltype(std::declval<typename FieldType::Scalar>() * std::declval<typename EigenType::Scalar>());
    static constexpr int StaticInputSize = FieldType::StaticInputSize;
    static constexpr int Rows = is_field_lhs ? FieldType::Rows : EigenType::RowsAtCompileTime;
    static constexpr int Cols = is_field_lhs ? EigenType::ColsAtCompileTime : FieldType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = FieldType::XprBits;

    matrix_field_eigen_product_impl(const Lhs& lhs, const Rhs& rhs) : Base(), lhs_(lhs), rhs_(rhs) {
        if constexpr (
          FieldType::Rows == Dynamic || FieldType::Cols == Dynamic || EigenType::RowsAtCompileTime == Dynamic ||
          EigenType::ColsAtCompileTime == Dynamic) {
            fdapde_assert(lhs.cols() == rhs.rows());
        }
    }
    int rows() const { return lhs_.rows(); }
    int cols() const { return rhs_.cols(); }
    constexpr int input_size() const {
        if constexpr (is_field_lhs)  return lhs_.input_size();
        else return rhs_.input_size();
    }
    int size() const { return lhs_.rows() * rhs_.cols(); }
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }

    template <typename Dest> constexpr void eval_at(const InputType& p, Dest& dest) const {
        fdapde_static_assert(
          std::is_invocable_v<Dest FDAPDE_COMMA int FDAPDE_COMMA int> ||
            internals::is_subscriptable<Dest FDAPDE_COMMA int>,
          DESTINATION_TYPE_MUST_EITHER_EXPOSE_A_MATRIX_LIKE_ACCESS_OPERATOR_OR_A_SUBSCRIPT_OPERATOR);

        // evaluate Lhs field in temporary
        constexpr bool is_dynamic_storage = FieldType::Rows == Dynamic || FieldType::Cols == Dynamic ||
                                            EigenType::RowsAtCompileTime == Dynamic ||
                                            EigenType::ColsAtCompileTime == Dynamic;
        using FieldStorageType = std::conditional_t<
          is_dynamic_storage, Eigen::Matrix<Scalar, Dynamic, Dynamic>, Eigen::Matrix<Scalar, Rows, Cols>>;
        FieldStorageType field_;
	int rows_ = is_field_lhs ? lhs_.rows() : rhs_.rows();
	int cols_ = is_field_lhs ? rhs_.rows() : rhs_.cols();
        if constexpr (is_dynamic_storage) field_.resize(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
	        if constexpr(is_field_lhs) field_(i, j) = lhs_.eval(i, j, p);
	        else field_(i, j) = rhs_.eval(i, j, p);
	    }
        }
        // perform standard matrix-matrix product using Eigen implementation
        if constexpr (std::is_invocable_v<Dest, int, int>) {
	    if constexpr (is_field_lhs) dest = field_ * rhs_;
	    else dest = lhs_ * field_;
        } else {
            using ProductResultType = std::conditional_t<
              is_dynamic_storage, Eigen::Matrix<Scalar, Dynamic, Dynamic>, Eigen::Matrix<Scalar, Rows, Cols>>;
            Eigen::Map<ProductResultType> map(dest, rows(), cols());
	    if constexpr (is_field_lhs) map = field_ * rhs_;
	    else map = lhs_ * field_;
        }
    }
    auto operator()(int i, int j) const {
        return [i, j, this](const InputType& p) {
            Scalar res = 0;
            for (int k = 0; k < lhs_.cols(); ++k) {
	        if constexpr (is_field_lhs)  res += lhs_.eval(i, k, p) * rhs_(k, j);
	        else res += lhs_(i, k) * rhs_.eval(k, j, p);
	    }
            return res;
        };
    }
    constexpr Scalar eval(int i, int j, const InputType& p) const {
        Scalar res = 0;
        for (int k = 0; k < lhs_.cols(); ++k) {
	    if constexpr (is_field_lhs)  res += lhs_.eval(i, k, p) * rhs_(k, j);
	    else res += lhs_(i, k) * rhs_.eval(k, j, p);
	}
        return res;
    }
    constexpr Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        Scalar res = 0;
        for (int k = 0; k < size(); ++k) {
            if constexpr (Rows == 1) {
                if constexpr (is_field_lhs) res += lhs_.eval(k, p) * rhs_(k, i);
                else res += lhs_[k] * rhs_.eval(k, i, p);
            } else {
                if constexpr (is_field_lhs) res += lhs_.eval(i, k, p) * rhs_[k];
                else res += lhs_(i, k) * rhs_.eval(k, p);
            }
        }
        return res;
    }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   protected:
    std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, const Lhs, internals::ref_select_t<const Lhs>> lhs_;
    std::conditional_t<internals::is_eigen_dense_xpr_v<Rhs>, const Rhs, internals::ref_select_t<const Rhs>> rhs_;
};

template <
  typename Lhs, typename Rhs, typename BinaryOperation,
  typename FieldType_ = std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, Rhs, Lhs>>
class matrix_field_eigen_binary_op_impl :
    public MatrixFieldBase<FieldType_::StaticInputSize, matrix_field_eigen_binary_op_impl<Lhs, Rhs, BinaryOperation>> {
    using FieldType = std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, Rhs, Lhs>;
    using EigenType = std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, Lhs, Rhs>;
    static constexpr bool is_field_lhs = std::is_same_v<FieldType, Lhs>;
    fdapde_static_assert(
      FieldType::Rows == Dynamic || FieldType::Cols == Dynamic || EigenType::RowsAtCompileTime == Dynamic ||
        EigenType::ColsAtCompileTime == Dynamic ||
        (FieldType::Rows == EigenType::RowsAtCompileTime && FieldType::Cols == EigenType::ColsAtCompileTime),
      INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_PRODUCT);
   public:
    using LhsDerived = Lhs;
    using RhsDerived = Rhs;
    template <typename T1, typename T2> using Meta = matrix_field_eigen_binary_op_impl<T1, T2, BinaryOperation>;
    using Base =
      MatrixFieldBase<FieldType::StaticInputSize, matrix_field_eigen_binary_op_impl<Lhs, Rhs, BinaryOperation>>;
    using InputType = typename FieldType::InputType;
    using Scalar = decltype(std::declval<BinaryOperation>().operator()(
      std::declval<typename FieldType::Scalar>(), std::declval<typename EigenType::Scalar>()));
    static constexpr int StaticInputSize = FieldType::StaticInputSize;
    static constexpr int Rows = FieldType::Rows;
    static constexpr int Cols = FieldType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = FieldType::XprBits;

    matrix_field_eigen_binary_op_impl(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) :
        Base(), lhs_(lhs), rhs_(rhs), op_(op) {
        if constexpr (
          FieldType::Rows == Dynamic || FieldType::Cols == Dynamic || EigenType::RowsAtCompileTime == Dynamic ||
          EigenType::ColsAtCompileTime == Dynamic) {
            fdapde_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
        }
    }
    int rows() const { return lhs_.rows(); }
    int cols() const { return lhs_.cols(); }
    constexpr int input_size() const {
        if constexpr (is_field_lhs)  return lhs_.input_size();
        else return rhs_.input_size();
    }
    int size() const { return lhs_.size(); }
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }
  
    Scalar eval(int i, int j, const InputType& p) const {
        if constexpr(is_field_lhs) return op_(lhs_.eval(i, j, p), rhs_(i, j));
        else return op_(lhs_(i, j), rhs_.eval(i, j, p));
    }
    Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_MATRICES);
        if constexpr(is_field_lhs) return op_(lhs_.eval(i, p), rhs_[i]);
        else return op_(lhs_[i], rhs_.eval(i, p));
    }
    auto operator()(int i, int j) const {
        return [i, j, this](const InputType& p) {
            if constexpr (is_field_lhs) return op_(lhs_.eval(i, j, p), rhs_(i, j));
            else return op_(lhs_(i, j), rhs_.eval(i, j, p));
        };
    }
    Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_MATRICES);
        return [i, this](const InputType& p) {
            if constexpr (is_field_lhs) return op_(lhs_.eval(i, p), rhs_[i]);
            else return op_(lhs_[i], rhs_.eval(i, p));
        };
    }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   protected:
    std::conditional_t<internals::is_eigen_dense_xpr_v<Lhs>, const Lhs, internals::ref_select_t<const Lhs>> lhs_;
    std::conditional_t<internals::is_eigen_dense_xpr_v<Rhs>, const Rhs, internals::ref_select_t<const Rhs>> rhs_;
    BinaryOperation op_;
};

}   // namespace internals

// Eigen matrix - matrix field product
template <typename Lhs, typename Rhs>
struct MatrixFieldProduct<Lhs, Eigen::MatrixBase<Rhs>> : public internals::matrix_field_eigen_product_impl<Lhs, Rhs> {
    MatrixFieldProduct(const Lhs& lhs, const Rhs& rhs) :
        internals::matrix_field_eigen_product_impl<Lhs, Rhs>(lhs, rhs) { }
};
template <typename Lhs, typename Rhs>
struct MatrixFieldProduct<Eigen::MatrixBase<Lhs>, Rhs> : public internals::matrix_field_eigen_product_impl<Lhs, Rhs> {
    MatrixFieldProduct(const Lhs& lhs, const Rhs& rhs) :
        internals::matrix_field_eigen_product_impl<Lhs, Rhs>(lhs, rhs) { }
};
  
template <typename Lhs, typename Rhs>
MatrixFieldProduct<Lhs, Eigen::MatrixBase<Rhs>>
operator*(const MatrixFieldBase<Lhs::StaticInputSize, Lhs>& lhs, const Eigen::MatrixBase<Rhs>& rhs) {
    return MatrixFieldProduct<Lhs, Eigen::MatrixBase<Rhs>> {lhs.derived(), rhs.derived()};
}
template <typename Lhs, typename Rhs>
MatrixFieldProduct<Eigen::MatrixBase<Lhs>, Rhs>
operator*(const Eigen::MatrixBase<Lhs>& lhs, const MatrixFieldBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return MatrixFieldProduct<Eigen::MatrixBase<Lhs>, Rhs> {lhs.derived(), rhs.derived()};
}

// Eigen matrix - matrix field binary addition and subtraction
template <typename Lhs, typename Rhs, typename BinaryOperation>
struct MatrixFieldBinOp<Lhs, Eigen::MatrixBase<Rhs>, BinaryOperation> :
    public internals::matrix_field_eigen_binary_op_impl<Lhs, Rhs, BinaryOperation> {
    MatrixFieldBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) :
        internals::matrix_field_eigen_binary_op_impl<Lhs, Rhs, BinaryOperation>(lhs, rhs, op) { }
};
template <typename Lhs, typename Rhs, typename BinaryOperation>
struct MatrixFieldBinOp<Eigen::MatrixBase<Lhs>, Rhs, BinaryOperation> :
    public internals::matrix_field_eigen_binary_op_impl<Lhs, Rhs, BinaryOperation> {
    MatrixFieldBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) :
        internals::matrix_field_eigen_binary_op_impl<Lhs, Rhs, BinaryOperation>(lhs, rhs, op) { }
};

#define FDAPDE_DEFINE_FIELD_EIGEN_BIN_OP(OPERATOR, FUNCTOR)                                                            \
    template <typename Lhs, typename Rhs>                                                                              \
    MatrixFieldBinOp<Lhs, Eigen::MatrixBase<Rhs>, FUNCTOR> OPERATOR(                                                   \
      const MatrixFieldBase<Lhs::StaticInputSize, Lhs>& lhs, const Eigen::MatrixBase<Rhs>& rhs) {                      \
        return MatrixFieldBinOp<Lhs, Eigen::MatrixBase<Rhs>, FUNCTOR> {lhs.derived(), rhs.derived(), FUNCTOR()};       \
    }                                                                                                                  \
    template <typename Lhs, typename Rhs>                                                                              \
    MatrixFieldBinOp<Eigen::MatrixBase<Lhs>, Rhs, FUNCTOR> OPERATOR(                                                   \
      const Eigen::MatrixBase<Lhs>& lhs, const MatrixFieldBase<Rhs::StaticInputSize, Rhs>& rhs) {                      \
        return MatrixFieldBinOp<Eigen::MatrixBase<Lhs>, Rhs, FUNCTOR> {lhs.derived(), rhs.derived(), FUNCTOR()};       \
    }
FDAPDE_DEFINE_FIELD_EIGEN_BIN_OP(operator+, std::plus<> )
FDAPDE_DEFINE_FIELD_EIGEN_BIN_OP(operator-, std::minus<>)

// integration with cexpr linear algebra
  
}   // namespace fdapde

#endif   // __FDAPDE_MATRIX_FIELD_H__
