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

#ifndef __FDAPDE_DIVERGENCE_H__
#define __FDAPDE_DIVERGENCE_H__

#include "header_check.h"

namespace fdapde {

template <typename Derived_>
class Divergence : public ScalarFieldBase<Derived_::StaticInputSize, Divergence<Derived_>> {
    fdapde_static_assert(
      Derived_::Cols == 1 && (Derived_::StaticInputSize == Dynamic || Derived_::StaticInputSize == Derived_::Rows),
      DIVERGENCE_OPERATOR_IS_FOR_VECTOR_FIELDS_ONLY);  
   public:
    using Derived = std::decay_t<Derived_>;
    template <typename T> using Meta = Divergence<T>;
    using Base = ScalarFieldBase<Derived::StaticInputSize, Divergence<Derived>>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    using FunctorType = PartialDerivative<
      internals::xpr_or_scalar_wrap_t<
        Derived, StaticInputSize, std::decay_t<decltype(std::declval<Derived>().operator[](std::declval<int>()))>>,
      1>;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = FunctorType::XprBits;

    explicit constexpr Divergence(const Derived_& xpr) : Base(), data_(), xpr_(xpr) {
        if constexpr (StaticInputSize == Dynamic) {
            fdapde_constexpr_assert(xpr_.input_size() == xpr_.rows());
            data_.resize(xpr_.rows());
        }
        for (int i = 0; i < xpr_.rows(); ++i) { data_[i] = FunctorType(xpr_[i], i); }
    }
    constexpr Scalar operator()(const InputType& p) const {
        Scalar div_ = 0;
        for (int i = 0; i < xpr_.rows(); ++i) { div_ += data_[i](p); }
        return div_;
    }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr const Derived& derived() const { return xpr_; }
   private:
    using StorageType = typename std::conditional_t<
      Derived::StaticInputSize == Dynamic, std::vector<FunctorType>, std::array<FunctorType, StaticInputSize>>;
    StorageType data_;
    internals::ref_select_t<const Derived> xpr_;
};

template <typename XprType> Divergence<XprType> constexpr div(const XprType& xpr) { return Divergence<XprType>(xpr); }

}   // namespace fdapde

#endif // __FDAPDE_DIVERGENCE_H__
