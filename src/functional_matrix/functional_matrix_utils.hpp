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
// OUT OF OR IN CONNECTION WITH fdagwr OR THE USE OR OTHER DEALINGS IN
// fdagwr.


#ifndef FUNCTIONAL_MATRIX_UTILS_HPP
#define FUNCTIONAL_MATRIX_UTILS_HPP


#include <functional>
#include <type_traits>
#include <concepts>
#include <cstddef>



/*!
* @file functional_matrix_utils.hpp
* @brief Contains some utilities for using functional matrices: aliases for the type of the stored functions and their input, concepts to avoid overloading for functional matrices operators
* @author Andrea Enrico Franzoni
*/



/*!
* @namespace fm_utils
* @brief Namespace containing aliases for the type of the stored functions and their input, and concepts to avoid overloading for functional matrices operators
*/
namespace fm_utils
{


/*!
* @struct function_traits
* @tparam T
* @brief traits for extracting a function type from a callable object definition
*/
template <typename T>
struct function_traits;

/*!
* @struct function_traits
* @tparam R output type
* @tparam Arg input type
* @brief Specialization of function_traits for calla object<R(Arg)>
*/
template <typename R, typename Arg>
struct function_traits<R(Arg)> {

    /*!Alias for output type*/
    using output_type = R;
    /*!Alias for input type, considering eventually the references*/
    using input_param_type = Arg;
    /*!Alias for input type, without const reference*/
    using input_type = std::remove_cv_t<std::remove_reference_t<Arg>>;
};

/*!
* @tparam R output type
* @tparam Arg input type
* @brief specialization for pointer to function
*/
template <typename R, typename Arg>
struct function_traits<R(*)(Arg)> : function_traits<R(Arg)> {};

/*!
* @tparam R output type
* @tparam Arg input type
* @brief specialization for std::function
*/
template <typename R, typename Arg>
struct function_traits<std::function<R(Arg)>> : function_traits<R(Arg)> {};

/*!
* @tparam C class
* @tparam R output type
* @tparam Arg input type
* @brief specialization for member function of C (non-const version) 
*/
template <typename C, typename R, typename Arg>
struct function_traits<R(C::*)(Arg)> : function_traits<R(Arg)> {};

/*!
* @tparam C class
* @tparam R output type
* @tparam Arg input type
* @brief specialization for member function of C (const version)
*/
template <typename C, typename R, typename Arg>
struct function_traits<R(C::*)(Arg) const> : function_traits<R(Arg)> {};

/*!
* @tparam R output type
* @tparam Arg input type
* @brief specialization for lambda/functor
*/
template<typename T>
struct function_traits : function_traits<decltype(&T::operator())> {};


/*!
* @tparam F callable object
* @brief extracting the type of the input in the callable object F, not considering const references
*/
template <typename F>
using input_t = typename function_traits<F>::input_type;

/*!
* @tparam F callable object
* @brief extracting the type of the input in the callable object F, considering eventual const references
*/
template <typename F>
using input_param_t = typename function_traits<F>::input_param_type;

/*!
* @tparam F callable object
* @brief extracting the type of the output in the callable object F
*/
template <typename F>
using output_t = typename function_traits<F>::output_type;



/*!
* @brief Concept to avoid the overloading with the Eigen overloading if calling within functional matrices operators
* @tparam T the type of the object to be checked that does not derive from an Eigen matrix object
* @note the concept check that an object does not derive from an Eigen matrix
*/
template <typename T>
concept not_eigen = !std::is_base_of_v<Eigen::MatrixBase<T>,T>;

/*!
* @brief Concept to detect objects that behave like functional matrices expressions used in this library
*        (i.e., provide rows(), cols(), size(), and element access via operator()(i,j)).
* @tparam T the type of the object to be checked that is a functional_matrix object
* @note  Tightening operator overload participation to these types avoids hijacking STL iterator arithmetic
*/
template <typename T>
concept functional_matrix_like = requires(const T& t) {
    { t.rows() } -> std::convertible_to<std::size_t>;
    { t.cols() } -> std::convertible_to<std::size_t>;
    { t.size() } -> std::convertible_to<std::size_t>;
    { t(std::size_t{}, std::size_t{}) };
} && not_eigen<T>;

}   //end namespace fm_utils

#endif  /*FUNCTIONAL_MATRIX_UTILS_HPP*/