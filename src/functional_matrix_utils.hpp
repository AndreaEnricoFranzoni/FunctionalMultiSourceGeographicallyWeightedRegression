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


#ifndef FUNCTIONAL_MATRIX_UTILS_HPP
#define FUNCTIONAL_MATRIX_UTILS_HPP


#include <functional>
#include <type_traits>


namespace fm_utils
{

// Traits for extracting the function type
template <typename T>
struct function_traits;

// Specialization for function "R(Arg)"
template <typename R, typename Arg>
struct function_traits<R(Arg)> {
    using output_type = R;
    using input_param_type = Arg;
    using input_type = std::remove_cv_t<std::remove_reference_t<Arg>>;
};

// Specialization for pointer to function
template <typename R, typename Arg>
struct function_traits<R(*)(Arg)> : function_traits<R(Arg)> {};
//Specialization for std::function
template <typename R, typename Arg>
struct function_traits<std::function<R(Arg)>> : function_traits<R(Arg)> {};
// Specialization for member function
template <typename C, typename R, typename Arg>
struct function_traits<R(C::*)(Arg)> : function_traits<R(Arg)> {};
// Specialization for member function (const version)
template <typename C, typename R, typename Arg>
struct function_traits<R(C::*)(Arg) const> : function_traits<R(Arg)> {};
// Specialization for lambda/functor
template<typename T>
struct function_traits : function_traits<decltype(&T::operator())> {};


// ---------- Helpers ----------
template <typename F>
using input_t = typename function_traits<F>::input_type;

template <typename F>
using input_param_t = typename function_traits<F>::input_param_type;

template <typename F>
using output_t = typename function_traits<F>::output_type;

}   //end namespace fm_utils

#endif  /*FUNCTIONAL_MATRIX_UTILS_HPP*/




/*
int foo(const std::string&);   // solo dichiarata!

int main() {
    // 1. std::function
    using F1 = std::function<int(const std::string&)>;
    static_assert(std::is_same_v<input_type_t<F1>, std::string>);
    static_assert(std::is_same_v<param_type_t<F1>, const std::string&>);
    static_assert(std::is_same_v<output_type_t<F1>, int>);

    // 2. puntatore a funzione
    using F2 = decltype(&foo);
    static_assert(std::is_same_v<input_type_t<F2>, std::string>);
    static_assert(std::is_same_v<param_type_t<F2>, const std::string&>);
    static_assert(std::is_same_v<output_type_t<F2>, double> == false); // foo ritorna int, non double

    // 3. lambda
    auto lambda = [](const double& d) { return d * 2.0; };
    using F3 = decltype(lambda);
    static_assert(std::is_same_v<input_type_t<F3>, double>);
    static_assert(std::is_same_v<param_type_t<F3>, const double&>);
    static_assert(std::is_same_v<output_type_t<F3>, double>);

    std::cout << "Tutti i test passati!\n";
}
*/
