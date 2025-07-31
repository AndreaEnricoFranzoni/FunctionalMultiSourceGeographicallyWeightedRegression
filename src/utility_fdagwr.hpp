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


#ifndef FDAGWR_UTILITIES_HPP
#define FDAGWR_UTILITIES_HPP


#include "traits_fdagwr.hpp"

#include <memory>
#include <type_traits>

// Generico: gestisce qualsiasi template <typename...> class
template <typename T>
struct extract_template;

// Pattern matching su classi template con qualsiasi numero di parametri di tipo
template <template <typename...> class Template, typename... Args>
struct extract_template<Template<Args...>> {
    using type = Template; // Nome del template puro
};

// Helper alias
template <typename T>
using extract_template_t = typename extract_template<
    typename std::remove_cv_t<  // rimuove const/volatile
        typename std::remove_reference_t<T> // rimuove ref
    >
>::type;


/*
template <typename Domain>
struct basis {};

int main() {
    std::unique_ptr<basis<int>> ptr;

    using PointeeType = typename decltype(ptr)::element_type; // basis<int>
    using BasisTemplate = extract_template_t<PointeeType>;    // basis

    // Ora puoi fare:
    wrapper<BasisTemplate> w;
}

*/



#endif  /*FDAGWR_UTILITIES_HPP*/