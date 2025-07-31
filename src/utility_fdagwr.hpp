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



// Primaria: vuota
template <typename T>
struct extract_template;

// Specializzazione generica per template con soli parametri di tipo
template <template <typename...> class TT, typename... Args>
struct extract_template<TT<Args...>> {
    // Alias template che ricrea il template di partenza
    template <typename... Ts>
    using template_type = TT<Ts...>;
};

// Helper per normalizzare il tipo (toglie const/ref)
template <typename T>
using extract_template_t = extract_template<
    std::remove_cv_t<std::remove_reference_t<T>>
>;






/*
template <typename Domain>
struct basis {};

template <template <typename> class BasisTemplate>
struct wrapper {};

int main() {
    std::unique_ptr<basis<int>> ptr;

    using PointeeType   = typename decltype(ptr)::element_type; // basis<int>
    using Extracted     = extract_template_t<PointeeType>;
    using BasisTemplate = Extracted::template_type; // <— qui è un alias template

    wrapper<BasisTemplate> w; // ✅ funziona
}

*/



#endif  /*FDAGWR_UTILITIES_HPP*/