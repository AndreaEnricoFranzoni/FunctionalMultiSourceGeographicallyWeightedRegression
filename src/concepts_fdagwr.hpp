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


#ifndef FDAGWR_BASIS_CONCEPTS_HPP
#define FDAGWR_BASIS_CONCEPTS_HPP


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"

namespace fdagwr_concepts
{

//concept for saying that the interval used to define the basis derives from the triangulation geometry in fdapde
//it is necessary to remove the const ref since, for fdaPDE workflow, knots are returned as const ref
template <typename T>
concept as_interval = std::derived_from<std::remove_cvref_t<T>, fdapde::TriangulationBase<1,1,fdapde::Triangulation<1,1>>>;



//concept for the basis type
template<typename T>
concept as_basis = requires(T x) {
  
  {x.knots()} -> as_interval;                                        //knots of the basis have to be as the class described in fdaPDE
  {x.eval_base(0.0)} -> std::same_as<FDAGWR_TRAITS::Dense_Matrix>;   //function to perform the evaluation
};
}


#endif  /*FDAGWR_BASIS_CONCEPTS_HPP*/