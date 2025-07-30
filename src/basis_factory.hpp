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


#ifndef FDAGWR_BASIS_FACTORY_HPP
#define FDAGWR_BASIS_FACTORY_HPP

#include "traits_fdagwr.hpp"
#include "concepts_fdagwr.hpp"
#include "basis_include.hpp"
#include "factory.hpp"
#include "factory_proxy.hpp"



namespace basis_factory{

    //identifier for the factory
    using basisIdentifier = std::string;

    //builder
    using basisBuilder = std::function<std::unique_ptr<basis_base_class<fdagwr_traits::Domain>>(const fdagwr_traits::Dense_Vector &, std::size_t, std::size_t)>;


    /**
    * \var typedef generic_factory::Factory<TailUpModel, std::string> TailUpFactory;
    * Factory for the tail-up model
    */
    typedef generic_factory::Factory<basis_base_class<fdagwr_traits::Domain>, basisIdentifier, basisBuilder> basisFactory;  // Use standard Builder

    /**
    * Proxy for the tail-up model
    */
    template<typename ConcreteProduct>
    using basisProxy = generic_factory::Proxy<basisFactory,ConcreteProduct>;

}   //end namespace basis_factory




/*

//general builder for the factory
template<typename domain_type>
    requires fdagwr_concepts::as_interval<domain_type>
using basisBuilder = std::function<std::unique_ptr<basis_base_class<domain_type>>(const fdagwr_traits::Dense_Vector &, std::size_t, std::size_t)>;

//factory
template<typename domain_type>
    requires fdagwr_concepts::as_interval<domain_type>
using basisFactory = GenericFactory::Factory<basis_base_class<domain_type>, basisIdentifier, basisBuilder<domain_type>>;

//builders: a builder for each one of the implemented basis
//bsplines
template<typename domain_type>
    requires fdagwr_concepts::as_interval<domain_type>
basisBuilder<domain_type> build_bsplines = [] (const fdagwr_traits::Dense_Vector &knots, std::size_t dg, std::size_t nb) { return std::make_unique<bsplines_basis<domain_type>>(knots,dg,nb);};
//constant basis
template<typename domain_type>
    requires fdagwr_concepts::as_interval<domain_type>
basisBuilder<domain_type> build_constant = [] (const fdagwr_traits::Dense_Vector &knots, std::size_t dg, std::size_t nb) { return std::make_unique<constant_basis<domain_type>>(knots,dg,nb);};

//loading the factory
void loadBasis(){    auto &basis_factory = basisFactory<fdagwr_traits::Domain>::Instance();}
namespace   //registering each time a new implemented basis type
{
GenericFactory::Proxy<basisFactory<fdagwr_traits::Domain>, bsplines_basis<fdagwr_traits::Domain>> bsplines_basis_obj{FDAGWR_BASIS_TYPES::_bsplines_, build_bsplines<fdagwr_traits::Domain>};
GenericFactory::Proxy<basisFactory<fdagwr_traits::Domain>, constant_basis<fdagwr_traits::Domain>> constant_basis_obj{FDAGWR_BASIS_TYPES::_constant_, build_constant<fdagwr_traits::Domain>};
}

*/


/*
template<typename domain_type, class... Args>
    requires fdagwr_concepts::as_interval<domain_type>
std::unique_ptr<basis_base_class<domain_type>> baseFactory(const std::string& basis_id, Args&&... args)
{
    if(basis_id == FDAGWR_BASIS_TYPES::_bsplines_){
        return std::make_unique<bsplines_basis<domain_type>>(std::forward<Args>(args)...);}
    
    if(basis_id == FDAGWR_BASIS_TYPES::_constant_ ){
        return std::make_unique<constant_basis<domain_type>>(std::forward<Args>(args)...); }

    else{
        return std::unique_ptr<basis_base_class<domain_type>>();}
}
*/

#endif  /*FDAGWR_BASIS_FACTORY_HPP*/