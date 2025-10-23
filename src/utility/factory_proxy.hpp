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

#ifndef FDAGWR_FACTORY_PROXY_HPP
#define FDAGWR_FACTORY_PROXY_HPP


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"


/*!
* @file factory_proxy.hpp
* @brief Contains a generic proxy for registering element into a factory
* @author Andrea Enrico Franzoni
*/



/*!
* @namespace generic_factory
* @brief Contains elements to define a generic factory
*/
namespace generic_factory 
{

  /*!
  * @class Proxy
  * @brief A proxy for registering into a factory. It provides the builder as static method and the automatic registration mechanism
  * @tparam Factory the type of the factory
  * @tparam ConcreteProduct it is the derived (concrete) type to be registered in the factory
  * @note the default builder provided by the factory has to be used. However, no check it is made to verify it
  */
  template <typename Factory, typename ConcreteProduct>
  class Proxy 
  {

  public:

    /*!
    * @typedef AbstractProduct_type
    * @brief Container for the rules
    */
    typedef typename  Factory::AbstractProduct_type AbstractProduct_type;

    /*!
    * @typedef Identifier_type
    * @brief Identifier
    */
    typedef typename  Factory::Identifier_type Identifier_type;

    /*!
    * @typedef Builder_type
    * @brief Builder type
    */
    typedef typename  Factory::Builder_type Builder_type;

    /*!
    * @typedef Factory_type
    * @brief Factory type.
    */
    typedef Factory Factory_type;

    /*!
    * @brief Constructor for the registration.
    */
    Proxy(Identifier_type const &);

    /*!
    * @brief Builder: it has to construct the basis. Static method
    * @param m dense vector with the knots
    * @param a left extreme domain
    * @param b right extreme domain
    * @return an unique pointer to the concrete object to be construct by the factory
    * @note the input parameters of this method have to comply with the ones of the constructor: all the constructors of the object constructed by the facotry have to be equal
    */
    static std::unique_ptr<AbstractProduct_type> Build(const FDAGWR_TRAITS::Dense_Vector &m, std::size_t a, std::size_t b){ return std::make_unique<ConcreteProduct>(m,a,b);}

    
  private:
    /*!
    * @brief Copy onstructor
    * @note deleted since it is a Singleton
    */
    Proxy(Proxy const &)=delete;

    /*!
    * @brief Assignment operator 
    * @note deleted since it is a Singleton
    */
    Proxy & operator=(Proxy const &)=delete;
  };


  /*!
  * @brief Proxy object: singleton used to register into the factory
  * @tparam F the factory
  * @tparam C the concrete product
  * @param name the identifier of the object to be registered
  */
  template<typename F, typename C>
  Proxy<F,C>::Proxy(Identifier_type const &name) 
  {
    //get the factory (if first time, it creates it)
    Factory_type & factory(Factory_type::Instance());
    //Insert the builder: registration. The & is not needed.
    factory.add(name,&Proxy<F,C>::Build);
  }

}   //end namespace generic_factory

#endif /*FDAGWR_FACTORY_PROXY_HPP*/