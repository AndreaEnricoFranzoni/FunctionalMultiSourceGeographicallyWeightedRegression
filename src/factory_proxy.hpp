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

#ifndef FDAGWR_FACTORY_PROXY_HPP
#define FDAGWR_FACTORY_PROXY_HPP


#include "traits_fdagwr.hpp"


namespace generic_factory {

  /**
  * A simple proxy for registering into a factory.
  * It provides the builder as static method
  * and the automatic registration mechanism.
  * \param Factory The type of the factory.
  * \param ConcreteProduct Is the derived (concrete) type to be
  * registered in the factory
  * @note I have to use the default builder provided by the factory. No check is made to verify it
  */

  template <typename Factory, typename ConcreteProduct>
  class Proxy {
  public:
    /**
    * \var typedef typename  Factory::AbstractProduct_type AbstractProduct_type
    * Container for the rules.
    */
    typedef typename  Factory::AbstractProduct_type AbstractProduct_type;

    /**
    * \var typedef typename  Factory::Identifier_type Identifier_type
    * Identifier.
    */
    typedef typename  Factory::Identifier_type Identifier_type;

    /**
    * \var typedef typename  Factory::Builder_type Builder_type
    * Builder type.
    */
    typedef typename  Factory::Builder_type Builder_type;

    /**
    * \var typedef Factory Factory_type
    * Factory type.
    */
    typedef Factory Factory_type;

    /**
    * Constructor for the registration.
    */
    Proxy(Identifier_type const &);

    /**
    * Builder.
    */
    static std::unique_ptr<AbstractProduct_type> Build(){ return std::make_unique<ConcreteProduct>();}

  private:
    /**
    * Copy onstructor deleted since it is a Singleton
    */
    Proxy(Proxy const &)=delete;

    /**
    * Assignment operator deleted since it is a Singleton
    */
    Proxy & operator=(Proxy const &)=delete;
  };


  template<typename F, typename C>
  Proxy<F,C>::Proxy(Identifier_type const & name) {
    // get the factory. First time creates it.
    Factory_type & factory(Factory_type::Instance());
    // Insert the builder. The & is not needed.
    factory.add(name,&Proxy<F,C>::Build);
    // std::cout<<"Added "<< name << " to factory"<<std::endl;
  }
}

#endif /*FDAGWR_FACTORY_PROXY_HPP*/
