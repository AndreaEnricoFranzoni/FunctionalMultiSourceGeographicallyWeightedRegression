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


/*
// Trait generico per std::function
template <typename T>
struct function_traits;

// Specializzazione per std::function
template <typename R, typename... Args>
struct function_traits<std::function<R(Args...)>> {
    using return_type = R;
    using argument_types = std::tuple<Args...>;

    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t N>
    using argument = std::tuple_element_t<N, std::tuple<Args...>>;
};

#include <iostream>

int main() {
    std::function<double(int, float)> f;

    using traits = function_traits<decltype(f)>;

    using ret_type = traits::return_type;          // double
    using args_tuple = traits::argument_types;     // std::tuple<int, float>
    using first_arg = traits::argument<0>;         // int

    static_assert(std::is_same_v<ret_type, double>);
    static_assert(std::is_same_v<first_arg, int>);

    std::cout << "Arity: " << traits::arity << "\n"; // Output: 2
}

*/







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
    static std::unique_ptr<AbstractProduct_type> Build(const fdagwr_traits::Dense_Vector &m, std::size_t a, std::size_t b){ return std::make_unique<ConcreteProduct>(m,a,b);}

    

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
