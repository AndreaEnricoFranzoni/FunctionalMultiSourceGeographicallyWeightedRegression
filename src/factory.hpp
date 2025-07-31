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


#ifndef FDAGWR_FACTORY_HPP
#define FDAGWR_FACTORY_HPP


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include <sstream>
#include <stdexcept>


namespace generic_factory{

  /**
  * A generic factory.
  * It is implemented as a Singleton. The compulsory way to
  * access a method is Factory::Instance().method().
  * Typycally to access the factory one does
  * \code
  * auto&  myFactory = Factory<A,I,B>::Instance();
  * myFactory.add(...)
  * \endcode
  */

  template <
            typename AbstractProduct, 
            typename Identifier, 
            typename Builder = std::function<std::unique_ptr<AbstractProduct>()>>
  class Factory{

  public:
    /*
    * The container for the rules.
    */
    using AbstractProduct_type = AbstractProduct;

    /*
    * The identifier.
    * We must have an ordering since we use a map with
    * the identifier as key.
    */
    using Identifier_type = Identifier;

    /* The builder type. Must be a callable object
    * The default is a function.
    */
    using Builder_type = Builder;

    /*
    * Method to access the only instance of the factory. We use Meyer's trick to istantiate the factory.
    */
    static Factory & Instance();

    /* Get the rule with given name
    * The pointer is null if no rule is present.
    */
    template<typename... Args>
    std::unique_ptr<AbstractProduct> create(Identifier const & name, Args&&... args) const;

    /*
    * Register the given rule.
    */
    void add(Identifier const &, Builder_type const &);

    /*
    * Returns a list of registered rules.
    */
    std::vector<Identifier> registered()const;

    /*
    * Unregister a rule.
    */
    void unregister(Identifier const & name){ _storage.erase(name);}

    /*
    * Default destructor.
    */
    ~Factory() = default;

  private:
    /**
    * \var typedef std::map<Identifier,Builder_type> Container_type
    * Type of the object used to store the object factory
    */
    typedef std::map<Identifier_type,Builder_type> Container_type;

    /**
    * Constructor made private since it is a Singleton
    */
    Factory() = default;

    /**
    * Copy constructor deleted since it is a Singleton
    */
    Factory(Factory const &) = delete;

    /**
    * Assignment operator deleted since it is a Singleton
    */
    Factory & operator =(Factory const &) = delete;

    /**
    * It contains the actual object factory.
    */
    Container_type _storage;
  };



  template <typename AbstractProduct, typename Identifier, typename Builder>
  Factory<AbstractProduct,Identifier,Builder> &
  Factory<AbstractProduct,Identifier,Builder>::Instance() {
    static Factory theFactory;
    return theFactory;
  }


  template <typename AbstractProduct, typename Identifier, typename Builder>
  template <typename... Args>
  std::unique_ptr<AbstractProduct>
  Factory<AbstractProduct,Identifier,Builder>::create(Identifier const & name,
                                                      Args &&...args) const {

    auto f = _storage.find(name); //C++11
    if (f == _storage.end()) {
	     std::string out="Identifier " + name + " is not stored in the factory";
	      throw std::invalid_argument(out);
    }
    else {
	       //return std::unique_ptr<AbstractProduct>(f->second(std::forward<Args>(args)...));
         return f->second(std::forward<Args>(args)...);
    }
  }

  template <typename AbstractProduct, typename Identifier, typename Builder>
  void
  Factory<AbstractProduct,Identifier,Builder>::add(Identifier const & name, Builder_type const & func){

    auto f =  _storage.insert(std::make_pair(name, func));
    if (f.second == false)
    throw std::invalid_argument("Double registration in Factory");
  }


  template <typename AbstractProduct, typename Identifier, typename Builder>
  std::vector<Identifier>
  Factory<AbstractProduct,Identifier,Builder>::registered() const {
    std::vector<Identifier> tmp;
    tmp.reserve(_storage.size());
    for(auto i=_storage.begin(); i!=_storage.end();++i)
      tmp.push_back(i->first);
    return tmp;
  }

}

#endif /* FDAGWR_FACTORY_HPP */