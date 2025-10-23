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


#ifndef FDAGWR_FACTORY_HPP
#define FDAGWR_FACTORY_HPP


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"
#include <sstream>
#include <stdexcept>


/*!
* @file factory.hpp
* @brief Contains a generic factory definition
* @author Andrea Enrico Franzoni
*/


/*!
* @namespace generic_factory
* @brief Contains elements to define a generic factory
*/
namespace generic_factory{

  /*!
  * @class Factory
  * @brief A generic factory, implemented as a Singleton. The compulsory way to access a method is Factory::Instance().method()
  * @tparam AbstractProduct the base object
  * @tparam Identifier the type of the identifier
  * @tparam Builder the type of the builder
  * @code
  * auto&  myFactory = Factory<A,I,B>::Instance();  //to access the factory
  * myFactory.add(...)
  * @endcode 
  */
  template <
            typename AbstractProduct, 
            typename Identifier, 
            typename Builder = std::function<std::unique_ptr<AbstractProduct>()>>
  class Factory{

  public:
    /*!
    * @brief The container for the rules.
    */
    using AbstractProduct_type = AbstractProduct;

    /*!
    * @brief The identifier.
    * @note It must have an ordering since the identifier is used as key for a map
    */
    using Identifier_type = Identifier;

    /*!
    * @brief The builder type. 
    * @note Must be a callable object. The default is a function.
    */
    using Builder_type = Builder;

    /*!
    * @brief Method to access the only instance of the factory (Meyer's trick to istantiate the factory)
    */
    static Factory & Instance();

    /*! 
    * @brief Getting the rule with given name. The pointer is null if no rule is present.
    * @tparam Args variadic templates for the constructor input
    * @param name identifier
    * @param args variadic inputs for the constructors
    * @return a unique pointer to the abstract base object
    */
    template<typename... Args>
    std::unique_ptr<AbstractProduct> create(Identifier const & name, Args&&... args) const;

    /*!
    * @brief Registering the given rule
    */
    void add(Identifier const &, Builder_type const &);

    /*!
    * @brief Returns a list of registered rules
    * @return a vector with the identifiers registered
    */
    std::vector<Identifier> registered()const;

    /*!
    * @brief Unregister a rule
    * @param name rule to be unregistered
    */
    void unregister(Identifier const & name){ _storage.erase(name);}

    /*!
    * @brief Default destructor.
    */
    ~Factory() = default;


  private:
    /*!
    * @typedef Container_type
    * @brief Type of the object used to store the object factory
    */
    typedef std::map<Identifier_type,Builder_type> Container_type;

    /*!
    * @brief Constructor 
    * @note made private since it is a Singleton
    */
    Factory() = default;

    /*!
    * @brief Copy constructor 
    * @note deleted since it is a Singleton
    */
    Factory(Factory const &) = delete;

    /*!
    * @brief Assignment operator 
    * @note deleted since it is a Singleton
    */
    Factory & operator =(Factory const &) = delete;

    /*!
    * @brief Actual object factory.
    */
    Container_type _storage;
  };



  /*!
  * @brief Getting the only instance of a factory
  * @tparam AbstractProduct the base object
  * @tparam Identifier the type of the identifier
  * @tparam Builder the type of the builder
  * @return the instance of the factory
  * @note static since it is a singleton
  */
  template <typename AbstractProduct, typename Identifier, typename Builder>
  Factory<AbstractProduct,Identifier,Builder> &
  Factory<AbstractProduct,Identifier,Builder>::Instance() {
    static Factory theFactory;
    return theFactory;
  }


  /*!
  * @brief Create one of the registered objects
  * @tparam AbstractProduct the base object
  * @tparam Identifier the type of the identifier
  * @tparam Builder the type of the builder
  * @tparam Args variadic template
  * @param name the identifier
  * @param args variadic inputs for the concrete object constructor
  * @return an unique pointer to the concrete object
  * @note it throws an exception if the object requested is not registered
  */
  template <typename AbstractProduct, typename Identifier, typename Builder>
  template <typename... Args>
  std::unique_ptr<AbstractProduct>
  Factory<AbstractProduct,Identifier,Builder>::create(Identifier const & name,
                                                      Args &&...args) 
  const 
  {

    auto f = _storage.find(name); //C++11
    //object not registered: exception
    if (f == _storage.end()) 
    {
	     std::string out="Identifier " + name + " is not stored in the factory";
	      throw std::invalid_argument(out);
    }
    else 
    {
         return f->second(std::forward<Args>(args)...);
    }
  }


  /*!
  * @brief Registering a new object in the factory
  * @tparam AbstractProduct the base object
  * @tparam Identifier the type of the identifier
  * @tparam Builder the type of the builder
  * @param name the identifier of the new object
  * @param func builder for the new object
  * @note it throws an exception if the object requested is already registered
  */
  template <typename AbstractProduct, typename Identifier, typename Builder>
  void
  Factory<AbstractProduct,Identifier,Builder>::add(Identifier const & name, Builder_type const & func)
  {
    auto f =  _storage.insert(std::make_pair(name, func));
    if (f.second == false)
    throw std::invalid_argument("Double registration in Factory");
  }


  /*!
  * @brief The registered objects
  * @tparam AbstractProduct the base object
  * @tparam Identifier the type of the identifier
  * @tparam Builder the type of the builder
  * @return a vector with the identifiers of the already registered objects
  * @note it throws an exception if the object requested is already registered
  */
  template <typename AbstractProduct, typename Identifier, typename Builder>
  std::vector<Identifier>
  Factory<AbstractProduct,Identifier,Builder>::registered() 
  const 
  {
    std::vector<Identifier> tmp;
    tmp.reserve(_storage.size());
    for(auto i=_storage.begin(); i!=_storage.end();++i)
      tmp.push_back(i->first);
    return tmp;
  }

}   //end namespace generic_factory

#endif /* FDAGWR_FACTORY_HPP */