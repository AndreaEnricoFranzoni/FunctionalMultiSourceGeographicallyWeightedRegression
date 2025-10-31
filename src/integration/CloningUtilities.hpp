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

#ifndef CLONINGANDPOINTERWRAPPER_H
#define CLONINGANDPOINTERWRAPPER_H


#include <functional>
#include <memory>
#include <type_traits>

/*!
* @file CloningUtilities.hpp
* @brief Contains traits, functions and classes for making a class clonable
* @author Luca Formaggia
* @note Taken from pacs-examples, folder of repository PACS Course (https://github.com/pacs-course), Advanced Programming for Scientific Computing, Politecnico di Milano
*/


/*
*
* Copyright (c) 2020
* Luca Formaggia
*
* Permission to use, copy, modify, distribute and sell this software for
* any purpose is hereby granted without fee, provided that the above
* copyright notice appear in all copies and that both that copyright
* notice and this permission notice appear in supporting documentation.
* The authors make no representations about
* the suitability of this
* software for any purpose. It is provided "as is" without express or
* implied warranty.
*/


/*!
* @namespace apsc
* @brief General namespace for utilitites
*/
namespace apsc
{

/*!
* @namespace TypeTraits
* @note To use this utilities you need apsc::TypeTraits,
*/
namespace TypeTraits
{
  /*!
  * @struct has_clone
  * @brief A type trait than checks if your class contains a method called clone()
  *        the template parameter. I use it to implement clonable classes which enable
  *        the prototype design pattern on a polymorphic family of classes. This is
  *        the primary template, which maps to false
  * @tparam T class to check if it is clonable
  * @tparam Sfinae
  */
  template <typename T, typename Sfinae = void>
  struct has_clone : public std::false_type
  {};

  /*! 
  * @struct has_clone
  * @brief Specialised version that is activated if T is clonable.
  *        Indeed, if T is not clonable the second template parameter cannot
  *        be substituted with a valid type. So SFINAE applies and this
  *        version is discarded. declval<T&>() allows to test the return type of clone() 
  *        with no need of creating an object of type T.
  * @tparam T the base class of the hierarchy of clonable classes
  * @note it inherits from std::true_type.
  */
  template <typename T>
  struct has_clone<
    T, typename std::enable_if_t<std::is_convertible_v<
         std::unique_ptr<T>, decltype(std::declval<const T>().clone())> > >
    : std::true_type
  {};

  /*!
  * @brief Helper function to understand if a class is clonable
  *  @tparam T the base class of the hierarchy of clonable classes
  * @return true if the class is clonable
  */
  template <class T>
  constexpr bool
  isClonable()
  {
    return has_clone<T>();
  }

  /*!
  * @brief C++17 style for extracting the value of type trait
  * @tparam T the base class of the hierarchy of clonable classes
  * @return true if T is clonable
  */
  template <typename T> constexpr bool has_clone_v = isClonable<T>();

  /*!
  * @brief Concept expressing the traits
  * @tparam T the class that could be clonable
  */
  template <class T>
  concept Clonable = has_clone_v<T>;

} // end namespace TypeTraits


/*!
* @class PointerWrapper
* @brief A smart pointer that handles cloning for compusing with polymorphic objects.
*        This class implements a generic wrapper around a unique pointer
*        useful to support the bridge pattern. Its role is to ease the memory
*        management of object which are composed polymorphically in a class
*        in order to implement a rule.  It is a extensive modification of the
*        class presented by Mark Joshi in his book "c++ design patterns and
*        derivative pricing", Cambridge Press. This version makes use of
*        std::unique_ptr to ease the handling of memory. It handles memory as a unique_ptr, 
*        but implements copy operations by cloning the resource.
* @tparam T the base class
*/
template <TypeTraits::Clonable T> class PointerWrapper
{
public:

  /*! 
  * @brief type of the stored unique pointer
  */
  using Ptr_t = std::unique_ptr<T>;
  /*! @brief This class imitates that of the unique pointer so it exposes the same member types*/
  using pointer = typename Ptr_t::pointer;
  using element_type = typename Ptr_t::element_type;
  using deleter_type = typename Ptr_t::deleter_type;

  /*!
  * @brief Default constructo 
  * @note The synthetic one is ok since the default constructor of a unique_ptr sets it to the null pointer.
  */
  PointerWrapper() = default;

  /*!
  * @brief Constructor
  * @param resource The resource to be cloned into
  * @note This constructor takes a reference to an object of type T or derived  from
  * T. It defines the conversion T& -> Wrapper<T> and thus the conversion
  * from any reference to a class derived from T.
  * It uses clone() to clone the resource.
  * It implements the Prototype Pattern
  * 
  */
  PointerWrapper(const T &resource) : DataPtr(resource.clone()) {}

  /*!
  * @brief Constructor
  * @param p The unique pointer to be moved into this class
  * @note A unique pointer to T is moved into the wrapper
  */
  PointerWrapper(Ptr_t &&p) noexcept : DataPtr(std::move(p)) {}

  /*!
  * @brief Constructor
  * @param p The pointer of type T*
  * @note Now the Wrapper has the ownership of the resource! Taking a pointer. Equivalent to  what unique_ptr does
  */
  explicit PointerWrapper(T *p) noexcept : DataPtr(p) {}

  /*!
  * @brief Copy constructor. Uses clone to clone the resource
  * @param original object to be copied
  */
  PointerWrapper(const PointerWrapper<T> &original)
    : DataPtr{original.get() ? original.DataPtr->clone() : Ptr_t{}}
  {}

  /*!
  * @brief Copy conversion constructor
  * @tparam U the type of the origin Wrapper, must be T or derived from T
  * @param original The original wrapper
  * @note This constructor takes any PointerWrapper<U> with U equal to T or a type
  *       derived from T. It allows the conversion from Wrapper<Derived> to Wrapper<Base>.
  */
  template <class U> PointerWrapper(const PointerWrapper<U> &original)
  {
    if(original.get())
      {
        // DataPtr.reset(static_cast<Ptr_t>(original.DataPtr->clone()));
        DataPtr = static_cast<Ptr_t>(original.get()->clone());
      }
  }

  /*!
  * @brief Copy-assignment operator. It resets current resource and clone that of the other wrapper
  * @param original the object to be copied
  * @return the copied object, non-const reference
  */
  PointerWrapper &
  operator=(const PointerWrapper<T> &original)
  {
    if(this != &original)
      DataPtr = original.DataPtr ? original.DataPtr->clone() : Ptr_t{};
    return *this;
  }

  /*!
  * @brief copying assignement allowing for conversions. This assignemt allow to convere PointerWrapper<Derived> in a PointeWrapper<Base>
  * @tparam U The derived type
  * @param original The Wrapper to convert-copy
  * @return a reference to myself
  */
  template <class U>
  PointerWrapper &
  operator=(const PointerWrapper<U> &original)
  {
    using OtherType = typename PointerWrapper<U>::Ptr_t;
    static_assert(std::is_constructible_v<Ptr_t, OtherType &&>,
                  "Cannot assign a non convertible PointerWrapper");
    DataPtr =
      original.get() ? static_cast<Ptr_t>(original.get()->clone()) : Ptr_t{};
    return *this;
  }

  /*!
  * @brief To move-assign a unique pointer
  * @param p object to be move-assigned
  * @return a reference to the move-assigned object, non-const reference
  * @note If argument is an rvalue unique_ptr it can be moved.
  */
  PointerWrapper &
  operator=(Ptr_t &&p) noexcept
  {
    DataPtr = std::move(p);
    return *this;
  }

  /*!
  * @brief Copy-Assignment of a unique pointer by cloning
  * @param p object to be copy-assigned
  * @return a reference to the copied-assigned object, non-const reference
  * @note If argument is an lvalue I need clone()
  */
  PointerWrapper &
  operator=(const Ptr_t &p)
  {
    DataPtr = p->clone(); // note that this is a move-assignment
    return *this;
  }

  /*!
  * @brief Move constructor
  * @param rhs the wrapper to be moved
  * @note unique_ptr can be moved
  */
  PointerWrapper(PointerWrapper<T> &&rhs) = default;

  /*!
  * @brief To allow conversion in move constructor
  * @tparam U Type of the object to be converted
  * @param rhs the wrapper to be moved
  */
  template <class U>
  PointerWrapper(PointerWrapper<U> &&rhs) noexcept
    : DataPtr{static_cast<T *>(rhs.release())}
  {}

  /*!
  * @brief Move assignement
  * @param rhs the wrapper to be moved
  * @return a wrapped pointed
  */
  PointerWrapper &operator=(PointerWrapper<T> &&rhs) = default;
  
  /*!
  * @brief To allow for conversion Derived -> Base
  * @tparam U Type of the object to be converted
  * @param rhs the wrapper to be moved, may be a wrapper to a derived type
  * @return the wrapped pointed
  * @note maybe not required since I have conversion in the move constructor.
  *       After the assignment the rhs is null.
  */
  template <class U>
  PointerWrapper &
  operator=(PointerWrapper<U> &&rhs) noexcept
  {
    using OtherPType = typename PointerWrapper<U>::pointer;
    static_assert(std::is_constructible<pointer, OtherPType>::value,
                  "Pointers must be convertible");
    if(this->get() != static_cast<pointer>(rhs.get()))
      {
        DataPtr.reset(rhs.release());
      }
    return *this;
  };

  /*! 
  * @brief Dereferencing operator, const version
  * @return the deferenced object
  * @note The PointerWrapper works like a pointer to T
  */
  const T &
  operator*() const noexcept
  {
    return *DataPtr;
  }

  /*! 
  * @brief Dereferencing operator, non-const version
  * @return the deferenced object
  * @note The PointerWrapper works like a pointer to T
  */
  T &
  operator*() noexcept
  {
    return *DataPtr;
  }

  /*! 
  * @brief Dereferencing operator, const version
  * @return the deferenced object pointer
  * @note The PointerWrapper works like a pointer to T
  */
  const T *
  operator->() const noexcept
  {
    return DataPtr.get();
  }

  /*! 
  * @brief Dereferencing operator, non-const version
  * @return the deferenced object pointer
  * @note The PointerWrapper works like a pointer to T
  */
  T *
  operator->() noexcept
  {
    return DataPtr.get();
  }

  /*!
  * @brief It releases the resource 
  * @return pointer
  */
  auto
  release() noexcept
  {
    return DataPtr.release();
  }

  /*! 
  * @brief Deletes the resource. You can pass the pointer of a new resource to hold
  * @param ptr the pointer to the new resource, defaulted to nullprt
  */
  void
  reset(pointer ptr = nullptr) noexcept
  {
    DataPtr.reset(ptr);
  }

  /*! 
  * @brief Swap wrappers 
  * @param other wrapper to be swapped
  */
  void
  swap(PointerWrapper<T> &other) noexcept
  {
    DataPtr.swap(other.DataPtr);
  }

  /*! 
  * @brief Get to the pointer 
  * @return the pointer
  */
  pointer
  get() const noexcept
  {
    return DataPtr.get();
  }

  /*!
  * @brief Getter for the deleter, non-const version
  * @return The deleter object which would be used for destruction of the managed object.
  */
  auto &
  get_deleter() noexcept
  {
    return DataPtr.get_deleter();
  }

  /*!
  * @brief Getter for the deleter, const version
  * @return The deleter object which would be used for destruction of the managed object.
  */
  auto const &
  get_deleter() const noexcept
  {
    return DataPtr.get_deleter();
  }

  /*! 
  * @brief conversion to bool 
  * @return a conversion of ptr to bool
  */
  explicit
  operator bool() const noexcept
  {
    return static_cast<bool>(DataPtr);
  }

private:
  /*!Pointer*/
  Ptr_t DataPtr;
};
//! Utility to make a PointerWrapper
/*!
Creates a PointerWrapper<Base> indicating Base and Derived class.
I need two compulsory template parameters, one for the Base
and one for the concrete Derived class since I need to construct
a derived object. Of  course,

@note I may have B=D.

@tparam B A base class. The function returns PointerWrapper<B>
@tparam D The derived class. The wrapper will own an object of type D.
@tparam Args Automatically deduced possible arguments for the constructor of D
@param args The (possible) arguments of type Args
@return A PointerWrapper<B>
 */
template <class B, class D, typename... Args>
PointerWrapper<B>
make_PointerWrapper(Args &&...args)
{
  return PointerWrapper<B>{std::make_unique<D>(std::forward<Args>(args)...)};
}
/*!
Creates a PointerWrapper the class to hold.
@tparam D The PointerWrapper will own an object of type D.
@tparam Args Automatically deduced possible arguments for the constructor of D
@param args The (possible) arguments of type Args
@return A PointerWrapper<D>
 */
// template <class D, typename... Args>
// PointerWrapper<D>
// make_PointerWrapper(Args &&... args)
//{
//  return PointerWrapper<D>{std::make_unique<D>(std::forward<Args>(args)...)};
//}
//! comparison operator
#if __cplusplus < 202002L
template <class T, class U>
bool
operator<(PointerWrapper<T> const &a, PointerWrapper<U> const &b)
{
  return a.get() < b.get();
}
//! comparison operator
template <class T, class U>
bool
operator<=(PointerWrapper<T> const &a, PointerWrapper<U> const &b)
{
  return a.get() <= b.get();
}
//! comparison operator
template <class T, class U>
bool
operator>(PointerWrapper<T> const &a, PointerWrapper<U> const &b)
{
  return a.get() > b.get();
}
//! comparison operator
template <class T, class U>
bool
operator>=(PointerWrapper<T> const &a, PointerWrapper<U> const &b)
{
  return a.get() >= b.get();
}
//! comparison operator
template <class T, class U>
bool
operator==(PointerWrapper<T> const &a, PointerWrapper<U> const &b)
{
  return a.get() == b.get();
}
//! comparison operator
template <class T, class U>
bool
operator!=(PointerWrapper<T> const &a, PointerWrapper<U> const &b)
{
  return a.get() != b.get();
}
#else
/* C++20 use the spaceship operator to simplify things */
template <class T, class U>
auto // you need auto, let the compiler to the stuff
operator<=>(PointerWrapper<T> const &a, PointerWrapper<U> const &b)
{
  return a.get() <=> b.get();
}
template <class T, class U>
//! Equivalence operator. I need it since it cannot be deduced by <=>
bool
operator==(PointerWrapper<T> const &a, PointerWrapper<U> const &b)
{
  return a.get() == b.get();
}
/*!
@brief Equivalence operator with nullptr
@details I have copy and pasted from the analogous declaration for the 
unique_ptr. The concepts verify that the underlying pointer is comparable with nullptr_t
The return type is the corresponding three way comparison type returnd by the spaceship operator.
However if you want you can simplify things eliminating concepts and use automatic return type
@code
template< class T>
auto
operator<=>( const PointerWrapper<T>& x, std::nullptr_t )
{
    return x.get() <=> nullptr;
};
@endcode
@note I need it since it cannot be deduced by <=>
@tparam T The type of the pointer
@param x The wrapper
@param nullptr
@return true if the pointer is null

*/
template< class T>
    requires std::three_way_comparable<typename PointerWrapper<T>::pointer>
std::compare_three_way_result_t<typename PointerWrapper<T>::pointer>
    operator<=>( const PointerWrapper<T>& x, std::nullptr_t )
    {
        return x.get() <=> nullptr;
    };
 template< class T>
bool
    operator==( const PointerWrapper<T>& x, std::nullptr_t )
    {
        return x.get() == nullptr;
    };
   
#endif

} // end namespace apsc
/*!
 *  Specialization of std::hash. I can store wrappers in unordered associative
 * containers
 *  @note I rely on the fact the the standard library provides the hash for any
 * pointer type
 */
template <class T> struct std::hash<class apsc::PointerWrapper<T> >
{
  std::size_t
  operator()(const apsc::PointerWrapper<T> &w) const noexcept
  {
    return std::hash<typename apsc::PointerWrapper<T>::pointer>{}(w.get());
  }
};

#endif  /*CLONINGANDPOINTERWRAPPER_H*/
