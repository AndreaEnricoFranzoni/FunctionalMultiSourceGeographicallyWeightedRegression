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


#ifndef FGWR_FACTORY_HPP
#define FGWR_FACTORY_HPP


#include "../utility/traits_fdagwr.hpp"
#include "fwr_FMSGWR_ESC.hpp"
#include "fwr_FMSGWR_SEC.hpp"
#include "fwr_FMGWR.hpp"
#include "fwr_FGWR.hpp"
#include "fwr_FWR.hpp"


/*!
* @file fwr_factory.hpp
* @brief Contains the definition of factory to create the correct functional weighted regression model fitting algorithm
* @author Andrea Enrico Franzoni
*/




/*!
* @brief Factory to create the correct functional weighted regression model fitting algorithm
* @tparam fdagwrType the functional weighted regression model to be fitted
* @tparam INPUT type of functional data abscissa
* @tparam OUTPUT type of functional data image
* @tparam Args variadic template
* @param args Arguments to be forwarded to the constructor of the right functional weighted regression model
* @return a unique pointer to the right functional weighted regression model, downcasted to the base class
*/
template< FDAGWR_ALGO fdagwrType, typename INPUT = double, typename OUTPUT = double, class... Args >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::unique_ptr< fwr<INPUT,OUTPUT> >
fwr_factory(Args &&... args)
{
    static_assert(fdagwrType == FDAGWR_ALGO::_FMSGWR_ESC_ ||
                  fdagwrType == FDAGWR_ALGO::_FMSGWR_SEC_ ||
                  fdagwrType == FDAGWR_ALGO::_FMGWR_      ||
                  fdagwrType == FDAGWR_ALGO::_FGWR_       ||
                  fdagwrType == FDAGWR_ALGO::_FWR_,
                  "Error in fdagwrType: wrong type specified.");

    //FMSGWR_ESC: multi-source: estimating: stationay -> station-dependent -> event-dependent
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMSGWR_ESC_)
        return std::make_unique<fwr_FMSGWR_ESC<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //FMSGWR_SEC: multi-source: estimating: stationay -> event-dependent -> station-dependent
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMSGWR_SEC_)
        return std::make_unique<fwr_FMSGWR_SEC<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //FMGWR: mixed: estimating: stationary -> non-stationary
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMGWR_)
        return std::make_unique<fwr_FMGWR<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //FGWR: only non-stationary
    if constexpr (fdagwrType == FDAGWR_ALGO::_FGWR_)
        return std::make_unique<fwr_FGWR<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //FWR: only stationary
    if constexpr (fdagwrType == FDAGWR_ALGO::_FWR_)
        return std::make_unique<fwr_FWR<INPUT,OUTPUT>>(std::forward<Args>(args)...);
}

#endif /*FGWR_FACTORY_HPP*/