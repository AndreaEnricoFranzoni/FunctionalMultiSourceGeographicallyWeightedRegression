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


#ifndef FGWR_FACTORY_HPP
#define FGWR_FACTORY_HPP


#include "../utility/traits_fdagwr.hpp"
#include "fgwr_fms_esc.hpp"
#include "fgwr_fms_sec.hpp"
#include "fgwr_fs.hpp"
#include "fgwr_fst.hpp"


/*!
* @tparam fdagwrType kind The type of Functional Geographical Weighted regression class desired.
* @param args Arguments to be forwarded to the constructor.
*/
template< FDAGWR_ALGO fdagwrType, typename INPUT = double, typename OUTPUT = double, class... Args >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::unique_ptr< fgwr<INPUT,OUTPUT> >
fgwr_factory(Args &&... args)
{
    static_assert(fdagwrType == FDAGWR_ALGO::_FMSGWR_ESC_ ||
                  fdagwrType == FDAGWR_ALGO::_FMSGWR_SEC_ ||
                  fdagwrType == FDAGWR_ALGO::_FMGWR_      ||
                  fdagwrType == FDAGWR_ALGO::_FGWR_       ||
                  fdagwrType == FDAGWR_ALGO::_FWR_,
                  "Error in fdagwrType: wrong type specified.");

    //FMS_ESC: multi-source: estimating: stationay -> station-dependent -> event-dependent
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMSGWR_ESC_)
        return std::make_unique<fgwr_fms_esc<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //FMS_SEC: multi-source: estimating: stationay -> event-dependent -> station-dependent
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMSGWR_SEC_)
        return std::make_unique<fgwr_fms_sec<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //GWR_FOS: one-source: estimating: stationary -> geographically dependent
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMGWR_)
        return std::make_unique<fgwr_fs<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //GWR_FST: stationary: estimating: stationary
    if constexpr (fdagwrType == FDAGWR_ALGO::_FWR_)
        return std::make_unique<fgwr_fst<INPUT,OUTPUT>>(std::forward<Args>(args)...);
}

#endif /*FGWR_FACTORY_HPP*/