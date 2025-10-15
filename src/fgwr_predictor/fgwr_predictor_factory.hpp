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


#ifndef FGWR_PREDICTOR_FACTORY_HPP
#define FGWR_PREDICTOR_FACTORY_HPP


#include "../utility/traits_fdagwr.hpp"
#include "fgwr_fms_esc_predictor.hpp"
#include "fgwr_fms_sec_predictor.hpp"
//#include "fgwr_fs_predictor.hpp"
#include "fgwr_fst_predictor.hpp"


/*!
* @tparam fdagwrType kind The type of Functional Geographical Weighted regression class desired.
* @param args Arguments to be forwarded to the constructor.
*/
template< FDAGWR_ALGO fdagwrType, typename INPUT = double, typename OUTPUT = double, class... Args >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::unique_ptr< fgwr_predictor<INPUT,OUTPUT> >
fgwr_predictor_factory(Args &&... args)
{
    static_assert(fdagwrType == FDAGWR_ALGO::_FMSGWR_ESC_ ||
                  fdagwrType == FDAGWR_ALGO::_FMSGWR_SEC_ ||
                  fdagwrType == FDAGWR_ALGO::_FMGWR_      ||
                  fdagwrType == FDAGWR_ALGO::_FGWR_       ||
                  fdagwrType == FDAGWR_ALGO::_FWR_,
                  "Error in fdagwrType: wrong type specified.");

    //FMS_ESC: multi-source: estimating: stationay -> station-dependent -> event-dependent
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMSGWR_ESC_)
        return std::make_unique<fgwr_fms_esc_predictor<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //FMS_SEC: multi-source: estimating: stationay -> event-dependent -> station-dependent
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMSGWR_SEC_)
        return std::make_unique<fgwr_fms_sec_predictor<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //GWR_FS: one-source: estimating: stationary -> geographically dependent
    //if constexpr (fdagwrType == FDAGWR_ALGO::_FGWR_FS_)
    //    return std::make_unique<fgwr_fos<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //GWR_FST: stationary: estimating: stationary
    if constexpr (fdagwrType == FDAGWR_ALGO::_FWR_)
        return std::make_unique<fgwr_fst_predictor<INPUT,OUTPUT>>(std::forward<Args>(args)...);
}

#endif /*FGWR_PREDICTOR_FACTORY_HPP*/