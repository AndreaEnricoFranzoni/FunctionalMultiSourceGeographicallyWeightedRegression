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


#include "traits_fdagwr.hpp"
#include "fgwr_fms_esc.hpp"
#include "fgwr_fms_sec.hpp"
#include "fgwr_fos.hpp"
#include "fgwr_fst.hpp"


/*!
* @tparam fdagwrType kind The type of Functional Geographical Weighted regression class desired.
* @param args Arguments to be forwarded to the constructor.
*/
template <FDAGWR_ALGO fdagwrType, class... Args>
std::unique_ptr<fgwr>
fgwr_factory(Args &&... args)
{
    static_assert(fdagwrType == FDAGWR_ALGO::GWR_FMS_ESC ||
                  fdagwrType == FDAGWR_ALGO::GWR_FMS_SEC ||
                  fdagwrType == FDAGWR_ALGO::GWR_FOS     ||
                  fdagwrType == FDAGWR_ALGO::GWR_FST,
                  "Error in fdagwrType: wrong type specified.");

    //FMS_ESC: multi-source: estimating: stationay -> station-dependent -> event-dependent
    if constexpr (fdagwrType == FDAGWR_ALGO::GWR_FMS_ESC)
        return std::make_unique<fgwr_fms_esc>(std::forward<Args>(args)...);

    //FMS_SEC: multi-source: estimating: stationay -> event-dependent -> station-dependent
    if constexpr (fdagwrType == FDAGWR_ALGO::GWR_FMS_SEC)
        return std::make_unique<fgwr_fms_sec>(std::forward<Args>(args)...);

    //GWR_FOS: one-source: estimating: stationary -> geographically dependent
    if constexpr (fdagwrType == FDAGWR_ALGO::GWR_FOS)
        return std::make_unique<fgwr_fos>(std::forward<Args>(args)...);

    //GWR_FST: stationary: estimating: stationary
    if constexpr (fdagwrType == FDAGWR_ALGO::GWR_FST)
        return std::make_unique<fgwr_fst>(std::forward<Args>(args)...);
}

#endif /*FGWR_FACTORY_HPP*/