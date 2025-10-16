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


#ifndef FWR_PREDICTOR_FACTORY_HPP
#define FWR_PREDICTOR_FACTORY_HPP


#include "../utility/traits_fdagwr.hpp"
#include "fwr_FMSGWR_ESC_predictor.hpp"
#include "fwr_FMSGWR_SEC_predictor.hpp"
#include "fwr_FMGWR_predictor.hpp"
#include "fwr_FGWR_predictor.hpp"
#include "fwr_FWR_predictor.hpp"


/*!
* @tparam fdagwrType kind The type of Functional Geographical Weighted regression class desired.
* @param args Arguments to be forwarded to the constructor.
*/
template< FDAGWR_ALGO fdagwrType, typename INPUT = double, typename OUTPUT = double, class... Args >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
std::unique_ptr< fwr_predictor<INPUT,OUTPUT> >
fwr_predictor_factory(Args &&... args)
{
    static_assert(fdagwrType == FDAGWR_ALGO::_FMSGWR_ESC_ ||
                  fdagwrType == FDAGWR_ALGO::_FMSGWR_SEC_ ||
                  fdagwrType == FDAGWR_ALGO::_FMGWR_      ||
                  fdagwrType == FDAGWR_ALGO::_FGWR_       ||
                  fdagwrType == FDAGWR_ALGO::_FWR_,
                  "Error in fdagwrType: wrong type specified.");

    //predictor of FMSGWR_ESC
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMSGWR_ESC_)
        return std::make_unique<fwr_FMSGWR_ESC_predictor<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //predictor of FMSGWR_SEC
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMSGWR_SEC_)
        return std::make_unique<fwr_FMSGWR_SEC_predictor<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //predictor of FMGWR
    if constexpr (fdagwrType == FDAGWR_ALGO::_FMGWR_)
        return std::make_unique<fwr_FMGWR_predictor<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //predictor of FGWR
    if constexpr (fdagwrType == FDAGWR_ALGO::_FGWR_)
        return std::make_unique<fwr_FGWR_predictor<INPUT,OUTPUT>>(std::forward<Args>(args)...);

    //predictor of FWR
    if constexpr (fdagwrType == FDAGWR_ALGO::_FWR_)
        return std::make_unique<fwr_FWR_predictor<INPUT,OUTPUT>>(std::forward<Args>(args)...);
}

#endif /*FWR_PREDICTOR_FACTORY_HPP*/