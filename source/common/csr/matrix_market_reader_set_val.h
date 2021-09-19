/*
 *     This file is part of Common_GPU_Operations.
 *     Copyright (C) 2009-2021  Evstigneev Nikolay Mikhaylovitch <evstigneevnm@ya.ru>, Ryabkov Oleg Igorevitch
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *      */
#ifndef __CSR__MATRIX_MARKET_READER_SET_VAL_H__
#define __CSR__MATRIX_MARKET_READER_SET_VAL_H__


#include<complex>
#include<thrust/complex.h>
//TODO: use traits?

namespace csr
{
    
    template<class T>
    struct complex_base_type
    {
        using real = T;
        
    };
    template<>
    struct complex_base_type<std::complex<float> >
    {
        using real = float;


    };
    template<>
    struct complex_base_type<std::complex<double> >
    {
        using real = double;
      
    };
    template<>
    struct complex_base_type<thrust::complex<float> >
    {
        using real = float;
        
    };
    template<>
    struct complex_base_type<thrust::complex<double> >
    {
        using real = double;       
    };

    template<class Tl>
    void set_val(Tl& out_, double in_1_, double in_2_)
    {
        out_ = in_1_;
    }
    template<>
    void set_val(thrust::complex<float>& out_, double in_1_, double in_2_)
    {
        out_ = thrust::complex<float>(in_1_, in_2_);
    }
    template<>
    void set_val(thrust::complex<double>& out_, double in_1_, double in_2_)
    {
        out_ = thrust::complex<double>(in_1_, in_2_);
    }
    template<>
    void set_val(std::complex<float>& out_, double in_1_, double in_2_)
    {
        out_ = std::complex<float>(in_1_, in_2_);
    }
    template<>
    void set_val(std::complex<double>& out_, double in_1_, double in_2_)
    {
        out_ = std::complex<double>(in_1_, in_2_);
    }

}
#endif