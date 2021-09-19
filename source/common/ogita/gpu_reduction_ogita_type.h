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
#ifndef __GPU_REDUCTION_OGITA_TYPE_H__
#define __GPU_REDUCTION_OGITA_TYPE_H__

#include <thrust/complex.h>

namespace gpu_reduction_ogita_type{

template<typename T_>
struct type_complex_cast
{
    using T = T_;
};
  
template<>
struct type_complex_cast< thrust::complex<float> >
{
    using T = float;
};
template<>
struct type_complex_cast< thrust::complex<double> >
{
    using T = double;
};    


template<typename T>
struct return_real
{
    using T_real = typename type_complex_cast<T>::T;
    T_real get_real(T val)
    {
        return val;
    }    
};


template<>
struct return_real< thrust::complex<float> >
{
    using T_real = float;//typename type_complex_cast< thrust::complex<float> >::T;
    T_real get_real(thrust::complex<float> val)
    {
        return val.real();
    }    
};
template<>
struct return_real< thrust::complex<double> >
{
    using T_real = double;//typename type_complex_cast< thrust::complex<double> >::T;
    T_real get_real(thrust::complex<double> val)
    {
        return val.real();
    }    
};

}


#endif

    