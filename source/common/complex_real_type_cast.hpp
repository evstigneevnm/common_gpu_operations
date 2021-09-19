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
#ifndef __COMPLEX_REAL_TYPE_CAST_HPP__
#define __COMPLEX_REAL_TYPE_CAST_HPP__

#include <thrust/complex.h>

namespace deduce_real_type_from_complex{
template<typename T>
struct recast_type
{
    using real = T;
};

template<>
struct recast_type< thrust::complex<float> >
{
    using real = float;
};

template<>
struct recast_type< thrust::complex<double> >
{
    using real = double;
};

}

#endif