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
#include <common/ogita/gpu_reduction_ogita_impl.cuh>
#include <thrust/complex.h>


template class gpu_reduction_ogita<float, float*>;
template class gpu_reduction_ogita<double, double*>;
template class gpu_reduction_ogita< thrust::complex<float>, thrust::complex<float>* >;
template class gpu_reduction_ogita< thrust::complex<double>, thrust::complex<double>* >;