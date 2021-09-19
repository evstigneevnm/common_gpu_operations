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
#ifndef __GPU_REDUCTION_IMPL_OGITA_SHMEM_CUH__
#define __GPU_REDUCTION_IMPL_OGITA_SHMEM_CUH__

#include <thrust/complex.h>

namespace gpu_reduction_ogita_gpu_kernels
{

template<class T>
struct __GPU_REDUCTION_OGITA_H__SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

//Dynamic shared memory specialization
template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};
template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory<float>
{
    __device__ inline operator       float *()
    {
        extern __shared__ float __smem_f[];
        return (float *)__smem_f;
    }

    __device__ inline operator const float *() const
    {
        extern __shared__ float __smem_f[];
        return (float *)__smem_f;
    }
};
template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory< thrust::complex<float> >
{
    __device__ inline operator       thrust::complex<float> *()
    {
        extern __shared__ thrust::complex<float>(__smem_C[]);
        return (thrust::complex<float> *) __smem_C;
    }

    __device__ inline operator const thrust::complex<float> *() const
    {
        extern __shared__ thrust::complex<float>(__smem_C[]);
        return (thrust::complex<float> *) __smem_C;
    }
};
template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory< thrust::complex<double> >
{
    __device__ inline operator       thrust::complex<double> *()
    {
        extern __shared__ thrust::complex<double>(__smem_Z[]);
        return (thrust::complex<double> *) __smem_Z;
    }

    __device__ inline operator const thrust::complex<double> *() const
    {
        extern __shared__ thrust::complex<double>(__smem_Z[]);
        return (thrust::complex<double> *) __smem_Z;
    }
};



}
#endif