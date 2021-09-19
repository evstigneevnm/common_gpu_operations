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

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.
// 

#ifndef __CUSOLVER_WRAP_H__
#define __CUSOLVER_WRAP_H__

#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
// #include <thrust/complex.h>
#include <POD/utils/cusolver_safe_call.h>
#include <POD/utils/cuda_safe_call.h>
#include <stdexcept>


template<class CUBLAS>
class cusolver_wrap
{
public:
    
    cusolver_wrap(): handle_created(false)
    {
        cusolver_create();
        handle_created=true;
    }

    cusolver_wrap(bool plot_info): handle_created(false)
    {
        if(plot_info)
        {
            cusolver_create_info();
            handle_created=true;
        }
        else
        {
            cusolver_create();
            handle_created=true;
        }
    }


    ~cusolver_wrap()
    {
        free_d_work_double();
        free_d_work_float();
        if(handle_created)
        {
            cusolver_destroy();
            handle_created=false;
        }

      
    }
 
    cusolverDnHandle_t* get_handle()
    {
        return &handle;
    }
    
    template<typename T> //only symmetric matrix is computed, only real eigenvalues
    void eig(size_t rows_cols, T* A, T* lambda); //returns matrix of left eigs in A

    void set_cublas_handle(const CUBLAS* cublas_p_)
    {
        
        
        bool blas_set = true;
    }


    template<typename T>
    void gesv(size_t rows_cols, T* A, T* b, T* x);



private:
    bool handle_created = false;

    cusolverDnHandle_t handle;        
    double* d_work_d = nullptr;
    float* d_work_f = nullptr;
    int work_size = 0;

    void cusolver_destroy()
    {
        CUSOLVER_SAFE_CALL(cusolverDnDestroy(handle));
    }
    
    void cusolver_create()
    {
        CUSOLVER_SAFE_CALL(cusolverDnCreate(&handle));
    }

    void cusolver_create_info()
    {
        cusolver_create();
        int cusolver_version;
        int major_ver, minor_ver, patch_level;
        CUSOLVER_SAFE_CALL(cusolverGetVersion(&cusolver_version));
        CUSOLVER_SAFE_CALL(cusolverGetProperty(MAJOR_VERSION, &major_ver));
        CUSOLVER_SAFE_CALL(cusolverGetProperty(MINOR_VERSION, &minor_ver));
        CUSOLVER_SAFE_CALL(cusolverGetProperty(PATCH_LEVEL, &patch_level));
        std::cout << "cuSOLVER v."<< cusolver_version << " (major="<< major_ver << ", minor=" << minor_ver << ", patch level=" << patch_level << ") handle created." << std::endl;
    }

    void free_d_work_double()
    {
        if(d_work_d!=nullptr)
        {
            cudaFree(d_work_d);
        }        
    }
    void free_d_work_float()
    {
        if(d_work_f!=nullptr)
        {
            cudaFree(d_work_f);
        }        
    }
    void set_d_work_double(int work_size_)
    {
        if(work_size<work_size_)
        {
            work_size = work_size_;
            free_d_work_double();
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_work_d, sizeof(double)*work_size) );     
        }
    }
    void set_d_work_float(int work_size_)
    {
        if(work_size<work_size_)
        {
            work_size = work_size_;
            free_d_work_float();
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_work_f, sizeof(float)*work_size) );     
        }
    }

};





template<> inline
void cusolver_wrap::eig(size_t rows_cols, double* A, double* lambda)
{
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    int m = rows_cols;
    int lda = m;
    int lwork = 0;
    
    int *devInfo = nullptr;
    CUDA_SAFE_CALL(cudaMalloc ((void**)&devInfo, sizeof(int)) );
    int info_gpu;

    CUSOLVER_SAFE_CALL
    (
        cusolverDnDsyevd_bufferSize
        (
            handle,
            jobz,
            uplo,
            m,
            A,
            lda,
            lambda,
            &lwork
        )
    );
    set_d_work_double(lwork);

    CUSOLVER_SAFE_CALL
    (
        cusolverDnDsyevd
        (
            handle,
            jobz,
            uplo,
            m,
            A,
            lda,
            lambda,
            d_work_d,
            lwork,
            devInfo
        )
    );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    CUDA_SAFE_CALL( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    if(info_gpu!=0)
    {
        throw std::runtime_error("cusolver_wrap::eig: info_gpu = " + std::to_string(info_gpu) );
    }

}

template<> inline
void cusolver_wrap::eig(size_t rows_cols, float* A, float* lambda)
{
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    int m = rows_cols;
    int lda = m;
    int lwork = 0;
    
    int *devInfo = nullptr;
    CUDA_SAFE_CALL(cudaMalloc ((void**)&devInfo, sizeof(int)) );
    int info_gpu;

    CUSOLVER_SAFE_CALL
    (
        cusolverDnSsyevd_bufferSize
        (
            handle,
            jobz,
            uplo,
            m,
            A,
            lda,
            lambda,
            &lwork
        )
    );
    set_d_work_float(lwork);

    CUSOLVER_SAFE_CALL
    (
        cusolverDnSsyevd
        (
            handle,
            jobz,
            uplo,
            m,
            A,
            lda,
            lambda,
            d_work_f,
            lwork,
            devInfo
        )
    );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    CUDA_SAFE_CALL( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    if(info_gpu!=0)
    {
        throw std::runtime_error("cusolver_wrap::eig: info_gpu = " + std::to_string(info_gpu) );
    }

}


#endif