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
#ifndef __GPU_FILE_OPERATIONS_FUNCTIONS_H__
#define __GPU_FILE_OPERATIONS_FUNCTIONS_H__

#include <utils/cuda_support.h>
#include <common/file_operations.h>

namespace gpu_file_operations_functions
{

    template <class T>
    void write_vector(const std::string &f_name, size_t N, T* vec_gpu, unsigned int prec=16)
    {
        T* vec_cpu = host_allocate<T>(N);
        device_2_host_cpy(vec_cpu, vec_gpu, N);
        file_operations::write_vector<T>(f_name, N, vec_cpu, prec);
        host_deallocate<T>(vec_cpu);
    }

    template <class T>
    void read_vector(const std::string &f_name, size_t N, T*& vec_gpu, unsigned int prec=16)
    {
        T* vec_cpu = host_allocate<T>(N);
        file_operations::read_vector<T>(f_name, N, vec_cpu, prec);
        host_2_device_cpy(vec_gpu, vec_cpu, N);
        host_deallocate<T>(vec_cpu);
    }
    template <class T>
    void write_matrix(const std::string &f_name, size_t Row, size_t Col, T* matrix_gpu, unsigned int prec=16)
    {  
        T* vec_cpu = host_allocate<T>(Row*Col);
        device_2_host_cpy(vec_cpu, matrix_gpu, Row*Col);
        file_operations::write_matrix<T>(f_name, Row, Col, vec_cpu, prec);
        host_deallocate<T>(vec_cpu);
    }

    template <class T>
    void read_matrix(const std::string &f_name, size_t Row, size_t Col, T*& matrix_gpu)
    {
        T* vec_cpu = host_allocate<T>(Row*Col);
        file_operations::read_matrix<T>(f_name, Row, Col, vec_cpu);
        host_2_device_cpy(matrix_gpu, vec_cpu, Row*Col);   
        host_deallocate<T>(vec_cpu);
 
    }


}


#endif
