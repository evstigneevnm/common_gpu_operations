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
#ifndef __CSR__GPU_VECTORS_ORDINAL_H__
#define __CSR__GPU_VECTORS_ORDINAL_H__

#include<cstddef>
#include<cstdlib>
#include<utils/cuda_support.h>

namespace csr
{
template<class I = int>
class gpu_vectors_ordinal
{
public:
    using scalar_type = I;
    using vector_type = I*;
    
    gpu_vectors_ordinal(size_t sz_):
    sz(sz_)
    {
    }
    
    ~gpu_vectors_ordinal()
    {
    }

    void init_vector(vector_type& x)const 
    {
        x = nullptr;
    }
    void free_vector(vector_type& x)const 
    {
        if (x != nullptr) 
        {
            cudaFree(x);
        }
    }
    void start_use_vector(vector_type& x)const
    {
        if (x == nullptr) 
        {
           x = device_allocate<scalar_type>(sz);
        }
    }
    void stop_use_vector(vector_type& x)const
    {
    }
    size_t get_vector_size()
    {
        return sz;
    }
    //sets a vector from a host vector. 
    void set(const vector_type& x_host_, vector_type& x_) const
    {
        if(x_!=nullptr)
        {
            host_2_device_cpy<scalar_type>(x_, x_host_, sz);
        }
    }    
private:
    size_t sz;
};
}
#endif