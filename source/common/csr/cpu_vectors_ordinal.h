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
#ifndef __CSR__CPU_VECTORS_ORDINAL_H__
#define __CSR__CPU_VECTORS_ORDINAL_H__

#include<cstddef>
#include<cstdlib>

namespace csr
{

template<class I = int> //assumed to be an ordinal type
class cpu_vectors_ordinal
{
public:
    using scalar_type = I;
    using vector_type = I*;

    cpu_vectors_ordinal(size_t sz_):
    sz(sz_)
    { }
    ~cpu_vectors_ordinal()
    { }
    
    void init_vector(vector_type& x)const 
    {
        x = nullptr;
    }
    void free_vector(vector_type& x)const 
    {
        if (x != nullptr)
        { 
            std::free(x);
        }
    }
    void start_use_vector(vector_type& x)const
    {
        if (x == nullptr) 
        {
           x = reinterpret_cast<vector_type>( std::malloc(sz*sizeof(I) ) );
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
            #pragma omp parallel for
            for(int j = 0;j<sz;j++)
            {
                x_[j] = x_host_[j];
            }
        }
    }   

private:
    size_t sz;

};
}
#endif