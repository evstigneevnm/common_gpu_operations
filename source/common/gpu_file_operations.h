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
#ifndef __GPU_FILE_OPERATIONS_H__
#define __GPU_FILE_OPERATIONS_H__

#include <common/file_operations.h>


template<class VectorOperations>
class gpu_file_operations
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    gpu_file_operations(VectorOperations* vec_op_):
    vec_op(vec_op_)
    {
        sz = vec_op->get_vector_size();
    }

    ~gpu_file_operations()
    {

    }

    void write_vector(const std::string &f_name, const T_vec& vec_gpu, unsigned int prec=16) const
    {
        file_operations::write_vector<T>(f_name, sz, vec_op->view(vec_gpu), prec);
    }

    void read_vector(const std::string &f_name, T_vec vec_gpu) const
    {
        
        file_operations::read_vector<T>(f_name, sz, vec_op->view(vec_gpu));
        vec_op->set(vec_gpu);
    }


private:
    VectorOperations* vec_op;
    size_t sz;

};




#endif // __GPU_FILE_OPERATIONS_H__