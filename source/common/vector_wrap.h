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
#ifndef __VECTOR_WRAP_H__
#define __VECTOR_WRAP_H__


template <class VecOps>
class vector_wrap
{
public: 
    typedef VecOps vector_operations;
    typedef typename VecOps::vector_type  vector_type;
    typedef typename VecOps::scalar_type  scalar_type;

private:
    typedef scalar_type T;
    typedef vector_type T_vec;


    VecOps* vec_ops;
    bool allocated = false;
    void set_op(VecOps* vec_ops_){ vec_ops = vec_ops_; }

public:
    vector_wrap()
    {
    }
    ~vector_wrap()
    {
        free();
    }

    void alloc(VecOps* vec_ops_)
    {
        set_op(vec_ops_);

        if(!allocated)
        {
            vec_ops->init_vector(x); vec_ops->start_use_vector(x); 
            allocated = true;
        }
    }
    void free()
    {
        
        if(allocated)
        {
            vec_ops->stop_use_vector(x); vec_ops->free_vector(x);
            allocated = false;
        }
    }

    T_vec& get_ref()
    {
        return(x);
    }

    T_vec x = nullptr;

};


#endif // __VECTOR_WRAP_H__