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
#ifndef __CSR__GPU_ILU0_PRECONDITIONER_H__
#define __CSR__GPU_ILU0_PRECONDITIONER_H__

#include <stdexcept>


namespace csr
{
template<class VectorOperations, class LinearOperator, class Matrix>
struct gpu_ilu0_preconditioner
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    gpu_ilu0_preconditioner(const VectorOperations* vec_ops_):
    vec_ops(vec_ops_)
    {
        vec_ops->init_vector(y); vec_ops->start_use_vector(y);
    }
    ~gpu_ilu0_preconditioner()
    {
        if(mat != nullptr) delete mat;
        vec_ops->stop_use_vector(y); vec_ops->free_vector(y);
    }

    void set_operator(const LinearOperator* lin_op_)
    {
        lin_op = lin_op_;
        operator_set = true;
    }
    void set_matrix(const Matrix matP_)
    {
        mat = new Matrix(matP_);
        matrix_set = true;
    }

    void apply(T_vec& x)const
    {
        if(!matrix_set)
        {
            //throw std::logic_error("csr::preconditioner: operator not set. Use set_operator(Matrix).");
            
        }
        else
        {
            vec_ops->assign(x, y);
            
        }
    }
private:
    mutable T_vec y = nullptr;
    bool operator_set = false;
    bool matrix_set = false;
    const VectorOperations* vec_ops;
    const LinearOperator* lin_op;
    Matrix* mat = nullptr;
    

};
}
#endif