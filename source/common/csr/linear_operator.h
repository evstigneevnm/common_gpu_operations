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
#ifndef __CSR__LINEAR_OPERATOR_H__
#define __CSR__LINEAR_OPERATOR_H__

namespace csr
{
template <class VectorOperations, class Matrix>
struct linear_operator
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    linear_operator(const VectorOperations* vec_ops_, Matrix* mat_):
        vec_ops(vec_ops_), mat(mat_)
    {
    }
    void apply(const T_vec& x, T_vec& f)const
    {
        mat->axpy(1.0, (T_vec&)x, 0.0, f);
    }
private:
    const VectorOperations* vec_ops;
    const Matrix* mat;       

};
}

#endif