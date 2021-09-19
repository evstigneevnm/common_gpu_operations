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
#ifndef __ARNOLDI_file_operations_H__
#define __ARNOLDI_file_operations_H__

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <common/macros.h>


namespace file_operations
{
template <class T>
void write_vector(const std::string &f_name, size_t N, T *vec, unsigned int prec=17)
{
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("print_vector: error while opening file " + f_name);

        for (size_t i = 0; i < N-1; ++i)
        {
            if (!(f << std::scientific << std::setprecision(prec) << vec[i] <<  std::endl))
                throw std::runtime_error("print_vector: error while writing to file " + f_name);
        }
        if (!(f << std::setprecision(prec) << vec[N-1]))
            throw std::runtime_error("print_vector: error while writing to file " + f_name);
        
        f.close();
}


template <class T>
void write_matrix(const std::string &f_name, size_t Row, size_t Col, T *matrix, unsigned int prec=17)
{
    size_t N=Col;
    std::ofstream f(f_name.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("print_matrix: error while opening file " + f_name);
    for (size_t i = 0; i<Row; i++)
    {
        for(size_t j=0;j<Col;j++)
        {
            if(j<Col-1)
                f << std::setprecision(prec) << matrix[I2_R(i,j,Row)] << " ";
            else
                f << std::setprecision(prec) << matrix[I2_R(i,j,Row)];

        }
        f << std::endl;
    } 
    
    f.close();
}

int read_matrix_size(const std::string &f_name)
{

    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_matrix_size: error while opening file " + f_name);
    std::string line;
    int matrix_size=0;
    while (std::getline(f, line)){
        matrix_size++;
    }
    f.close();
    return matrix_size;
}

template <class T>
void read_matrix(const std::string &f_name,  size_t Row, size_t Col,  T *matrix){
    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_matrix: error while opening file " + f_name);
    for (size_t i = 0; i<Row; i++)
    {
        for(size_t j=0;j<Col;j++)
        {
            // double val=0;  
            // fscanf(stream, "%le",&val);                
            // matrix[I2(i,j,Row)]=(real)val;
            T val;
            f >> val;
            matrix[I2_R(i,j,Row)]=(T)val;
        }
        
    } 

    f.close();
}

template <class T>
int read_vector(const std::string &f_name,  size_t N,  T *vec){

    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_vector: error while opening file " + f_name);
    for (size_t i = 0; i<N; i++)
    {
        T val=0;   
        f >> val;             
        vec[i]=(T)val;           
    } 
    f.close();
    return 0;
}


}

#endif