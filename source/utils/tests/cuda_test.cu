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

#include <stdio.h>
#include <string>
#include <utils/InitCUDA.h>
#include <utils/cuda_safe_call.h>

int main(int argc, char **argv)
{
        bool    do_error = false;
        if ((argc >= 2)&&(std::string(argv[1]) == std::string("1"))) do_error = true;
        if (do_error) printf("you specified do error on purpose\n");
        try {
                if (!InitCUDA(0)) throw std::runtime_error("InitCUDA failed");

                int     *p;
                if (!do_error)
                        CUDA_SAFE_CALL( cudaMalloc((void**)&p, sizeof(int)*512) );
                else
                        CUDA_SAFE_CALL( cudaMalloc((void**)&p, -100 ) );

                return 0;

        } catch (std::runtime_error &e) {
                printf("%s\n", e.what());

                return 1;
        }
}