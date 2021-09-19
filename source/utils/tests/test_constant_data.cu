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

#include <cstdio>
#include <utils/init_cuda.h>
#include <utils/constant_data.h>

struct t_test
{
    int x;
};

void test_init();
void test_test();

DEFINE_CONSTANT_BUFFER(t_test, buf)

__global__ void ker_test()
{
    printf("device test1: buf().x = %d\n", buf().x);
}

int main()
{
    t_test  test;
    test.x = 152;

    utils::init_cuda(0);

    COPY_TO_CONSTANT_BUFFER(buf, test);
    test_init();

    printf("host test1: buf().x = %d\n", buf().x);
    ker_test<<<1,1>>>();
    cudaDeviceSynchronize();

    test_test();

    return 0;
}