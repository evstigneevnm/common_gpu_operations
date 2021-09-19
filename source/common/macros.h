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
#ifndef __ARNOLDI_MACROS_H__
#define __ARNOLDI_MACROS_H__

#ifndef I2_R
	#define I2_R(i , j, Rows) (i)+(j)*(Rows)
#endif

#ifndef I2
    #define I2(i, j, Col) (i)*(Col)+(j)
#endif

#ifndef I2P
    #define I2P(j, k) ((j)>(Nx-1)?(j)-Nx:(j)<0?(Nx+(j)):(j))*(Ny)+((k)>(Ny-1)?(k)-Ny:(k)<0?(Ny+(k)):(k))
#endif

#ifndef _I3
    #define _I3(i, j, k, Nx, Ny, Nz) (i)*(Ny*Nz) + (j)*(Nz) + (k)
#endif

#ifndef I3 //default for Nx, Ny, Nz
    #define I3(i, j, k) _I3(i, j, k, Nx, Ny, Nz)
#endif

#ifndef I3P
	#define I3P(i, j, k) (Ny*Nz)*((i)>(Nx-1)?(i)-Nx:(i)<0?(Nx+(i)):(i))+((j)>(Ny-1)?(j)-Ny:(j)<0?(Ny+(j)):(j))*(Nz)+((k)>(Nz-1)?(k)-Nz:(k)<0?(Nz+(k)):(k))
#endif

#endif