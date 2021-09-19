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

#ifndef __LOG_MPI_H__
#define __LOG_MPI_H__

#include <string>
#include <exception>
#include <stdexcept>
#include <cstdarg>
#include <cstdio>
#include <mpi.h>
#include "log.h"

namespace utils
{

class log_mpi : public log
{
    int     log_lev;
    int     comm_rank_, comm_size_;
public:
    log_mpi() : log_lev(1) 
    {  
        if (MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank_) != MPI_SUCCESS) throw std::runtime_error("log_mpi::MPI_Comm_rank failed");
        if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size_) != MPI_SUCCESS) throw std::runtime_error("log_mpi::MPI_Comm_size failed");
    }

    virtual void msg(const std::string &s, t_msg_type mt = INFO, int _log_lev = 1)
    {
        if ((mt != ERROR)&&(_log_lev > log_lev)) return;
        //TODO
        if (mt == INFO) {
            if (comm_rank_ == 0) printf("INFO:    %s\n", s.c_str());
        } else if (mt == INFO_ALL) {
            printf("INFO(%2d):%s\n", comm_rank_, s.c_str());
        } else if (mt == WARNING) {
            printf("WARNING: %s\n", s.c_str());
        } else if (mt == ERROR) {
            printf("ERROR:   %s\n", s.c_str());
        } else 
            throw std::logic_error("log_mpi::log: wrong t_msg_type argument");
    }
    virtual void set_verbosity(int _log_lev = 1) { log_lev = _log_lev; }
};

}

#endif
