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

#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <utils/log.h>
#include <utils/main_try_catch_macro.h>

int main(int argc, char **args)
{
    if (argc < 2) {
        std::cout << "USAGE: " << std::string(args[0]) << " <block_number>" << std::endl;
        return 0;
    }

    utils::log_std  log;
    USE_MAIN_TRY_CATCH(log)  

    int block_number = atoi(args[1]);

    MAIN_TRY("test block 1")
    if (block_number == 1) throw std::runtime_error("error block 1");
    MAIN_CATCH(1)

    MAIN_TRY("test block 2")
    if (block_number == 2) throw std::runtime_error("error block 2");
    MAIN_CATCH(2)

    MAIN_TRY("test block 3")
    if (block_number == 3) throw std::runtime_error("error block 3");
    MAIN_CATCH(3)

    return 0;
}