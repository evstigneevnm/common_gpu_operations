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

#ifndef __STATIC_ASSERT_H__
#define __STATIC_ASSERT_H__

//unforunatly under nvcc this STATIC_ASSERT gives unreadable results (msg is not written and 'STATIC_ASSERT' is not written)
//seems that similar problems has even thrust static_assert
//on the other hand nvcc at least gives number lines of callers where assert was failed

//these with __ are not supposed to be used (just intermediate help macros)
#define __STATIC_ASSERT__CTASTR2(pre,post) pre ## post
#define __STATIC_ASSERT__CTASTR(pre,post) __STATIC_ASSERT__CTASTR2(pre,post)
//no line append becuase compiler seems to give it by himself
//__COUNTER__ requiers gcc at least 4.3; works on cl at least in VS2008 (did not tested former versions)
//ISSUE may be make more 'stupid' realisations for earlier versions of compilers
#define STATIC_ASSERT(cond,msg) \
    typedef struct { int __STATIC_ASSERT__CTASTR(STATIC_ASSERTION_FAILED_,msg) : !!(cond); } \
    __STATIC_ASSERT__CTASTR(STATIC_ASSERTION_FAILED_,__COUNTER__)

#endif