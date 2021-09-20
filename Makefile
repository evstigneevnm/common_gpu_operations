DTYPE = -DTYPE=double
FTYPE = -DTYPE=float
NVCC = /usr/local/cuda/bin/nvcc
GCC = /usr/bin/gcc
GPP = /usr/bin/g++
GCC_kern = /usr/bin/g++
SM_CUDA = sm_35
CPP_STANDARD = c++14
TARGET = -g

NVCCFLAGS = -Wno-deprecated-gpu-targets $(TARGET) -arch=$(SM_CUDA) -std=$(CPP_STANDARD) -ccbin=$(GCC_kern)
LIBFLAGS = --compiler-options -fPIC
GCCFLAGS = $(TARGET) -std=$(CPP_STANDARD)
ICUDA = -I/usr/local/cuda/include
IPROJECT = -I source/     
IBOOST = -I/opt/boost/include/

LCUDA = -L/usr/local/cuda/lib64
LBOOST = -L/opt/boost/lib/
LCUBLAS = -lcublas
LCURAND = -lcurand
LCUFFT = -lcufft
LCUSOLVER = -lcusolver 
LIBCUSPARSE = -lcusparse
LOPENBLAS = -L/opt/OpenBLAS/lib -lopenblas
LPTHREAD = -lpthread


blas_test:
	2>result.make
	${NVCC} $(DTYPE) ${NVCCFLAGS} ${IPROJECT} ${ICUDA} ${LIBFLAGS} ${LCUDA} ${LCUBLAS} ${LCUSOLVER} source/external_libraries/tests/test_cublas_cusolver.cu -o test_blas.bin 2>>result.make


