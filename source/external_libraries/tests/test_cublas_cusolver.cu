#include<iostream>
#include<vector>
#include<string>
#include<cuda_runtime.h>
#include<cuda.h>


#include <external_libraries/cublas_wrap.h>
#include <external_libraries/cusolver_wrap.h>
#include <utils/cuda_support.h>






template<class T>
void print_matrix(int Nrows, int Ncols, T* A)
{
    for(int j =0;j<Nrows;j++)
    {
        for(int k=0;k<Ncols;k++)
        {
            std::cout << A[j+Nrows*k] << " ";
        }
        std::cout << std::endl;
    }

}

template<class T>
void print_vector(int Ncols, T* v)
{
    for(int k=0;k<Ncols;k++)
    {
        std::cout << v[k] << std::endl;
    }

}


template<class T>
T check_vs_exact(int sz_, const std::vector<T>& x, const std::vector<T>& x_ref)
{

    T norm2ls = 0;
    T norm2exact = 0;
    for(int j=0;j<sz_;j++)
    {
        T diff = x[j]-x_ref[j];
        norm2ls += diff*diff;
        norm2exact += x_ref[j]*x_ref[j];
    }
    return std::sqrt(norm2ls)/std::sqrt(norm2exact)*100.0;
}



int main(int argc, char const *argv[])
{
    using real = TYPE;

    int Nrows = 6;
    int Ncols = 3;
    int Nproj = 1;
    int j_proj = 2;
    int N = Nrows*Ncols;
    int M = Ncols*Ncols;
    std::vector<real> A(N, 2.0);
    std::vector<real> C(M, 0.0);
    std::vector<real> L(Ncols, 0.0);
    std::vector<real> eigv(Nrows*Nproj, 0.0);

    
    A = {0.0802, 0.6483, 0.3984, 0.4138, 0.4663, 0.8825, 0.1808, 0.0953, 0.5390, 0.9956, 0.9678, 0.5437, 0.5204, 0.6201, 0.4425, 0.8853, 0.1560, 0.1837};
    std::vector<real> L_exact = {0.467100125626155, 0.541837411068829, 4.992724153305018};

    std::cout << "A:" << std::endl;
    print_matrix(Nrows, Ncols, A.data() );

    real* Ad = nullptr;
    real* Cd = nullptr;
    real* Ld = nullptr;
    real* eigv_d = nullptr;

    if(init_cuda(6) == 0)
    {
        return 0;
    }
    cublas_wrap cublas(true);
    cusolver_wrap cusolver(true);
    
    Ad = device_allocate<real>(N);
    Cd = device_allocate<real>(M);
    Ld = device_allocate<real>(Ncols);
    eigv_d = device_allocate<real>(Nrows*Nproj);

    host_2_device_cpy<real>(Ad, A.data(), N);

    //gemm
    //(const char opA, 
    //const char opB, 
    //size_t RowA, 
    //size_t ColBC, 
    //size_t ColARowB, 
    //const float alpha, 
    //const float* A, 
    //size_t LDimA, 
    //const float* B, 
    //size_t LDimB, 
    //const float beta, 
    //float* C, 
    //size_t LDimC)
    cublas.gemm('T', 'N', Ncols, Ncols, Nrows, real(1.0), Ad, Nrows, Ad, Nrows, real(0.0), Cd, Ncols);
    device_2_host_cpy(C.data(), Cd, M);    
    std::cout << "C:" << std::endl;
    print_matrix(Ncols, Ncols, C.data() );

    cusolver.eig(Ncols, Cd, Ld);

    device_2_host_cpy(L.data(), Ld, Ncols);
    device_2_host_cpy(C.data(), Cd, M);
    std::cout << "V(C):" << std::endl;
    print_matrix(Ncols, Ncols, C.data() );
    std::cout << "/\\(C):" << std::endl;
    print_vector(Ncols, L.data() );
    std::cout << "|| /\\ - /\\_{exact} ||_2:" << std::endl;
    
    
    std::cout << check_vs_exact(Ncols, L, L_exact) << std::endl;
    

    //gemm
    //(const char opA, 
    //const char opB, 
    //size_t RowA, 
    //size_t ColBC, 
    //size_t ColARowB, 
    //const float alpha, 
    //const float* A, 
    //size_t LDimA, 
    //const float* B, 
    //size_t LDimB, 
    //const float beta, 
    //float* C, 
    //size_t LDimC)
    cublas.gemm('N', 'N', Nrows, Nproj, Ncols, real(1.0), Ad, Nrows, &Cd[j_proj*Ncols], Ncols, real(0.0), eigv_d, Nrows);
    device_2_host_cpy(eigv.data(), eigv_d, Nrows*Nproj);

    std::cout << "Eigenvector projections: " << std::endl;
    print_matrix(Nrows, Nproj, eigv.data() );


    std::cout << "tridiag solution. Original matrix:" << std::endl;
    std::vector<real> Atr(3*3, 2.0);
    Atr = {3.0, 0.0, 0.0, 2.1, -0.42, 0.0, -4.0, 102.53, 1.0/70.0};
    print_matrix(3, 3, Atr.data() );
    std::cout << "rhs:" << std::endl;
    std::vector<real> btr(3, -3.0); btr = {-3.0/21.4, 3234.123456789, 3.0/11.0};
    print_vector(3, btr.data());

    real *Atr_d = nullptr;
    real *xtr_d = nullptr;
    Atr_d = device_allocate<real>(3*3);
    xtr_d = device_allocate<real>(3);
    host_2_device_cpy<real>(Atr_d, Atr.data(), 3*3);
    host_2_device_cpy<real>(xtr_d, btr.data(), 3);

    //A from the left, upper trinagular, no transpose, the diagnial is not unit
    cublas.trsm('L', 'U', 'N', false, 3, 1, real(1.0), Atr_d, 3, xtr_d, 3);
    
    std::vector<real> xtr(3, 0.0);
    device_2_host_cpy<real>(xtr.data(), xtr_d, 3);
    std::cout << "solution:" << std::endl;
    print_vector(3, xtr.data());
    
    std::vector<real> x_exact(3, 0.0); x_exact = {4735146651831757.0/2199023255552.0, -6684677532162513.0/2199023255552.0, 210.0/11.0};
    std::cout << "solution error:" << std::endl;
    std::cout << check_vs_exact(3, xtr, x_exact) << " %" << std::endl;    


    std::cout << "Linear system dense solution. Matrix:" << std::endl;    
    
    size_t Nls = 10;
    std::vector<real> Als(Nls*Nls,0);
    Als={0.515, 0.721, 0.230, 0.939, 0.921, 0.001, 0.780, 0.765, 0.960, 0.374, 0.731, 0.377, 0.486, 0.696, 0.435, 0.432, 0.530, 0.618, 0.862, 0.991, 0.293, 0.807, 0.351, 0.100, 0.796, 0.581, 0.987, 0.098,  0.752, 0.846, 0.688, 0.861, 0.232, 0.182, 0.875, 0.178, 0.344, 0.240, 0.222, 0.394, 0.655, 0.296, 0.648, 0.921, 0.967, 0.435, 0.783, 0.325, 0.133, 0.144, 0.547, 0.466, 0.491, 0.424, 0.641, 0.322, 0.800, 0.638, 0.574, 0.945, 0.935, 0.438, 0.744, 0.123, 0.634, 0.935, 0.440, 0.312, 0.959, 0.169, 0.189, 0.303, 0.782, 0.109, 0.224, 0.716, 0.544, 0.207, 0.147, 0.919, 0.114, 0.035, 0.967, 0.031, 0.544, 0.763, 0.890, 0.184, 0.592, 0.625, 0.974, 0.483, 0.137, 0.484, 0.174, 0.013, 0.442, 0.066, 0.671, 0.220};
    print_matrix(Nls, Nls, Als.data() );
    std::vector<real> zls(Nls, 0);
    std::vector<real> bls(Nls, 0);
    std::vector<real> xls(Nls, 0);
    bls = {0.3, 0.8, 0.74, 0.877, 0.606, 0.259, 0.852, 0.088, 0.776, 0.913};
    std::cout << "rhs:" << std::endl;
    print_vector(Nls, bls.data());

    real* Als_d = device_allocate<real>(Nls*Nls);
    real* xls_d = device_allocate<real>(Nls);

    host_2_device_cpy<real>(Als_d, Als.data(), Nls*Nls);
    host_2_device_cpy<real>(xls_d, bls.data(), Nls);
    
    cusolver.gesv((&cublas), Nls, Als_d, xls_d);

    device_2_host_cpy<real>(xls.data(), xls_d, Nls);
    std::cout << "solution:" << std::endl;
    print_vector(Nls, xls.data() );   

    std::vector<real> xls_exact(Nls, 0.0); 
    xls_exact = {4529738040764787.0/4503599627370496.0, 4149931559841307.0/4503599627370496.0, -3162884293294961.0/72057594037927936.0, 7709507427820179.0/9007199254740992.0, -7501191770050445.0/36028797018963968.0, -5322232508147663.0/2251799813685248.0, -605723551812547.0/562949953421312.0, 2207752075266355.0/2251799813685248.0, 1298630493819959.0/1125899906842624.0, 1504775877020433.0/2251799813685248.0};
    std::cout << "solution error:" << std::endl;
    std::cout << check_vs_exact(Nls, xls, xls_exact) << " %" << std::endl;   
    std::cout << "check for other interfaces:" << std::endl;
    cusolver.set_cublas( (&cublas) );
    
    real* bls_d = device_allocate<real>(Nls);
    host_2_device_cpy<real>(Als_d, Als.data(), Nls*Nls);
    host_2_device_cpy<real>(bls_d, bls.data(), Nls);
    host_2_device_cpy<real>(xls_d, zls.data(), Nls);

    cusolver.gesv(Nls, Als_d, bls_d, xls_d);
    device_2_host_cpy<real>(xls.data(), xls_d, Nls);
    std::cout << "solution error [gesv(const size_t rows_cols, T* A, const T* b, T* x)]:" << std::endl;
    std::cout << check_vs_exact(Nls, xls, xls_exact) << " %" << std::endl;      

    host_2_device_cpy<real>(Als_d, Als.data(), Nls*Nls);
    host_2_device_cpy<real>(xls_d, zls.data(), Nls);
    cusolver.gesv(Nls, (const real*)Als_d, bls_d, xls_d);
    device_2_host_cpy<real>(xls.data(), xls_d, Nls);
    std::vector<real> Als_check(Nls*Nls, 0);
    device_2_host_cpy<real>(Als_check.data(), Als_d, Nls*Nls);
    std::cout << "solution error [gesv(const size_t rows_cols, const T* A, const T* b, T* x)]:" << std::endl;
    std::cout << check_vs_exact(Nls, xls, xls_exact) << " %" << std::endl;  
    std::cout << "check that matrix remains const [A - A_gpu]: " << check_vs_exact(Nls*Nls, Als_check, Als) << std::endl;



    CUDA_SAFE_CALL(cudaFree(eigv_d));
    CUDA_SAFE_CALL(cudaFree(Ad));
    CUDA_SAFE_CALL(cudaFree(Cd));
    CUDA_SAFE_CALL(cudaFree(Ld));
    CUDA_SAFE_CALL(cudaFree(Atr_d));
    CUDA_SAFE_CALL(cudaFree(xtr_d));
    CUDA_SAFE_CALL(cudaFree(Als_d));
    CUDA_SAFE_CALL(cudaFree(xls_d));
    CUDA_SAFE_CALL(cudaFree(bls_d));

    return 0;
}