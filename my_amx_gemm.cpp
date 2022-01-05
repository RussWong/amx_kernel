#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <type_traits>
#include <immintrin.h>
#include <emmintrin.h>
#include <chrono>

#define M 32768         // Number of rows in the A or C matrices
#define K 1024          // Number of columns in the A or rows in the B matrices
#define N 1024          // Number of columns in the B or C matrices
#define M_ACC 2         // Number of tiles in M dimension
#define N_ACC 2         // Number of tiles in N dimension
#define TILE_M 16       // Number of rows in an A or C tile
#define TILE_K 64       // Number of columns in an A tile or B tile
#define TILE_N 16       // Number of columns in a B or C tile
#define KPACK 4         // Block format to cache hit and vectorize. Nums packed in K column
typedef int8_t type_t;     // The type of the data being operated on
typedef int res_type_t; // The data type of the result
type_t A_mem[M][K];              // A matrix
type_t B_mem[K/KPACK][N][KPACK]; // B matrix
type_t B_org[K][N];
res_type_t C_mem[M][N];          // C matrix

struct tileconfig_t {
    uint8_t  palette_id;    //1byte
    uint8_t  reserved[15];  //15bytes
    uint16_t colb[16];      //32bytes, bytes_per_row
    uint8_t  rows[16];      //16bytes, nums_of_rows
} tc = {0};

void config_tiles(){
    tc.palette_id = 1;
    for (int i = 0; i < 4; i++) {
        tc.rows[i] = TILE_M;
        tc.colb[i] = TILE_N * sizeof(int);
    }
    for (int i = 4; i < 6; i++) {
        tc.rows[i] = TILE_M;
        tc.colb[i] = TILE_K * sizeof(int8_t);
    }
    for (int i = 6; i < 8; i++) {
        tc.rows[i] = TILE_K / KPACK;
        tc.colb[i] = TILE_N * KPACK * sizeof(int8_t);
    }
    _tile_loadconfig(&tc);
};

void init_input(){
    for (int i = 0; i < sizeof(A_mem) / sizeof(type_t); ++i){
        ((type_t*)A_mem)[i] = rand() % 5 -2;//2D表示，一行全是rand() % 5 -2
    }
    for (int i = 0; i < sizeof(B_org) / sizeof(type_t); ++i){
        ((type_t*)A_mem)[i] = rand() % 11 -5;//2D表示，一行全是rand() % 5 -2    
    }
};

void B_relayout(){
    for (int k=0; k<K; k++){
        for (int n=0; n<N; n++){
            B_mem[k/KPACK][n][k*KPACK] = B_org[k][n];
    }
};

void amx_gemm() {
    #pragma omp parallel for
    for (int m = 0; m < M; m += M_ACC * TILE_M) {
        _tile_loadconfig(&tc);
        for (int n = 0; n < N; n += N_ACC * TILE_N) {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            for (int k = 0; k < K; k += TILE_K) {
                _tile_loadd(6, B_mem[k/KPACK][n], N*sizeof(type_t)*KPACK);
                _tile_loadd(4, A_mem[m][k], K*sizeof(type_t));
                _tile_dpbssd(0, 4, 6);

                _tile_loadd(5, &A_mem[m + TILE_M][k], K*sizeof(type_t));
                _tile_dpbssd(2, 5, 6);

                _tile_loadd(7, B_mem[k/KPACK][n + TILE_N], N*sizeof(type_t)*KPACK);
                _tile_dpbssd(1, 4, 6);

                _tile_dpbssd(3, 5, 6);
            }
            _tile_stored(0, &C_mem[m][n], N*sizeof(res_type_t));
            _tile_stored(2, &C_mem[m + TILE_M][n], N*sizeof(res_type_t));
            _tile_stored(1, &C_mem[m][n + TILE_N], N*sizeof(res_type_t));
            _tile_stored(3, &C_mem[m + TILE_M][n + TILE_N], N*sizeof(res_type_t));
        }
    }
};

int main()
{
    init_input();
    config_tiles();
    B_relayout();
    double time = 0.0;
    for (int i=0; i<1000; ++i){
	    auto begin = std::chrono::high_resolution_clock::now();
            amx_gemm();
	    auto end = std::chrono::high_resolution_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    }
    printf("total time %f ms, GOPS: %lf \n", time/1e6, double(M)*K*N*2 / time * 1000);
};