#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <type_traits>
#include <immintrin.h>
#include <emmintrin.h>
#include <chrono>
//the intention of setting H=W=36 and KH=KW=5 is making the corresponding A rows of each outchannel equal to 1024, and each M_ACC iter can handle 32 rows, so that avoiding the junk A rows in A tile.
#define H 36            // The height of the activation frame
#define W 36            // The width of the activation frame
#define MA (H*W)        // The M dimension (rows) of the A matrix
#define K 1024          // Number of activation channels
#define N 1024          // Number of output channels
#define KH 5            // The height of the weights kernel
#define KW 5            // The width of the weights kernel
#define SH 1            // The vertical stride of the convolution
#define SW 1            // The horizontal stride of the convolution
#define M_ACC 2         // Number of C accumulators spanning the M dimension
#define N_ACC 2         // Number of C accumulators spanning the N dimension
#define TILE_M 16       // Number of rows in an A or C tile
#define TILE_K 64       // Number of columns in an A tile or rows in a B tile
#define TILE_N 16       // Number of columns in a B or C tile

#define HC ((H-KH)/SH+1) // The height of the output frame, =36-5+1=32
#define WC ((W-KW)/SW+1) // The width of the output frame,=36-5+1=32
#define MC (HC*WC)       // The M dimension (rows) of the C matrix

typedef int8_t type_t;      // The type of the data being operated on
typedef int res_type_t;  // The data type of the result

#define KPACK (4/sizeof(type_t)) // Vertical K packing into Dword

type_t A_mem[H][W][K];                   // A matrix (equivalent to A_mem[H*W][K])
type_t B_mem[KH][KW][K/KPACK][N][KPACK]; // B matrices
res_type_t C_mem[MC][N];                 // C matrix

template<size_t rows, size_t col_bytes> 
class Tile
{
    public:
        __tile1024i& getTile()
        {
            return tile;
        }
    private:
        __tile1024i tile {rows, col_bytes};
};

// template<class T>
// void tilezero (T& t){
//     _tile_zero(&t.getTile())
// }

// template<class T>
// void tileload (T& t, void* src, size_t stride){
//     _tile_loadd(&t.getTile(), src, stride);
// };

// template<class T>
// void tilestore(T& t, void* dst, size_t stride){
//     _tile_store(&t.getTile(), dst, stride);
// };

// template<class TA, class TB, class TC> 
// void tdp(TA& tA, TB& tB, TC& tC){
//     _tile_dpbusds(&tC.getTile(), tA.getTile(), tB.getTile());
// };

int mc_to_ha(int mc) {return mc / HC * SH;} // C matrix M -> A tensor h coord
int mc_to_wa(int mc) {return mc % HC * SW;} // C matrix M -> A tensor w coord

void type_t_convolution() {
    for (int n = 0; n < N; n += N_ACC*TILE_N) {
    for (int m = 0; m < MC; m += M_ACC*TILE_M) {
        Tile<TILE_M, TILE_N*sizeof(res_type_t)> tC[M_ACC][N_ACC];
        Tile<TILE_M, TILE_K*sizeof(type_t)> tA[M_ACC];
        Tile<TILE_K/KPACK, TILE_N*KPACK> tB; 

        for (int n_acc = 0; n_acc < N_ACC; ++n_acc)
            for (int m_acc = 0; m_acc < M_ACC; ++m_acc)
                _tile_zero(tC[m_acc][n_acc]);//clear Tile C register

        for (int k = 0; k < K; k += TILE_K) { //input channel
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    for (int n_acc = 0; n_acc < N_ACC; ++n_acc) {
                        int nc = n + n_acc*TILE_N;
                        //load kernel/filter
                        _tile_loadd(tB, B_mem[kh][kw][k/KPACK][nc], N*sizeof(type_t)*KPACK);
                        for (int m_acc = 0; m_acc < M_ACC; ++m_acc) {
                            int mc = m + m_acc*TILE_M;
                            //index current A row by outchannel row
                            if (n_acc == 0) {
                                int ha = mc_to_ha(mc)+kh, wa = mc_to_wa(mc)+kw; 
                                _tile_loadd(tA[m_acc], &A_mem[ha][wa][k], K*SW*sizeof(type_t));
                            }
                            //accum intermediate res into tmm C
                            _tile_dpbusds(tC[m_acc][n_acc], tA[m_acc], tB);
                            if (k + kh + kw == K - TILE_K + KH + KW - 2){
                                _tile_store(tC[m_acc][n_acc], &C_mem[mc][nc], N*sizeof(res_type_t));
                            }
                        }
                    }
                }
            }
        }
    }
}
