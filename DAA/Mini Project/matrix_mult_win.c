/* matrix_mult_win.c - CPU + multithread comparison + CUDA launcher */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <process.h>
#include <math.h>
#include <time.h>

typedef double real;

/* Timer */
typedef struct { LARGE_INTEGER start, end, freq; } Timer;
static void timer_start(Timer *t){ QueryPerformanceFrequency(&t->freq); QueryPerformanceCounter(&t->start); }
static double timer_elapsed_ms(Timer *t){ QueryPerformanceCounter(&t->end); return (double)(t->end.QuadPart - t->start.QuadPart)*1000.0/(double)t->freq.QuadPart; }

typedef struct { int N; real *A, *B, *C; } MatContext;

static real *alloc_matrix(int N){ return (real*)malloc(sizeof(real)*N*N); }
static void fill_random(real *M, int N){ for(int i=0;i<N*N;i++) M[i] = (rand()%100)/10.0; }
static void zero_matrix(real *M,int N){ memset(M,0,sizeof(real)*N*N); }

/* --- Single Thread --- */
static void matmul_single(const MatContext *ctx){
    int N=ctx->N;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            real sum=0;
            for(int k=0;k<N;k++)
                sum+=ctx->A[i*N+k]*ctx->B[k*N+j];
            ctx->C[i*N+j]=sum;
        }
}

/* --- Per Row Thread (Thread Pool Version) --- */
typedef struct {
    const MatContext *ctx;
    int start_row;
    int end_row;
} ThreadBatchArg;

static unsigned __stdcall row_batch_thread(void *arg) {
    ThreadBatchArg *b = (ThreadBatchArg*)arg;
    const MatContext *ctx = b->ctx;
    int N = ctx->N;

    for (int i = b->start_row; i < b->end_row; i++) {
        for (int j = 0; j < N; j++) {
            real sum = 0;
            for (int k = 0; k < N; k++)
                sum += ctx->A[i*N + k] * ctx->B[k*N + j];
            ctx->C[i*N + j] = sum;
        }
    }

    free(b);
    _endthreadex(0);
    return 0;
}

static void matmul_per_row(const MatContext *ctx, int max_threads) {
    int N = ctx->N;

    // Automatically choose number of threads
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int num_threads = (max_threads < si.dwNumberOfProcessors) ? max_threads : si.dwNumberOfProcessors;
    if (num_threads > N) num_threads = N; // No more threads than rows

    HANDLE *h = (HANDLE*)malloc(sizeof(HANDLE) * num_threads);

    int rows_per_thread = (N + num_threads - 1) / num_threads; // Ceiling division

    for (int t = 0; t < num_threads; t++) {
        int start = t * rows_per_thread;
        int end = (start + rows_per_thread < N) ? start + rows_per_thread : N;
        if (start >= end) break;

        ThreadBatchArg *arg = (ThreadBatchArg*)malloc(sizeof(ThreadBatchArg));
        arg->ctx = ctx;
        arg->start_row = start;
        arg->end_row = end;

        unsigned tid;
        h[t] = (HANDLE)_beginthreadex(NULL, 0, row_batch_thread, arg, 0, &tid);
    }

    WaitForMultipleObjects(num_threads, h, TRUE, INFINITE);
    for (int t = 0; t < num_threads; t++) CloseHandle(h[t]);
    free(h);
}

/* --- CUDA --- */
#ifdef __cplusplus
extern "C" {
#endif
void cuda_matmul(double *A, double *B, double *C, int N);
#ifdef __cplusplus
}
#endif

/* --- MAIN --- */
int main(void){
    srand((unsigned)time(NULL));
    printf("Matrix Multiplication Benchmark (CPU vs Threads vs CUDA)\n");

    int tests; 
    printf("Enter number of test sizes: "); 
    scanf("%d",&tests);

    int *sizes = (int*)malloc(sizeof(int)*tests);
    for(int i=0;i<tests;i++){
        printf("Enter matrix size %d: ",i+1); 
        scanf("%d",&sizes[i]);
    }

    FILE *out = fopen("results.csv","w");
    if (!out) {
        perror("Failed to open results.csv");
        return 1;
    }
    fprintf(out,"Size,Single(ms),PerRow(ms),CUDA(ms)\n");

    for(int t=0;t<tests;t++){
        int N = sizes[t];
        printf("\n=== Testing %dx%d ===\n",N,N);

        real *A=alloc_matrix(N), *B=alloc_matrix(N), *C1=alloc_matrix(N), *C2=alloc_matrix(N), *C3=alloc_matrix(N);
        fill_random(A,N); fill_random(B,N); zero_matrix(C1,N); zero_matrix(C2,N); zero_matrix(C3,N);

        Timer timer;
        timer_start(&timer);
        MatContext ctx={N,A,B,C1};
        matmul_single(&ctx);
        double t1=timer_elapsed_ms(&timer);

        timer_start(&timer);
        matmul_per_row(&ctx, 8); // 8 threads max (auto-adjusted inside)
        double t2=timer_elapsed_ms(&timer);

        timer_start(&timer);
        cuda_matmul(A,B,C3,N);
        double t3=timer_elapsed_ms(&timer);

        printf("Single-threaded: %.3f ms\nPer-row threads: %.3f ms\nCUDA GPU: %.3f ms\n",t1,t2,t3);
        fprintf(out,"%d,%.6f,%.6f,%.6f\n",N,t1,t2,t3);

        free(A); free(B); free(C1); free(C2); free(C3);
    }

    fclose(out);
    printf("\nAll results saved to results.csv\n");
    free(sizes);
    return 0;
}
