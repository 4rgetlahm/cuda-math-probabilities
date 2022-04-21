
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>


#define ATTEMPT_COUNT 100000

/* 
#define N 5
#define M 13
*/

//GPU kernel for generating answer to probability problem where you need to find if there will be any bus stops that won't have any people leaving the bus
/*__global__ void generate_random_numbers(int max, unsigned long long* result) {

    int tId = threadIdx.x + (blockIdx.x * blockDim.x);

    curandState state;
    curand_init((unsigned long long)clock() + tId, 0, 0, &state);

    short stops[N] = {0};

    for (short i = 0; i < M; i++) {
        unsigned long long rand_number = ceilf(curand_uniform(&state) * max);
        stops[rand_number-1]++;
    }

    short count = 0;

    for (short i = 0; i < N; i++) {
        if (stops[i] == 0) {
            ++count;
        }
    }
    if (count == 2) {
        atomicAdd(result, 1);
        return;
    }
}*/

/*
#define FIRST_SHOOTER_CHANCE 0.42
#define SECOND_SHOOTER_CHANCE 0.34

__global__ void generate_shooting_range_probabilities(unsigned long long* result) {
    int tId = threadIdx.x + (blockIdx.x * blockDim.x);

    curandState state;
    curand_init((unsigned long long)clock() + tId, 0, 0, &state);

    //printf("%f\n%f\n", FIRST_SHOOTER_CHANCE, SECOND_SHOOTER_CHANCE);
    unsigned int first_shots = 0, second_shots = 0;
    while(true){
        ++first_shots;
        if (curand_uniform(&state) <= FIRST_SHOOTER_CHANCE) {
            break;
        }
    }
    while(true){
        ++second_shots;
        if (curand_uniform(&state) <= SECOND_SHOOTER_CHANCE) {
            break;
        }
    }
    
    if (first_shots > second_shots) {
        atomicAdd(result, 1);
    }
}
*/

/*
#define HEAD_RATE 0.58
#define ROLL_AMOUNT 10

__global__ void generate_movement_roll_probabilities(unsigned long long* result) {
    int tId = threadIdx.x + (blockIdx.x * blockDim.x);

    curandState state;
    curand_init((unsigned long long)clock() + tId, 0, 0, &state);

    short pos = 0;
    for (short i = 1; i <= ROLL_AMOUNT; ++i) {
        if (curand_uniform(&state) <= HEAD_RATE) {
            ++pos;
        }
        else {
            --pos;
        }
    }
    if (pos == 0) {
        atomicAdd(result, 1);
    }
}*/

#define URN1_WHITE_COUNT 13
#define URN1_BLACK_COUNT 33
#define URN2_WHITE_COUNT 33
#define URN2_BLACK_COUNT 7

__global__ void generate_urn_take_white_two_times(unsigned long long* result) {

    int tId = threadIdx.x + (blockIdx.x * blockDim.x);

    curandState state;
    curand_init((unsigned long long)clock() + tId, 0, 0, &state);
    if (curand_uniform(&state) <= 0.5) { //URN1
        //if (curand_uniform(&state) <= ((double) URN1_WHITE_COUNT / ((double) URN1_WHITE_COUNT + (double) URN1_BLACK_COUNT))) {
            if (curand_uniform(&state) >= ((double) URN1_WHITE_COUNT / ((double) URN1_WHITE_COUNT + (double) URN1_BLACK_COUNT))) {
                atomicAdd(result, 1);
                return;
            }
        //}
    }
    else {
        //if (curand_uniform(&state) <= ((double) URN2_WHITE_COUNT / ((double) URN2_WHITE_COUNT + (double) URN2_BLACK_COUNT) )) {
            if (curand_uniform(&state) >= ((double) URN2_WHITE_COUNT / ((double) URN2_WHITE_COUNT + (double) URN2_BLACK_COUNT) )) {
                atomicAdd(result, 1);
                return;
            }
        //}
    }

}


int main() {
    unsigned long long result = 0;
    unsigned long long *gpu_result;

    cudaMalloc((void**)&gpu_result, sizeof(unsigned long long));

    printf("%f\n", ((double)URN1_WHITE_COUNT / ((double) URN1_WHITE_COUNT + (double) URN1_BLACK_COUNT)));

    generate_urn_take_white_two_times <<<ATTEMPT_COUNT, 1024>>> (gpu_result);

    cudaMemcpy(&result, gpu_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    unsigned long long total = ATTEMPT_COUNT * (unsigned long long)1024;

    printf("%I64d\n", result);
    printf("%I64d\n", total);
    cudaFree(gpu_result);

    return 0;
}