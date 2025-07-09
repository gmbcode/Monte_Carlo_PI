#include <cuda_runtime.h>
#include <iostream>
#include <curand_kernel.h>
#include <cmath>
__device__ unsigned long long int countin = 0;
// TODO increase efficiency and utilisation 
__global__ void setup_kernel(curandState *state)
{
  auto id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(123456, id, 0, &state[id]);
}

__global__ void simulate_mc_pi(curandState *state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread's local counter
    unsigned long long local_inside = 0;
    
    // Generate random points
    for (unsigned long long i = 0; i < 100000; i++) {
        // Generate random point in unit square [-1, 1] x [-1, 1]
        double x = curand_uniform(&state[tid]) * 2.0 - 1.0;
        double y = curand_uniform(&state[tid]) * 2.0 - 1.0;
        
        // Check if point is inside unit circle
        if (x * x + y * y <= 1.0) {
            local_inside++;
        }
    }
    
    // Add to global counter atomically
    atomicAdd(&countin, local_inside);
}

__global__ void readerin(unsigned long long int* result){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = countin;
    }
}

int main() {
    curandState* dStates;
    cudaMalloc((void **) &dStates, sizeof(curandState) * 2048 * 256);  
    setup_kernel<<<2048,256>>>(dStates);
    cudaDeviceSynchronize();
    simulate_mc_pi<<<2048, 256>>>(dStates);
    cudaDeviceSynchronize();
    unsigned long long int* resultin;
    cudaMalloc(&resultin, sizeof(unsigned long long int));
    unsigned long long int counter_in;

    readerin<<<1, 1>>>(resultin);
    cudaDeviceSynchronize();
    cudaMemcpy(&counter_in, resultin, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaFree(resultin); 
    cudaFree(dStates);
    unsigned long long int total_points_sampled = 2048ULL * 256ULL * 100000ULL;
    std::cout << "Total number of points is : " << total_points_sampled << std :: endl;
    std::cout << "Number of points inside the circle are : " << counter_in << std :: endl;
    long double pi_estimate = 4.0 * (long double)counter_in / (long double)total_points_sampled;
    std::cout << "Estimated Pi : " << pi_estimate << std::endl;

    return 0;
}