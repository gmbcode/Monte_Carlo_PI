# Monte Carlo Pi Estimation with CUDA

A simple GPU implementation of Monte Carlo method for estimating the value of π (pi) using CUDA.  

Will be improved later for higher performance and GPU utilisation.
## Overview

This program uses the Monte Carlo method to estimate π by randomly sampling points in a unit square and counting how many fall inside a unit circle. The ratio of points inside the circle to total points approximates π/4, so π ≈ 4 × (points_inside_circle / total_points).

## Algorithm

1. Generate random points (x, y) in the range [-1, 1] × [-1, 1]
2. Check if each point is inside the unit circle: x² + y² ≤ 1
3. Count the total points inside the circle
4. Estimate π = 4 × (inside_count / total_points)

## Performance

- **Threads**: 2,048 blocks × 256 threads = 524,288 parallel threads
- **Samples per thread**: 100,000
- **Total samples**: 52,428,800,000 (52.4 billion)
- **Expected accuracy**: 4-5 decimal places

## Requirements

- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit 10.0 or later
- C++14 compatible compiler

## Compilation

```bash
nvcc -O3 monte_carlo_pi.cu -lcurand -o monte_carlo_pi
```

### CMake (Optional)

If using the provided CMakeLists.txt:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Usage

```bash
./monte_carlo_pi
```

## Sample Output

```
Total number of points is : 5242880000
Number of points inside the circle are : 41177446391
Estimated Pi: 3.14159
```

## Code Structure

### Key Components

- **`setup_kernel`**: Initializes CUDA random number generators for each thread
- **`simulate_mc_pi`**: Main Monte Carlo simulation kernel
- **`readerin`**: Reads the final count from GPU memory
- **Atomic operations**: Ensures thread-safe counting

### Memory Management

- Uses `curandState` for high-quality GPU random number generation
- Employs atomic operations (`atomicAdd`) to avoid race conditions
- Proper GPU memory allocation and cleanup

## Technical Details

### Random Number Generation
- Uses CUDA's `curand` library for GPU-optimized random numbers
- Each thread has its own random state to avoid contention
- Generates uniform random numbers in [-1, 1] range

### Parallelization Strategy
- Each thread generates 100,000 samples independently
- Local counting reduces atomic operation overhead
- Single atomic add per thread minimizes contention

### Accuracy vs Performance
- More samples = better accuracy but longer runtime
- Accuracy improves with √(samples) - so 100× more samples = 10× better accuracy
- Current configuration balances speed and precision

## Customization

### Adjusting Sample Count
Modify the loop in `simulate_mc_pi`:
```cuda
for (unsigned long long i = 0; i < 100000; i++) {  // Change 100000
```

### Changing Thread Configuration
Modify the kernel launch parameters:
```cuda
simulate_mc_pi<<<2048, 256>>>(dStates);  // blocks, threads_per_block
```

### Improving Accuracy
- Increase samples per thread for better precision
- Increase total number of threads/blocks

## Common Issues

1. **Compilation Error**: Make sure CUDA toolkit is properly installed
2. **Runtime Error**: Ensure your GPU supports the specified compute capability
3. **Poor Accuracy**: Increase sample count or check random number generation
4. **Memory Errors**: Verify GPU has sufficient memory for the allocated arrays

## Performance Tuning

1. **Increase samples per thread** to reduce atomic operation overhead
2. **Tune block size** based on your GPU architecture
3. **Use shared memory reduction** for even better performance
4. **Profile with `nvprof`** to identify bottlenecks

## Mathematical Background

The Monte Carlo method relies on the Law of Large Numbers. As the number of samples approaches infinity, the ratio of points inside the circle converges to π/4.

**Circle area**: π × r² = π (for unit circle)  
**Square area**: (2r)² = 4 (for square containing unit circle)  
**Ratio**: π/4  
**Therefore**: π = 4 × (points_inside / total_points)

## License

This code is licensed under the [MIT License](LICENSE)

## Contributing

Feel free to submit issues or pull requests to improve the implementation or add new features like:
- Different random number generators
- Shared memory optimization
- Precision analysis tools