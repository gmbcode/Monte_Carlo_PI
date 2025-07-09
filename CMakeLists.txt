cmake_minimum_required(VERSION 3.18)
project(MonteCarloPi LANGUAGES CXX CUDA)

# Enable CUDA language
enable_language(CUDA)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 50;60;70;75;80;86;89;90)

# Create executable
add_executable(monte_carlo_pi main.cu)

# Set standards
set_property(TARGET monte_carlo_pi PROPERTY CXX_STANDARD 14)
set_property(TARGET monte_carlo_pi PROPERTY CUDA_STANDARD 14)

# toggle later if more files in project
#set_property(TARGET monte_carlo_pi PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# link CUDA runtime
target_link_libraries(monte_carlo_pi CUDA::cudart)


if(WIN32)
    set_property(TARGET monte_carlo_pi PROPERTY WIN32_EXECUTABLE FALSE)
endif()

if(UNIX)
    target_link_libraries(monte_carlo_pi m)  # Math library
endif()