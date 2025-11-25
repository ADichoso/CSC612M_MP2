# Cooperative Particle Swarm Optimization - Integrating Project

This project performs Cooperative Particle Swarm Optimization (CPSO) based on given functions.

Group Members:

Aaron Gabrielle Dichoso

Luis Miguel Antonio Razon

John Kirsten Espiritu

## Project Details

CPSO is an optimization algorithm that searches for optimal values in a given function by distributing agents (called particles) across each swarms, with each swarm assigned to optimize 1 dimension.
Each agent is assigned to search for the optimal value in 1 dimension, collaborating with other agents to find the overall optimal value in the entire space.

This Project includes C and CUDA Kernels to compare the effectiveness of SIMT Parallelism Techniques using CUDA to improve the performance of the Sequential implementation of CPSO.

Through the use of CUDA, the algorithm will be parallelized by dedicating 1 block to every swarm in the function.
Each block will also be assigned 1 thread to every particle assigned to a given dimension.

In other words, every particle is assigned its own thread in the CUDA Kernel Implementation of CPSO.

Additional threads will also be included to handle the synchronization of global bests across each dimension.


# A. Execution Output and Correctness Check

Below are screenshots of the program execution. 
Each kernel was executed 30 times to obtain the average execution time. 
Additionally, each run of CPSO used 100 particles in each dimension and stopped after 500 iterations.

These results can be used in the Results directory.

The C kernel served as the standard of checking the implementation of the CUDA kernels. 
The values in the outputs of the CUDA kernels have to be equal to the results in the C kernel.

The C and CUDA Kernels were tested using 3 formulas with known global minimums:
1. The Sphere Function
2. The Ackley Function
3. The Rosenbrock Function

Additionally, the C and CUDA kernels tested variants of each formula containing 4, 256, 1024, and 8192 dimensions.
Due to time constraints, however, the C kernels were not tested with 8192 dimensions.

## C Kernel Executions
![](Figures/C_all.png)

## CUDA Kernel Executions
### N = 4 (2^2)
#### Sphere
![](Figures/CUDA_Sphere_4.png)
#### Ackley
![](Figures/CUDA_Ackley_4.png)
#### Rosenbrock
![](Figures/CUDA_Rosenbrock_4.png)

### N = 256 (2^8)
#### Sphere
![](Figures/CUDA_Sphere_256.png)
#### Ackley
![](Figures/CUDA_Ackley_256.png)
#### Rosenbrock
![](Figures/CUDA_Rosenbrock_256.png)

### N = 1024 (2^10)
#### Sphere
![](Figures/CUDA_Sphere_2(10).png)
#### Ackley
![](Figures/CUDA_Ackley_2(10).png)
#### Rosenbrock
![](Figures/CUDA_Rosenbrock_2(10).png)

### N = 8192 (2^13)
#### Sphere
![](Figures/CUDA_Sphere_2(13).png)
#### Ackley
![](Figures/CUDA_Ackley_2(13).png)
#### Rosenbrock
![](Figures/CUDA_Rosenbrock_2(13).png)

# B. Execution Times
## Summary of Results 
### Average Execution Times
The table below shows the average execution time of each kernel according to dimension size and function:
![](Figures/AverageExecutionTimes.png)

Here is the same data, visualized using a bar graph:
![](Figures/c_vs_cuda_avg_times.png)

### Function Speedup
The Speedup of utilizing ASM kernel functions compared to the performance of the C kernel was also obtained by dividing the average execution time of the C kernel with the average execution time of the ASM kernel functions.
This resulted in the following results:
![](Figures/cuda_vs_c_plot.png)

As expected, all ASM kernel functions obtained a speedup value greater than 1, indicating that our implementation of the matrix vector product in the ASM kernels consistently obtained faster execution times compared to the C kernel.

Additionally, the XMM kernel version obtained a higher speedup value compared to the non-SIMD kernel version, and the YMM kernel version obtained the highest speedup value compared to the other ASM kernel versions. We see these results as expected, as the utilization of SIMD operations allowing for a higher throughput of data to be processed simultaneously, as opposed to iterating for each element in the given matrix and vector to compute for the matrix-vector product. The XMM kernel function can compute for the dot product of 4 elements in the matrix in one iteration, while the YMM kernel can process 8 elements in one iteration.

# E. Reflection
This project was an interesting one, as it really elucidated the group regarding the effectiveness of utilizing SIMD for parallelizable domains. In particular, we first had problems in deciding the structure of our matrix. We initially had our matrix in a row-major format, where the elements were contiguous across a row. Then, we planned to process one row at a time, but we were not satisfied with the implementation.

Thus, we had thought of partially processing multiple rows at a time. The partial result would be stored while we had processed in a per-column basis. Thus, we had changed our matrix implementation to a column-major format, where the addresses of elements were contiguous along a column. This design was what allowed us to create the YMM and XMM kernel versions that we had implemented in this program.

