# Cooperative Particle Swarm Optimization - Integrating Project

This project computes for the matrix vector product of a given matrix and vector, and outputs it to a vector.

Group Members:

Aaron Gabrielle Dichoso

Luis Miguel Antonio Razon

John Kirsten Espiritu

There are 4 versions of the computation performed to compare execution time between methods:
1. C kernel - Only in C
2. ASM Non SIMD - Using Assembly Language, without SIMD operations
3. ASM XMM SIMD - Using Assembly Language, with XMM SIMD operations
4. ASM YMM SIMD - Using Assembly Language, with YMM SIMD operations

# A. Execution Output and Correctness Check

Below are screenshots of the program execution. Each kernel was executed 30 times to obtain the average execution time. These results can be used in performance_results.csv:

The C kernel served as the standard of checking the implementation of the ASM kernels. The values in the outputs of the ASM kernels have to be equal to the results in the C kernel. This checking was performed on the first 5 iterations of every kernel, in every tested matrix size:

## N = 1024 (2^10)
### C & ASM (Non SIMD)
![](Figures/Tests/1024NONSIMD.png)
### ASM (XMM SIMD)
![](Figures/Tests/1024XMM.png)
### ASM (YMM SIMD)
![](Figures/Tests/1024YMM.png)

## N = 8192 (2^13)
### C & ASM (Non SIMD)
![](Figures/Tests/8192NONSIMD.png)
### ASM (XMM SIMD)
![](Figures/Tests/8192XMM.png)
### ASM (YMM SIMD)
![](Figures/Tests/8192YMM.png)

## N = 32768 (2^15)
### C & ASM (Non SIMD)
![](Figures/Tests/32768NONSIMD.png)
### ASM (XMM SIMD)
![](Figures/Tests/32768XMM.png)
### ASM (YMM SIMD)
![](Figures/Tests/32768YMM.png)

# B. Execution Times
## Summary of Results 
### Average Execution Times
The table below shows the average execution time of each kernel function according to matrix size (N):
![](Figures/Average%20Execution%20Times.png)

Here is the same data, visualized using a bar graph:
![](Figures/Average%20Execution%20Time%20Graph.png)

### Function Speedup
The Speedup of utilizing ASM kernel functions compared to the performance of the C kernel was also obtained by dividing the average execution time of the C kernel with the average execution time of the ASM kernel functions.
This resulted in the following results:
![](Figures/Speedups%20per%20Function%20according%20to%20Matrix%20Size.png)

As expected, all ASM kernel functions obtained a speedup value greater than 1, indicating that our implementation of the matrix vector product in the ASM kernels consistently obtained faster execution times compared to the C kernel.

Additionally, the XMM kernel version obtained a higher speedup value compared to the non-SIMD kernel version, and the YMM kernel version obtained the highest speedup value compared to the other ASM kernel versions. We see these results as expected, as the utilization of SIMD operations allowing for a higher throughput of data to be processed simultaneously, as opposed to iterating for each element in the given matrix and vector to compute for the matrix-vector product. The XMM kernel function can compute for the dot product of 4 elements in the matrix in one iteration, while the YMM kernel can process 8 elements in one iteration.

# D. Boundary Checking (N = 15) of Outputs
To ensure the correctness of values regardless of any valid matrix size, the YMM and XMM kernels implemented boundary checking, using non-SIMD processing of values for certain elements in a matrix when N is not divisible by 8 or 4, respectively. As such, it is recommended to pad the input matrix into the desired matrix size when using the YMM and XMM kernels.

![](Figures/Tests/BoundaryTest.png)

# E. Reflection
This project was an interesting one, as it really elucidated the group regarding the effectiveness of utilizing SIMD for parallelizable domains. In particular, we first had problems in deciding the structure of our matrix. We initially had our matrix in a row-major format, where the elements were contiguous across a row. Then, we planned to process one row at a time, but we were not satisfied with the implementation.

Thus, we had thought of partially processing multiple rows at a time. The partial result would be stored while we had processed in a per-column basis. Thus, we had changed our matrix implementation to a column-major format, where the addresses of elements were contiguous along a column. This design was what allowed us to create the YMM and XMM kernel versions that we had implemented in this program.

