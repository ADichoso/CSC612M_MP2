
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

#define POSITION_MIN -5
#define POSITION_MAX 5
#define VELOCITY_MAX 0.2 * POSITION_MAX
#define VELOCITY_MIN -VELOCITY_MAX // Similar to the experiment setup in the study for rosenbrack

//==============================================================
//                     STRUCT DEFINITIONS
//==============================================================
typedef struct
{
  double position;
  double velocity;
  double personalBest; // Position in dimension
  double personalBestFitness;
} Particle;

typedef struct
{
  Particle *particles;
  double globalBest;
  double globalBestFitness;
} Swarm;

typedef double (*ObjectiveFunction)(size_t, double *, double, size_t);

//==============================================================

//==============================================================
//                     UTILITY FUNCTIONS
//==============================================================
// CPU-function RNG_UNIFORM
static inline double RNG_UNIFORM()
{
  return (double)rand() / (double)RAND_MAX;
}
// CUDA-function RNG
__device__
static inline uint32_t mix32(uint32_t z) {
    z = (z ^ 61u) ^ (z >> 16);
    z *= 9u;
    z = z ^ (z >> 4);
    z *= 0x27d4eb2du;
    z = z ^ (z >> 15);
    return z;
}
__device__
static inline double CUDA_RNG_UNIFORM()
{
  unsigned long long t = clock64();
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t state = mix32((uint32_t)t ^ tid);

  const uint32_t A = 1664525u;
  const uint32_t C = 1013904223u;
  uint32_t x = state;
  x = A * x + C;        // wraps mod 2^32 automatically
  state = x;

  // Use the top 24 bits for a well-distributed uniform in [0,1)
  double out = ((x >> 8) & 0x00FFFFFF) * (1.0 / 16777216.0);
  return out;
}

__device__ __host__
static inline double square(double b)
{
  return b * b;
}

__device__
static inline double clamp(double value, double min, double max)
{
  if (value < min)
  {
    return min;
  }
  if (value > max)
  {
    return max;
  }
  return value;
}

static inline double randomInRange(double min, double max)
{
  return RNG_UNIFORM() * (max - min) + min;
}

//==============================================================

//==============================================================
//              OBJECTIVE FUNCTIONS
//==============================================================
/**
 * Creates an n-dimensional search space using the Rosenbrock function
 * @param n dimensions
 * @param x coordinates
 */

double rosenbrock(size_t n, double *x)
{
  double value = 0.0;

  // f(x) = ∑(i=1 to N−1) [100(x_{i+1} − x_i^2)^2 + (1 − x_i)^2]   where   x = (x_1, …, x_N) ∈ ℝ^N
  for (size_t i = 0; i < n - 1; ++i)
  { 
    value += 100.0 * square(x[i + 1] - square(x[i])) + square(1.0 - x[i]);
  }

  return value;
}

double sphere(size_t n, double *x)
{
  double value = 0.0f;
  for (size_t i = 0; i < n; i++)
  {
    value += square(x[i]);
  }

  return value;
}

double ackley(size_t n, double *x)
{
  double a = 20.0;
  double b = 0.2;
  double c = 2.0 * M_PI;

  double sumSq = 0.0;
  double sumCos = 0.0;

  for (int i = 0; i < n; i++)
  {
    sumSq += square(x[i]);
    sumCos += cos(c * x[i]);
  }

  double term1 = -a * exp(-b * sqrt(sumSq / n));
  double term2 = -exp(sumCos / n);

  return term1 + term2 + a + exp(1);
}

__device__ __host__
double evalRosenbrock(size_t n, double *x, double z, size_t swarmIdx)
{
  double value = 0.0;

  // f(x) = ∑(i=1 to N−1) [100(x_{i+1} − x_i^2)^2 + (1 − x_i)^2]   where   x = (x_1, …, x_N) ∈ ℝ^N
  for (size_t i = 0; i < n - 1; ++i)
  { 
    double xi = (i == swarmIdx) ? z : x[i];
    double xip1 = (i + 1 == swarmIdx) ? z : x[i + 1];
    value += 100.0 * square(xip1 - square(xi)) + square(1.0 - xi);
  }

  return value;
}
__device__ __host__
double evalSphere(size_t n, double *x, double z, size_t swarmIdx)
{
  double value = 0.0f;
  for (size_t i = 0; i < n; i++)
  {
    double xi = (i == swarmIdx) ? z : x[i];
    value += square(xi);
  }

  return value;
}

__device__ __host__
double evalAckley(size_t n, double *x, double z, size_t swarmIdx)
{
  double a = 20.0;
  double b = 0.2;
  double c = 2.0 * M_PI;

  double sumSq = 0.0;
  double sumCos = 0.0;

  for (int i = 0; i < n; i++)
  {
    double xi = (i == swarmIdx) ? z : x[i];
    sumSq += square(xi);
    sumCos += cos(c * xi);
  }

  double term1 = -a * exp(-b * sqrt(sumSq / n));
  double term2 = -exp(sumCos / n);

  return term1 + term2 + a + exp(1);
}

// Device function-pointer symbols for objective functions
__device__ ObjectiveFunction d_rosenbrock_ptr = evalRosenbrock;
__device__ ObjectiveFunction d_sphere_ptr     = evalSphere;
__device__ ObjectiveFunction d_ackley_ptr     = evalAckley;

//==============================================================

//==============================================================

//==============================================================
//              ALGORITHM-DEFINED FUNCTIONS
//==============================================================
__device__
void b(double *context, size_t n, int j, double z, double *result)
{
  for (int i = 0; i < n; i++)
  {
    if (i == j)
    {
      result[i] = z;
    }
    else
    {
      result[i] = context[i];
    }
  }
}

/**
 * Velocity update function
 * v_{i,j}(t+1) = w v_{i,j}(t) + c_1 r_{1,i}(t) [ y_{i,j}(t) - x_{i,j}(t) ] + c_2 r_{2,i}(t) [ \hat{y}_j(t) - x_{i,j}(t) ]
 *
 */
__device__
void updateVelocity(Particle *particle, double globalBest, int currentIter, int maxIter)
{
  double w = 1.0 - ((double)currentIter / maxIter); // intertia weight, linear scaling
  // printf("w: %f\n", w);
  double c1 = 2; // acceleration coefficient 1
  double c2 = 2; // acceleration coefficient 2
  double r1 = CUDA_RNG_UNIFORM();
  double r2 = CUDA_RNG_UNIFORM();
  double updatedVelocity = (w * particle->velocity) + (c1 * r1 * (particle->personalBest - particle->position)) + (c2 * r2 * (globalBest - particle->position));

  particle->velocity = clamp(updatedVelocity, VELOCITY_MIN, VELOCITY_MAX);
}

/**
 * Position update function
 *
 */
__device__
void updatePosition(Particle *particle)
{
  particle->position += particle->velocity;
  particle->position = clamp(particle->position, POSITION_MIN, POSITION_MAX);
}

__global__
void particleAction(Swarm *swarms, size_t n, int s, double *result, int iter, int maxIter, ObjectiveFunction objectiveFunc)
{

  // particle index per swarm
  size_t tid = threadIdx.x;
  size_t swarmIdx = blockIdx.x;

  Swarm *swarm = &swarms[swarmIdx];
  Particle *particle = &swarm->particles[(int)tid];

  // double* candidate = (double*)malloc(n * sizeof(double));
  // memcpy(candidate, result, n * sizeof(double));

  // no need to copy result since it is operating on a per swarm, so its values (apart from z) will be replaced but it will not matter.
  // b(result, n, swarmIdx, particle->position, candidate);

  double fitness = (*objectiveFunc)(n, result, particle->position, swarmIdx);

  // free(candidate);

  if (fitness < particle->personalBestFitness)
  {
    particle->personalBestFitness = fitness;
    particle->personalBest = particle->position;
  }

  // START OF ATOMIC CHECKING
  __syncthreads();

  // thread 0 finds global best
  if (tid == 0) {
    for (int i = 0; i < s; i++) {
      Particle *p = &swarm->particles[i];
      if (p->personalBestFitness < swarm->globalBestFitness) {
        swarm->globalBestFitness = p->personalBestFitness;
        swarm->globalBest = p->personalBest;
      }
    }
  }

  __syncthreads();
  // END OF ATOMIC CHECKING

  // stopping condition -- Not included to measure raw machine performance
  // if (swarms[i].globalBestFitness <= threshold)
  // {
  //   return;
  // }

  // PSO Updates
  updateVelocity(particle, swarm->globalBest, iter, maxIter);
  updatePosition(particle);

}

//==============================================================

/**
 * CPSO-S Algorithm using CUDA. The blocks represent the swarm and each threads represents a particle. Since a block can only have atmost 1024 threads, each swarm is limited to have at most 1024 particles.
 * @param n dimensions
 * @param s particle size per swarm
 * @param objectiveFunction objective function where the positions of the particles will be evaluated (minima search)
 * @param maxIterations maximum iterations for when to stop the search
 * @param threshold when this value is reached (based on the objectiveFunction), search immediately stops
 *
 * @result @param result pointer, stores the best coordinate found
 */
double* CPSO_S(size_t n, int s, ObjectiveFunction objectiveFunc, int maxIterations, double threshold)
{
  int device = -1;
  cudaGetDevice (&device);

  // blocks is per dimension.
  size_t numBlocks = n;
  // max 1024 particles per swarm
  size_t numThreads = (s < 1024) ? s : 1024;

  if (s > 1024) {
    printf("Particle exceeds max CUDA thread count. Particle count is limited to 1024.");
    s = 1024;
  }

  Swarm *swarms;
  cudaMallocManaged(&swarms, n * sizeof(Swarm));

  cudaMemAdvise(swarms, n * sizeof(Swarm), cudaMemAdviseSetPreferredLocation, device);
  cudaMemAdvise(swarms, n * sizeof(Swarm), cudaMemAdviseSetAccessedBy, device);


  for (size_t i = 0; i < n; i++) {
    cudaMallocManaged(&swarms[i].particles, s * sizeof(Particle));
    cudaMemAdvise(swarms[i].particles, s * sizeof(Particle), cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(swarms[i].particles, s * sizeof(Particle), cudaMemAdviseSetAccessedBy, device);
  }

  double *out;
  cudaMallocManaged(&out, n * sizeof(double));

  // 1. Initialize b(j, z)
  cudaMemset(out, 0, n * sizeof(double));


  // create CPU page memory and prefetch CPU- GPU
  cudaMemPrefetchAsync(swarms, n * sizeof(Swarm), cudaCpuDeviceId, NULL);
  for (size_t i = 0; i < n; i++) {
    cudaMemPrefetchAsync(swarms[i].particles, s * sizeof(Particle), cudaCpuDeviceId, NULL);
  }

  //  prefetch CPU- GPU
  cudaMemPrefetchAsync(swarms, n * sizeof(Swarm), device, NULL); //prefetches got rid of memory thrashes (CPU - GPU)
  for (size_t i = 0; i < n; i++) {
    cudaMemPrefetchAsync(swarms[i].particles, s * sizeof(Particle), device, NULL);
  }
  cudaMemPrefetchAsync(out, n * sizeof(double), device, NULL);

  cudaDeviceSynchronize();

  // 2. Initialize n one-d swarms
  for (int i = 0; i < n; i++)
  {
    swarms[i].globalBest = 0;
    swarms[i].globalBestFitness = __DBL_MAX__;

    for (int j = 0; j < s; j++)
    {
      swarms[i].particles[j].position = randomInRange(POSITION_MIN, POSITION_MAX);
      swarms[i].particles[j].velocity = randomInRange(VELOCITY_MIN, VELOCITY_MAX);

      swarms[i].particles[j].personalBest = swarms[i].particles[j].position;
      out[i] = swarms[i].particles[j].personalBest;
      swarms[i].particles[j].personalBestFitness = __DBL_MAX__;
    }
  }

  // 2. Main loop
  for (int iter = 0; iter < maxIterations; iter++)
  {
    particleAction<<<numBlocks, numThreads>>>(swarms, n, s, out, iter, maxIterations, objectiveFunc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Kernel launch failed at iter %d: %s\n", iter, cudaGetErrorString(err));
    }

    // update out, prefetch GPU-CPU (only need out)
    cudaMemPrefetchAsync(out, n * sizeof(double), cudaCpuDeviceId, NULL);
    cudaDeviceSynchronize(); 
    for (int i = 0; i < n; i++)
    {
      out[i] = swarms[i].globalBest;
    }
    cudaMemPrefetchAsync(out, n * sizeof(double), device, NULL);

  }

  cudaDeviceSynchronize(); 

  for (int i = 0; i < n; i++)
  {
    cudaFree(swarms[i].particles);
  }

  cudaFree(swarms);
  return out;
}

int main(void)
{

  srand(time(NULL));

  FILE *file;
  ObjectiveFunction objectiveFunction = nullptr;

  const char *filename = "result_cuda_sphere_4.csv";
  size_t n = 4;
  cudaMemcpyFromSymbol(&objectiveFunction, d_sphere_ptr, sizeof(ObjectiveFunction));


  int max = 500;
  size_t loops = 30;
  int particles = 100;


  file = fopen(filename, "w");

  // Check if the file was opened successfully
  if (file == NULL)
  {
    printf("Error opening file %s\n", filename);
    exit(1); // Exit the program with an error code
  }
  fprintf(file, "Function,Dimension,Loop,Fitness\n");



  for (size_t iter = 0; iter < loops; iter++) {
    double* result = nullptr;

    result = CPSO_S(n, particles, objectiveFunction, max, 0.25);
    cudaDeviceSynchronize();

    fprintf(file, "Sphere,%lu,%lu,%f\n", n, iter, sphere(n, result));
    cudaFree(result);

  }


  // printf("The position (");
  // for (int i = 0; i < n - 1; i++)
  // {
  //  printf("%f,", result[i]);
  // }
  // printf("%f) is the best found, with a fitness value of: %f after %d iterations", result[n - 1], sphere(n, result), max);
  // cudaFree(result);
  return 0;
}
