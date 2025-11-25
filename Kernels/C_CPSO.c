
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#define RNG_UNIFORM() ((double)rand() / (double)RAND_MAX)
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

typedef double (*ObjectiveFunction)(size_t, double *);

//==============================================================

//==============================================================
//                     UTILITY FUNCTIONS
//==============================================================
static inline double square(double b)
{
  return b * b;
}

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
//==============================================================

//==============================================================
//              ALGORITHM-DEFINED FUNCTIONS
//==============================================================
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
 */
void updateVelocity(Particle *particle, double globalBest, int currentIter, int maxIter)
{
  double w = 1.0 - ((double)currentIter / maxIter); // intertia weight, linear scaling
  double c1 = 2; // acceleration coefficient 1
  double c2 = 2; // acceleration coefficient 2
  double r1 = RNG_UNIFORM();
  double r2 = RNG_UNIFORM();
  double updatedVelocity = (w * particle->velocity) + (c1 * r1 * (particle->personalBest - particle->position)) + (c2 * r2 * (globalBest - particle->position));

  particle->velocity = clamp(updatedVelocity, VELOCITY_MIN, VELOCITY_MAX);
}

/**
 * Position update function
 */
void updatePosition(Particle *particle)
{
  particle->position += particle->velocity;
  particle->position = clamp(particle->position, POSITION_MIN, POSITION_MAX);
}

void particleAction(Swarm *swarm, int swarmIdx, Particle *particle, double *result, size_t n, int iter, int maxIter, ObjectiveFunction objectiveFunc)
{
  b(result, n, swarmIdx, particle->position, result);

  double fitness = objectiveFunc(n, result);

  if (fitness < particle->personalBestFitness)
  {
    particle->personalBestFitness = fitness;
    particle->personalBest = particle->position;
  }

  if (particle->personalBestFitness < swarm->globalBestFitness)
  {
    swarm->globalBestFitness = particle->personalBestFitness;
    swarm->globalBest = particle->personalBest;
  }

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
 * CPSO-S Algorithm.
 * @param n dimensions
 * @param s particle size per swarm
 * @param objectiveFunction objective function where the positions of the particles will be evaluated (minima search)
 * @param maxIterations maximum iterations for when to stop the search
 * @param threshold when this value is reached (based on the objectiveFunction), search immediately stops
 *
 * @result @param out pointer, stores the best coordinate found
 */
void CPSO_S(size_t n, int s, ObjectiveFunction objectiveFunc, int maxIterations, double threshold, double *out)
{
  Swarm swarms[n];

  // 1. Initialize b(j, z)
  for (int i = 0; i < n; i++)
  {
    out[i] = 0;
  }

  // 2. Initialize n one-d swarms
  for (int i = 0; i < n; i++)
  {
    swarms[i].globalBest = 0;
    swarms[i].globalBestFitness = __DBL_MAX__;
    swarms[i].particles = (Particle *)malloc(s * sizeof(Particle));

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
    // for each swarm i ∈ [1...n]
    for (int i = 0; i < n; i++)
    {
      // Copy out. this way, the global values are updated after all has completed an iteration.
      double *coordinateVector = malloc(n * sizeof(double));
      memcpy(coordinateVector, out, n * sizeof(double));
      for (int j = 0; j < s; j++)
      {
        particleAction(&(swarms[i]), i, &(swarms[i].particles[j]), coordinateVector, n, iter, maxIterations, objectiveFunc);
      }
      free(coordinateVector);
    }

    // update out
    for (int i = 0; i < n; i++)
    {
      out[i] = swarms[i].globalBest;
    }
  }

  for (int i = 0; i < n; i++)
  {
    free(swarms[i].particles);
  }
}

static void assertClose(const char *label, double actual, double expected, double tolerance)
{
  if (fabs(actual - expected) > tolerance)
  {
    fprintf(stderr, "%s failed: expected %.9f but got %.9f\n", label, expected, actual);
    exit(EXIT_FAILURE);
  }
}

int main(void)
{
  srand(time(NULL));

  double x[] = {0, 0, 0, 0};
  double x2[] = {1, 1, 1, 1};

  printf("Rosenbrock at (0, 0, 0, 0) = %f\n", rosenbrock(4, x));
  printf("Rosenbrock at (1,1,1,1) (minima) = %f\n", rosenbrock(4, x2));
  printf("Sphere at (0, 0, 0, 0) (minima) = %f\n", sphere(4, x));
  printf("Sphere at (1,1,1,1) = %f\n", sphere(4, x2));
  printf("Ackley at (0, 0, 0, 0) (minima) = %f\n", ackley(4, x));
  printf("Ackley at (1,1,1,1) = %f\n", ackley(4, x2));

  // ======================= TESTING PROPER ====================================
  size_t dimensions[5] = {4, 256, 1 << 10};
  size_t dimensionCount = 3;

  ObjectiveFunction functions[3] = {sphere, ackley, rosenbrock};
  size_t functionCount = 3;

  int particles = 100;
  int maxIterations = 500;

  double threshold = 0.25;

  clock_t start, end;
  size_t loops = 30;

  FILE *file;
  const char *filename = "result_c.csv";

  file = fopen(filename, "w");

  // Check if the file was opened successfully
  if (file == NULL)
  {
    printf("Error opening file %s\n", filename);
    exit(1); // Exit the program with an error code
  }

  fprintf(file, "Function,Dimension,Loop,Time Taken,Fitness\n");

  fclose(file);
  // For each objective function,
  for (size_t i = 0; i < functionCount; i++)
  {
    // For each dimension,
    for (size_t j = 0; j < dimensionCount; j++)
    {
      // Get the total execution time
      double elapse, time_taken;
      elapse = 0.0f;
      // Run the program 30 times
      for (size_t k = 0; k < loops; k++)
      {
        size_t n = dimensions[j];
        file = fopen(filename, "a");

        double *result = (double *)calloc(n, sizeof(double));
        ObjectiveFunction objectiveFunction = functions[i];

        start = clock();
        CPSO_S(n, particles, objectiveFunction, maxIterations, threshold, result);
        end = clock();

        time_taken = ((double)(end - start)) * 1E3 / CLOCKS_PER_SEC;
        elapse = elapse + time_taken;

        fprintf(file, "%lu,%lu,%lu,%f,%f\n", i, n, k, time_taken, objectiveFunction(n, result));
        fclose(file);
        free(result);
      }

      printf("Function (in C) average time for %lu loops is %f milliseconds to execute function %lu at %lu dimensions \n", loops, elapse / loops, i, dimensions[j]);
    }
  }

  return 0;
}
