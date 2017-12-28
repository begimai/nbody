#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>
#define PI 3.14159265358979323846

typedef struct Planet_s {
	float x, y; 	// position
	float ax, ay; // acceleration
	float vx, vy; // velocity
	float mass;		// mass
} Planet;

float generateRandom() {
  return (float)(rand() / RAND_MAX);
}

Planet  *initializePlanets (int nPlanets) {
  srand(time(NULL));
	const float accelerationScale = 100.0;
	Planet *planets = (Planet *) malloc(sizeof(*planets) * nPlanets);
	for (int i = 0; i < nPlanets; i++) {
		float angle = 	((float) i / nPlanets) * 2.0 * PI + ((generateRandom() - 0.5) * 0.5);
		float initialMass = 1000;
		Planet object = {
			.x = generateRandom(), .y = generateRandom(),
			.vx = cos(angle) * accelerationScale * generateRandom(),
			.vy = sin(angle) * accelerationScale * generateRandom(),
			.mass = initialMass * generateRandom() + initialMass * 0.5
		};
    planets[i] = object;
  }
  return planets;
}

void integrate(Planet *planet, float deltaTime) {
	planet->vx += planet->ax * deltaTime;
	planet->vy += planet->ay * deltaTime;
	planet->x += planet->vx * deltaTime;
	planet->y += planet->vy * deltaTime;
}

void calculateNewtonGravityAcceleration(Planet *a, Planet *b, float *ax, float *ay) {
	float softening = 10000;
	float distanceX = b->x - a->x;
	float distanceY = b->y - a->y;
	float vectorDistance = a->x * a->x + a->y * a->y + softening;
	float vectorDistanceCubed = vectorDistance * vectorDistance * vectorDistance;
  float inverse = 1.0 / sqrt(vectorDistanceCubed);
  float scale = b->mass * inverse;
	*ax = (distanceX * scale);
	*ay = (distanceY * scale);
}

void simulateWithBruteforce(int rank, int totalPlanets, int nPlanets, Planet *planets, Planet *local_planets, float dt) {
	for(size_t i = 0; i < nPlanets; i++) {
		float total_ax = 0, total_ay = 0;
		for (size_t j = 0; j < totalPlanets; j++) {
			if (j == nPlanets * rank + i) {
				continue;
			}
			float ax, ay;
			calculateNewtonGravityAcceleration(&local_planets[i], &planets[j], &ax, &ay);
			total_ax += ax;
			total_ay += ay;
		}
		local_planets[i].ax = total_ax;
		local_planets[i].ay = total_ay;
		integrate(&local_planets[i], dt);
	}
}

int main(int argc, char **argv) {

	int nPlanets = 80;
	float simulationTime = 1.0;
	float dt = 0.1;
	if (argc > 3) {
		nPlanets = atoi(argv[1]);
		simulationTime = atof(argv[2]);
		dt = atof(argv[3]);
	}
	double parallel_average_time = 0.0;

  MPI_Init(&argc, &argv);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Datatype dt_planet;
	MPI_Type_contiguous(7, MPI_FLOAT, &dt_planet);
	MPI_Type_commit(&dt_planet);

	Planet *planets = initializePlanets(nPlanets);
  if (rank == 0) {
    parallel_average_time -= MPI_Wtime();
  }
  size_t items_per_process = nPlanets / world_size;
	Planet *local_planets = (Planet *) malloc(sizeof(*local_planets) * items_per_process);

  MPI_Scatter(
	  planets,
	  items_per_process,
	  dt_planet,
	  local_planets,
	  items_per_process,
	  dt_planet,
	  0,
	  MPI_COMM_WORLD
  );

	simulateWithBruteforce(rank, nPlanets, items_per_process, planets, local_planets, dt);
	Planet *gathered_planets = NULL;
	if (rank == 0) {
		gathered_planets = (Planet *) malloc(sizeof(*gathered_planets) * nPlanets);
	}
	MPI_Gather(
	  local_planets,
	  items_per_process,
	  dt_planet,
	  gathered_planets,
	  items_per_process,
	  dt_planet,
	  0,
	  MPI_COMM_WORLD
  );

	if (rank == 0) {
	  parallel_average_time += MPI_Wtime();
	  printf("%d\n", nPlanets);
	  printf("%.5f\n", simulationTime);
	  printf("%.5f\n", dt);
	  //printf("one cicle time %.5f\n", parallel_average_time);
	  for (size_t i = 0; i < nPlanets; ++i) {
	  	printf("%.5f %.5f\n", planets[i].x, planets[i].y);
	  	printf("%.5f %.5f\n", planets[i].ax, planets[i].ay);
	  	printf("%.5f %.5f\n", planets[i].vx, planets[i].vy);
	  	printf("%.5f\n", planets[i].mass);
	  }
	  for (float j = 0.0; j < simulationTime; j += dt) {
	  	for (size_t i = 0; i < nPlanets; ++i) {
	  		printf("%.5f %.5f\n", gathered_planets[i].ax, gathered_planets[i].ay);
	  		integrate(&gathered_planets[i], dt);
	  	}
	  }
  }
	if (planets != NULL) {
    free(planets);
  }
  free(local_planets);
  if (gathered_planets != NULL) {
    free(gathered_planets);
  }
  MPI_Finalize();
	return 0;
}
