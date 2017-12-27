#include <stdio.h>


struct Position {
    int x;
    int y;
};

struct Acceleration {
    int somenumber1;
    int somenumber2;
};

 struct Planet {
     struct Position position;
     int mass;
     int velocity;
     struct Acceleration acceleration;
 };

void GenerateDebugData(int planetCount) {
    printf("you entered %d", planetCount);
}

void calculateNewtonGravityAcceleration(body *a, body *b, float *ax, float *ay) {
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

void SimulateWithBruteforce (int rank, int totalBodies, int nBodies, body *bodies, body *local_bodies, float dt) {
    for(size_t i = 0; i < nBodies; i++) {
        float total_ax = 0, total_ay = 0;
        for (size_t j = 0; j < totalBodies; j++) {
            if (j == nBodies * rank + i) {
                continue;
            }
            float ax, ay;
            calculateNewtonGravityAcceleration(&local_bodies[i], &bodies[j], &ax, &ay);
            total_ax += ax;
            total_ay += ay;
        }
        local_bodies[i].ax = total_ax;
        local_bodies[i].ay = total_ay;
        integrate(&local_bodies[i], dt);
    }
}


int main() {
    int planetCount;
    printf("Enter a number of planets: ");
    scanf("%d", &planetCount);
    GenerateDebugData(planetCount);
}
