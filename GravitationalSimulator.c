#include <stdio.h>


struct Position {
    int x;
    int y;
};

struct Acceleration {
    int somenumber1;
    int somenumber2;
};

// struct Planet {
//     Position position;
//     int mass;
//     int velocity;
//     Acceleration acceleration;
// };

void GenerateDebugData(int planetCount) {
    printf("you entered %d", planetCount);
}

void SimulateWithBruteforce () {
    foreach (PlanetController planet in planets) {
        if (!planet.IsAlive)
            continue;

        Vector2 acceleration = Vector2.zero;
        foreach (PlanetController anotherPlanet in planets) {
            if (planet == anotherPlanet || !anotherPlanet.IsAlive)
                continue;

            acceleration += CalculateNewtonGravityAcceleration (planet, anotherPlanet);
        }

        planet.Acceleration = acceleration;
    }
}



Vector2 CalculateNewtonGravityAcceleration (IBody firstBody, IBody secondBody) {
    ++interactions;

    Vector2 acceleration = Vector2.zero;

    Vector2 galacticPlaneR = secondBody.Position - firstBody.Position;

    float distanceSquared = galacticPlaneR.sqrMagnitude + distanceSquared * distanceSquared * distanceSquared;
    float inverse = 1.0f / Mathf.Sqrt (distanceSquaredCubed);
    float scale = secondBody.Mass * inverse;

    acceleration += galacticPlaneR * scale;

    return acceleration;
}

int main() {
    int planetCount;
    printf("Enter a number of planets: ");
    scanf("%d", &planetCount);
    GenerateDebugData(planetCount);
}
