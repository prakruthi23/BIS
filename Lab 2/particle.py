import numpy as np

# Rastrigin function
def rastrigin_function(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Particle class representing a particle in the swarm
class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)  # Initialize position
        self.velocity = np.random.uniform(-1, 1, dim)                 # Initialize velocity
        self.best_position = np.copy(self.position)                   # Best position (personal best)
        self.best_value = float('inf')                                # Best value (personal best)
        self.current_value = float('inf')                             # Current value of the function

# PSO optimizer class
class PSO:
    def __init__(self, func, dim, bounds, num_particles, max_iter):
        self.func = func                          # Objective function
        self.dim = dim                            # Dimensionality
        self.bounds = bounds                      # Bounds for the variables
        self.num_particles = num_particles        # Number of particles
        self.max_iter = max_iter                  # Maximum iterations
        self.swarm = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best_position = np.zeros(dim) # Global best position
        self.global_best_value = float('inf')     # Global best value

    def optimize(self):
        w = 0.5    # Inertia weight
        c1 = 1.5   # Cognitive parameter (personal influence)
        c2 = 1.5   # Social parameter (swarm influence)

        for iteration in range(self.max_iter):
            for particle in self.swarm:
                # Evaluate current particle's fitness
                particle.current_value = self.func(particle.position)
                
                # Update personal best if the current value is better
                if particle.current_value < particle.best_value:
                    particle.best_value = particle.current_value
                    particle.best_position = np.copy(particle.position)
                
                # Update global best if the current value is better
                if particle.current_value < self.global_best_value:
                    self.global_best_value = particle.current_value
                    self.global_best_position = np.copy(particle.position)

            # Update velocity and position for each particle
            for particle in self.swarm:
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                cognitive_velocity = c1 * r1 * (particle.best_position - particle.position)
                social_velocity = c2 * r2 * (self.global_best_position - particle.position)
                
                particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity
                particle.position += particle.velocity
                
                # Enforce bounds
                particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

            # Output the progress
            print(f"Iteration {iteration+1}/{self.max_iter}, Global Best Value: {self.global_best_value}")

        return self.global_best_position, self.global_best_value

# Parameters
dim = 10  # Number of dimensions
bounds = [-5.12, 5.12]  # Bounds for the Rastrigin function
num_particles = 30  # Number of particles
max_iter = 10  # Maximum number of iterations

# PSO optimization
pso = PSO(rastrigin_function, dim, bounds, num_particles, max_iter)
best_position, best_value = pso.optimize()

print(f"\nBest Position: {best_position}")
print(f"Best Value: {best_value}")
