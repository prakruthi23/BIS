import random

# Step 1: Define the problem
def fitness_function(x):
    # The mathematical function to optimize (maximize x^2)
    return x**2

# Step 2: Initialize parameters
population_size = 100  # Number of individuals in the population
mutation_rate = 0.1    # Probability of mutation
crossover_rate = 0.8   # Probability of crossover
num_generations = 50   # Number of generations
lower_bound = -10      # Lower bound of the search space
upper_bound = 10       # Upper bound of the search space

# Step 3: Create initial population
def generate_population(size, lower_bound, upper_bound):
    # Generate random individuals within the specified bounds
    return [random.uniform(lower_bound, upper_bound) for _ in range(size)]

# Step 4: Evaluate fitness of each individual
def evaluate_population(population):
    # Calculate fitness for each individual
    return [fitness_function(ind) for ind in population]

# Step 5: Selection - Roulette wheel selection
def select_parents(population, fitnesses):
    # Select two parents based on their fitness (probabilistic)
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    parent1 = random.choices(population, weights=selection_probs, k=1)[0]
    parent2 = random.choices(population, weights=selection_probs, k=1)[0]
    return parent1, parent2

# Step 6: Crossover
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        # Perform crossover
        crossover_point = random.random()  # Randomly choose a crossover point
        child1 = crossover_point * parent1 + (1 - crossover_point) * parent2
        child2 = crossover_point * parent2 + (1 - crossover_point) * parent1
        return child1, child2
    else:
        # No crossover, return parents as offspring
        return parent1, parent2

# Step 7: Mutation
def mutate(individual, mutation_rate, lower_bound, upper_bound):
    if random.random() < mutation_rate:
        # Randomly mutate within the search space
        individual = random.uniform(lower_bound, upper_bound)
    return individual

# Step 8: Iteration - Main GA loop
def genetic_algorithm():
    # Generate initial population
    population = generate_population(population_size, lower_bound, upper_bound)

    for generation in range(num_generations):
        # Evaluate fitness of the population
        fitnesses = evaluate_population(population)

        # Create a new population
        new_population = []
        while len(new_population) < population_size:
            # Select parents
            parent1, parent2 = select_parents(population, fitnesses)
            
            # Perform crossover to create offspring
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            
            # Apply mutation to offspring
            child1 = mutate(child1, mutation_rate, lower_bound, upper_bound)
            child2 = mutate(child2, mutation_rate, lower_bound, upper_bound)
            
            # Add offspring to the new population
            new_population.extend([child1, child2])

        # Update population with the new generation
        population = new_population[:population_size]

    # Step 9: Output the best solution
    # Evaluate the final population fitness
    final_fitnesses = evaluate_population(population)
    # Find the best individual
    best_individual = population[final_fitnesses.index(max(final_fitnesses))]
    best_fitness = max(final_fitnesses)
    return best_individual, best_fitness

# Running the Genetic Algorithm
best_solution, best_fitness = genetic_algorithm()
print(f"Best solution found: {best_solution}")
print(f"Fitness of the best solution: {best_fitness}")
