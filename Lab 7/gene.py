import random
import numpy as np

class GeneExpressionProgramming:
    def _init_(self, population_size, generations, target_function, gene_length=2, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.target_function = target_function
        self.gene_length = gene_length
        self.mutation_rate = mutation_rate
        self.population = self.create_population()

    def create_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        return population

    def create_individual(self):
        # Create an individual with gene values for x and y (both in the range [0, 10])
        return [random.uniform(0, 10) for _ in range(self.gene_length)]

    def fitness(self, individual, x_val, y_val):
        # Calculate fitness as the difference between the target and the sum of x and y
        sum_genes = sum(individual)  # sum of genes (x + y)
        return abs(self.target_function(x_val, y_val) - sum_genes)

    def selection(self, x_val, y_val):
        # Rank individuals based on fitness and select the best half
        fitness_values = [self.fitness(individual, x_val, y_val) for individual in self.population]
        selected_indices = np.argsort(fitness_values)[:self.population_size // 2]
        selected_population = [self.population[i] for i in selected_indices]
        return selected_population

    def crossover(self, parent1, parent2):
        # Perform single-point crossover between two parents (swap genes)
        crossover_point = random.randint(0, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutation(self, individual):
        # Mutate by changing one gene value randomly
        if random.random() < self.mutation_rate:
            mutation_point = random.randint(0, len(individual) - 1)
            individual[mutation_point] = random.uniform(0, 10)  # random value in [0, 10]
        return individual

    def evolve(self, x_val, y_val):
        for generation in range(self.generations):
            selected_population = self.selection(x_val, y_val)
            new_population = []

            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutation(child1))
                new_population.append(self.mutation(child2))

            self.population = new_population[:self.population_size]
            best_individual = min(self.population, key=lambda ind: self.fitness(ind, x_val, y_val))
            print(f"Generation {generation + 1}: Best Individual = {best_individual}, Fitness = {self.fitness(best_individual, x_val, y_val)}")

target_function = lambda x, y: x + y
gep = GeneExpressionProgramming(population_size=10, generations=20, target_function=target_function)
gep.evolve(x_val=3, y_val=2)
