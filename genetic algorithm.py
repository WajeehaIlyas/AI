import random 

def generate_chromosome():
    return [random.randint(0,7) for _ in range(8)]

#fitness 
def fitness(chromosome):
    horizontal_collisions = sum([chromosome.count(queen)-1 for queen in chromosome])/2
    diagonal_collisions = 0

    for i in range(8):
        for j in range(8):
            if i != j:
                dx = abs(i-j)
                dy = abs(chromosome[i] - chromosome[j])
                if dx == dy:
                    diagonal_collisions += 1

    return 28 - (horizontal_collisions + diagonal_collisions)

def selection(population):
    return random.choices(population, weights=[fitness(chromosome) for chromosome in population], k=2)

def crossover(parent1, parent2):
    crossover_point = random.randint(1, 7)
    return parent1[:crossover_point] + parent2[crossover_point:]

def mutation(chromosome):
    mutated_chromosome = chromosome[:]
    index = random.randint(0, 7)
    new_value = random.randint(0, 7)

    if new_value != mutated_chromosome[index]: 
        mutated_chromosome[index] = new_value

    return mutated_chromosome if fitness(mutated_chromosome) >= fitness(chromosome) else chromosome

def genetic_algorithm():
    population = [generate_chromosome() for _ in range(10)]
    best_fitness = 0
    generation= 0

    for gen in range(1000):
        generation = gen + 1
        new_population = sorted(population, key=fitness, reverse=True)[:5]
        for _ in range(5):
            parent1, parent2 = selection(population)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:
                child = mutation(child)
            new_population.append(child)
        population = new_population

        current_best_fitness = fitness(max(population, key=fitness))
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness

        if best_fitness == 28:  
            break

    best_solution = max(population, key=fitness)

    return best_solution, generation

#result as board
def display(chromosome):
    for row in range(8):
        line = ""
        for column in range(8):
            if chromosome[column] == row:
                line += "Q "
            else:
                line += ". "
        print(line)
    print("\n")

def main():
    solution, generations = genetic_algorithm()
    print("Solution found in {} generations".format(generations))
    display(solution)
if __name__ == '__main__':
    main()

