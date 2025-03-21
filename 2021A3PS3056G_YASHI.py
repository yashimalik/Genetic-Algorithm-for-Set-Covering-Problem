def check_coverage(lists_of_lists):
    # Create a set of all numbers from 1 to 100
    required_numbers = set(range(1, 101))

    # Combine all the lists into a single set
    combined_numbers = set()
    for sublist in lists_of_lists:
        combined_numbers.update(sublist)

    # Check if all required numbers are covered
    return required_numbers.issubset(combined_numbers)

import sys
import random
import json

class SetCoveringProblemCreator:
    def __init__(self):
        pass

    def _SampleWithoutReplacement(self, k, usize):
        return random.sample(range(1, usize + 1), k)

    def _FindMissingIntegers(self, input_set, max_num):
        all_integers_set = set(range(1, max_num + 1))
        missing_integers_set = all_integers_set - input_set
        missingIntegersList = list(missing_integers_set)
        return missingIntegersList

    def _CreateOneSet(self, usize, setOfSets, elementsCovered):
        k = random.randint(1, 10) #set size
        newSet = frozenset(self._SampleWithoutReplacement(k, usize))
        setOfSets.add(newSet)
        return elementsCovered.union(newSet)

    def Create(self, usize, totalSets):
        """
        The Create function generates subsets for the elements in the universe.
        usize is the total number of elements in the universe.
        totalSets is the total number of subsets that are part of the Set Covering Problem.
        The Create function returns a list of subsets as a list of lists.
        """
        if usize != 100:
            exit('Universe size (usize) must be 100.')
        setOfSets = set()
        elementsCovered = set()
        while len(setOfSets) < totalSets - 1:
            elementsCovered = self._CreateOneSet(usize, setOfSets, elementsCovered)
        missingIntegers = self._FindMissingIntegers(elementsCovered, usize)
        if len(missingIntegers) == 0:
            while len(setOfSets) < totalSets:
                elementsCovered = self._CreateOneSet(usize, setOfSets, elementsCovered)
        else:
            newSet = frozenset(missingIntegers)
            setOfSets.add(newSet)
            elementsCovered = elementsCovered.union(newSet)
        listOfSets = list(setOfSets)
        return listOfSets

    def WriteSetsToJson(self, listOfSets, usize, totalSets):
        # Convert frozensets to lists
        list_of_lists = [list(fs) for fs in listOfSets]

        # Write the list of lists to a JSON file
        fileName = f"scp_{totalSets}.json"
        with open(fileName, 'w') as json_file:
            json.dump(list_of_lists, json_file)

        #print(f"A random instance of Set Covering Problem is created in {fileName} file:")
        #print(f"universe-size = {usize}, number-of-subsets = {totalSets}.")

    def ReadSetsFromJson(self, fileName):
        """
        ReadSetsFromJson reads a list of lists from a json file.
        The list read will contain all the subsets in the Set Covering Problem.
        """
        try:
            with open(fileName, 'r') as json_file:
                listOfSubsets = json.load(json_file)

#            # Convert lists back to frozensets
#            listOfSets = [frozenset(lst) for lst in list_of_lists]
            return listOfSubsets
        except FileNotFoundError:
            print(f"Error: The file {fileName} was not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: The file {fileName} is not a valid JSON file.")
            return None


import random
import time
import matplotlib.pyplot as plt

def fitness(solution, subsets, universe):
    covered = set()
    for i in range(len(solution)):
        if solution[i] == 1:
            covered.update(subsets[i])
    uncovered_elements = universe - covered
    if not uncovered_elements:
        return sum(solution)  # minimize number of subsets used
    else:
        return float('inf')  # invalid solution

def initial_population(population_size, num_subsets):
    population = []
    for _ in range(population_size):
        individual = [random.randint(0, 1) for _ in range(num_subsets)]
        population.append(individual)
    return population

def selection(population, fitness_func):
    total_fitness = sum(fitness_func(ind) for ind in population)
    if total_fitness == 0:
        return random.choice(population)
    selection_probs = [fitness_func(ind) / total_fitness for ind in population]
    return random.choices(population, weights=selection_probs, k=1)[0]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # flip between 0 and 1

def ensure_coverage(individual, subsets, universe):
    """Ensure the individual covers the entire universe."""
    covered = set()
    for i in range(len(individual)):
        if individual[i] == 1:
            covered.update(subsets[i])

    missing_elements = universe - covered
    if missing_elements:
        for i in range(len(subsets)):
            if missing_elements.intersection(subsets[i]):
                individual[i] = 1
                covered.update(subsets[i])
            if not missing_elements.difference(covered):
                break

def genetic_algorithm(subsets, universe, max_time=45, population_size=100, generations=1000, mutation_rate=0.01, elitism_rate=0.01, culling_rate=0.95, no_improvement_limit=100):
    start_time = time.time()
    num_subsets = len(subsets)
    population = initial_population(population_size, num_subsets)
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []
    no_improvement_count = 0

    def fitness_func(sol): return fitness(sol, subsets, universe)

    for generation in range(generations):
        # Ensure all individuals cover the universe
        for individual in population:
            ensure_coverage(individual, subsets, universe)

        # Sort population by fitness (lower is better)
        population.sort(key=fitness_func)

        # Elitism: Preserve the top individuals
        num_elites = int(elitism_rate * population_size)
        elites = population[:num_elites]

        # Culling: Remove the worst individuals
        num_to_cull = int(culling_rate * population_size)
        population = population[:-num_to_cull]

        current_best = population[0]
        current_fitness = fitness_func(current_best)

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = current_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if time.time() - start_time > max_time or no_improvement_count >= no_improvement_limit:
            break

        new_population = elites[:]  # Start the new population with elites
        while len(new_population) < population_size:
            parent1 = selection(population, fitness_func)
            parent2 = selection(population, fitness_func)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population[:population_size]  # Trim to the desired population size
        fitness_history.append(num_subsets-best_fitness)

    time_taken = time.time() - start_time

    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, label='Best Fitness', marker='o', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Genetic Algorithm Progress')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if best_fitness == float('inf'):
        print("No valid solution found.")
    return best_solution, best_fitness, time_taken

def print_best(best_solution):
    print("Solution :")
    print(f"{0}:",best_solution[0],end="",sep="")
    for i in range(1, len(best_solution)):
      print(f", {i}:",best_solution[i],end="",sep="")
    print()

def main():
    print("Roll no: 2021A3PS3056G")
    scp = SetCoveringProblemCreator()

    # Generate a random Set Covering Problem instance with 50 subsets
    subsets = scp.Create(usize=100, totalSets=150)
    scp.WriteSetsToJson(subsets, usize=100, totalSets=150)

    # Read the subsets from the scp_test.json file
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
    print("Number of subsets in scp_test.json file:", len(listOfSubsets))

    if listOfSubsets is not None:
        subsets = [set(subset) for subset in listOfSubsets]
        universe = set(range(1, 101))

        # Solve the Set Covering Problem using Genetic Algorithm
        best_solution, best_fitness, time_taken = genetic_algorithm(subsets, universe)

        if best_solution is not None:
            #print("Best Solution:", best_solution)
            print_best(best_solution)
            print("Fitness value of Best state:", len(subsets)-best_fitness)
            print("Minimum number of subsets that can cover the entire Universe-set:", best_fitness)
        else:
            print("No valid solution was found.")
        print("Time Taken (seconds):", time_taken)

if __name__ == "__main__":
    main()