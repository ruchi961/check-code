import time
import re
import logging

# Initialize logger
log = logging.getLogger(__name__)


class rto:
    def __init__(self,llm) -> None:
        self.llm = llm
        
    def generate(self,request_list,return_model_answer=True):
        if return_model_answer:
            model_response_returned = self.llm.generate(request_list)
        
        
        rto_response_list = self.round_trip_optimization(request_list)
        model_response_list=[]
        for response_out in model_response_returned:
            model_response_list.append(response_out[1]['choices'][0]['message']['content'])
        response=[]
        for model_response, rto_response in zip(model_response_list,rto_response_list):
            response.append({'model_response':model_response,"rto_response":rto_response})
        response=[{"model_response":"""Here\'s a simple implementation of a genetic algorithm in Python that uses NumPy for efficient computation. This implementation assumes a binary representation for the individuals, and the fitness function used is a simple function that sums the individual\'s binary digits.\n\n```python\nimport numpy as np\nimport random\nimport operator\n\n# Define constants\nPOPULATION_SIZE = 100\nMUTATION_RATE = 0.01\nCROSSOVER_RATE = 0.7\nGENERATIONS = 1000\nINDIVIDUAL_LENGTH = 10\n\n# Define fitness function\ndef fitness(individual):\n    return -sum(individual)\n\n# Initialize population\ndef initialize_population(population_size, individual_length):\n    return [np.random.randint(2, size=individual_length) for _ in range(population_size)]\n\n# Selection function (Tournament selection)\ndef selection(population, num_parents):\n    parents = []\n    for _ in range(num_parents):\n        parents.append(max(random.sample(population, len(population)), key=fitness))\n    return parents\n\n# Crossover function (Single-point crossover)\ndef crossover(parent1, parent2):\n    crossover_point = random.randint(1, len(parent1) - 1)\n    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))\n    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))\n    return offspring1, offspring2\n\n# Mutation function (Bit-flip mutation)\ndef mutation(individual):\n    for i in range(len(individual)):\n        if random.random() < MUTATION_RATE:\n            individual[i] = 1 - individual[i]\n    return individual\n\n# Main loop\ndef genetic_algorithm():\n    population = initialize_population(POPULATION_SIZE, INDIVIDUAL_LENGTH)\n    for generation in range(GENERATIONS):\n        parents = selection(population, POPULATION_SIZE)\n        offspring = []\n        while len(offspring) < POPULATION_SIZE:\n            if random.random() < CROSSOVER_RATE:\n                parent1, parent2 = random.sample(parents, 2)\n                offspring1, offspring2 = crossover(parent1, parent2)\n                offspring.append(offspring1)\n                offspring.append(offspring2)\n            else:\n                offspring.append(random.choice(parents))\n        for i in range(POPULATION_SIZE):\n            offspring[i] = mutation(offspring[i])\n        population = offspring\n\n    return population[0]\n\n# Run the algorithm\nresult = genetic_algorithm()\nprint("Final individual:", result)\nprint("Fitness:", fitness(result))\n```\n\nThis code will output the best individual found by the genetic algorithm and its fitness value. Please note that this is a basic implementation and may not be optimal for every problem. The choice of parameters like population size, mutation rate, crossover rate, and number of generations can affect the performance of the algorithm.\n\nTo make this code even faster, consider the following optimizations:\n\n1.  Use a more efficient data structure, such as NumPy arrays, for storing and manipulating the individuals.\n2.  Implement a more efficient selection method, such as rank selection or truncation selection, which can reduce the computational overhead of the selection process.\n3.  Use a more efficient crossover method, such as uniform crossover or arithmetic crossover, which can be faster than single-point crossover.\n4.  Consider using a multi-threaded or multi-process approach to parallelize the computation, especially for large population sizes or complex fitness functions.\n5.  Use a more efficient optimization method, such as simulated annealing or ant colony optimization, which can be more efficient than genetic algorithms for certain problems.\n\nHere are some possible improvements for each of these suggestions:\n\n1.  Data structure:\n\n    *   Use NumPy arrays instead of Python lists to store and manipulate individuals.\n    *   Consider using a data structure like a binary tree or a heap to store individuals and improve selection efficiency.\n2.  Selection method:\n\n    *   Implement rank selection or truncation selection, which can be faster than tournament selection.\n    *   Use a more efficient data structure, such as a heap or a balanced binary search tree, to store the individuals and improve selection efficiency.\n3.  Crossover method:\n\n    *   Implement uniform crossover or arithmetic crossover, which can be faster than single-point crossover.\n    *   Use a more efficient algorithm for crossover, such as a random index generator or a hash function.\n4.  Parallelization:\n\n    *   Use the `multiprocessing` module to parallelize the computation using multiple processes.\n    *   Use the `concurrent.futures` module to parallelize the computation using multiple threads.\n5.  Optimization method:\n\n    *   Implement simulated annealing or ant colony optimization, which can be more efficient than genetic algorithms for certain problems.\n    *   Use a more efficient optimization algorithm, such as gradient descent or quasi-Newton methods, which can be faster than genetic algorithms for certain problems.""","rto_response":"""```python\nimport numpy as np\nimport random\n\nclass GeneticAlgorithm:\n    def __init__(self, population_size, num_generations, mutation_rate, bounds, fitness_function):\n        self.population_size = population_size\n        self.num_generations = num_generations\n        self.mutation_rate = mutation_rate\n        self.bounds = bounds\n        self.fitness_function = fitness_function\n        self.population = self.initialize_population()\n\n    def initialize_population(self):\n        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, 2))\n\n    def selection(self):\n        fitness_values = np.array([self.fitness_function(individual) for individual in self.population])\n        indices = np.argsort(fitness_values)[-int(0.2 * self.population_size):]\n        return self.population[indices]\n\n    def crossover(self, parent1, parent2):\n        child = (parent1 + parent2) / 2\n        return child\n\n    def mutation(self, individual):\n        if random.random() < self.mutation_rate:\n            individual += np.random.uniform(-1, 1, 2)\n            individual = np.clip(individual, self.bounds[0], self.bounds[1])\n        return individual\n\n    def evolve(self):\n        selected = self.selection()\n        offspring = []\n        while len(offspring) < self.population_size:\n            parent1, parent2 = random.sample(selected, 2)\n            child = self.crossover(parent1, parent2)\n            child = self.mutation(child)\n            offspring.append(child)\n        self.population = np.array(offspring)\n\n    def run(self):\n        for _ in range(self.num_generations):\n            self.evolve()\n        return self.population\n\ndef fitness(individual):\n    x, y = individual\n    return x + 2 * y\n\nbounds = (-10, 10)\npopulation_size = 100\nnum_generations = 100\nmutation_rate = 0.1\n\nga = GeneticAlgorithm(population_size, num_generations, mutation_rate, bounds, fitness)\nbest_individual = ga.run()[-1]\nprint(f"Best individual: {best_individual}")\nprint(f"Fitness: {fitness(best_individual)}")\n```"""}]
        return response


    def extract_code(self,text_content: str):
        # Define regex to extract code given by model between triple backticks
        code_block_pattern = r"```(.*?)```"
        
        # Attempt to find code block
        match = re.search(code_block_pattern, text_content, re.DOTALL)
        
        # If code block found, return it after stripping whitespace
        if match:
            return match.group(1).strip()
        else:
            log.warning("Failed to extract the code block. Returning the original text.")
            return text_content
    def gather_requests(self,request_list: list):
        request_list_return =[]
        for request in request_list:
            initial_query=""
            system_prompt = ""
            conversation = []
    
            for message in request['messages']:
                role = message['role']
                content = message['content']
                
                if role == 'system':
                    system_prompt = content
                elif role in ['user', 'assistant']:
                    conversation.append(f"{role.capitalize()}: {content}")
            
            initial_query = "\n".join(conversation)
            request_list_return.append([system_prompt,initial_query])
        return request_list_return

    def round_trip_optimization(self, request_list: list) -> list:
        request_list_modified = self.gather_requests(request_list)
        response_list=[]

        request_c1_list =[]
        for request_m,request in zip(request_list_modified,request_list):
            messages=[]
            system_prompt = request_m[0]
            initial_query = request_m[1]
           
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": initial_query}]
            request['messages'] = messages
            request_c1_list.append(request)
        # print("request_c1_list",request_c1_list)
        # Generate initial code (C1)
        response_c1_list = self.llm.generate(request_c1_list)
        # print("response_c1_list",response_c1_list)
        c1_list=[]
        for response_c1_index in range(len(response_c1_list)):
            
            c1 = response_c1_list[response_c1_index][1]['choices'][0]['message']['content']
            c1_list .append(c1)    
            # Generate description of the code (Q2)
            
            request_c1_list[response_c1_index]['messages'].append({"role": "assistant", "content": c1})
            request_c1_list[response_c1_index]['messages'].append({"role": "user", "content": "Summarize or describe the code given to you. \
                             Ensure, that the summary should be in such form of instruction that, given the same instruction you can create the code by yourself."})
        # print("request_c1_list",request_c1_list)
        response_q2_list =self.llm.generate(request_c1_list)
        # print("response_q2_list",response_q2_list)
        request_c2_list=[]
        for response_q2_index in range(len(response_q2_list)):
            q2 = response_q2_list[response_q2_index][1]['choices'][0]['message']['content']
            messages=[]
            # Generate second code based on the description (C2)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": q2}]
            request = request_list[response_q2_index]
            request['messages'] = messages
            request_c2_list.append(request)
        # print("request_c2_list",request_c2_list)
        response_c2_list = self.llm.generate(request_c2_list)
        c2_list=[]
        for response_c2 in response_c2_list:
            c2_list.append(response_c2[1]['choices'][0]['message']['content'])


        request_c3_list=[]
        for c1,c2,request_m,request in zip(c1_list,c2_list,request_list_modified,request_list) :
            c1 = self.extract_code(c1)
            c2 = self.extract_code(c2)
            system_prompt = request_m[0]
            initial_query = request_m[1]
            

            if c1.strip() == c2.strip():
                return c1
            messages=[]
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Initial query: {initial_query}\n\nFirst generated code (C1):\n{c1}\n\nSecond generated code (C2):\n{c2}\n\nBased on the initial query and these two different code implementations, generate a final, optimized version of the code. Only respond with the final code, do not return anything else."}]
            request['messages'] = messages
            request_c3_list.append(request)
        # print("request_c3_list",request_c3_list)
        response_c3_list =self.llm.generate(request_c3_list)
        # print("response_c3_list",response_c3_list)
        for response_c3 in response_c3_list:

            c3 = response_c3[1]['choices'][0]['message']['content']
            response_list.append(response_c3[1]['choices'][0]['message']['content'])
        # print(response_list)
       
        return response_list
    

