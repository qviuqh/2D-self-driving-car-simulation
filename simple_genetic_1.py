import numpy as np
import random
import pygame
import time
import json
from car_environment import CarEnvironment

class SimpleNeuralNetwork:
    def __init__(self, input_size=6, hidden_size=8, output_size=2):
        """
        Simple neural network for genetic algorithm
        input_size: 5 (speed, angle, 3 sensors)
        output_size: 2 (steering, throttle)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.random.randn(hidden_size) * 0.5
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.random.randn(output_size) * 0.5
    
    def forward(self, x):
        """Forward pass through network"""
        # Hidden layer
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        # Output layer
        output = np.tanh(np.dot(h, self.w2) + self.b2)
        return output
    
    def get_weights(self):
        """Get all weights as a flat array"""
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten()
        ])
    
    def set_weights(self, weights):
        """Set weights from a flat array"""
        idx = 0
        
        # w1
        w1_size = self.input_size * self.hidden_size
        self.w1 = weights[idx:idx+w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        # b1
        self.b1 = weights[idx:idx+self.hidden_size]
        idx += self.hidden_size
        
        # w2
        w2_size = self.hidden_size * self.output_size
        self.w2 = weights[idx:idx+w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        # b2
        self.b2 = weights[idx:idx+self.output_size]
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        """Mutate weights"""
        weights = self.get_weights()
        
        # Random mutations
        for i in range(len(weights)):
            if random.random() < mutation_rate:
                weights[i] += random.gauss(0, mutation_strength)
        
        self.set_weights(weights)
    
    def crossover(self, other):
        """Create offspring through crossover"""
        weights1 = self.get_weights()
        weights2 = other.get_weights()
        
        # Single point crossover
        crossover_point = random.randint(1, len(weights1) - 1)
        
        child1_weights = np.concatenate([
            weights1[:crossover_point],
            weights2[crossover_point:]
        ])
        
        child2_weights = np.concatenate([
            weights2[:crossover_point],
            weights1[crossover_point:]
        ])
        
        child1 = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        child2 = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        
        child1.set_weights(child1_weights)
        child2.set_weights(child2_weights)
        
        return child1, child2

class GeneticCarTrainer:
    def __init__(self, population_size=20, render=True):
        self.population_size = population_size
        self.render = render
        self.env = CarEnvironment()
        
        # Create initial population
        self.population = [
            SimpleNeuralNetwork() for _ in range(population_size)
        ]
        
        self.generation = 0
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        
    def evaluate_individual(self, neural_network, max_steps=5000, render=False):
        """Evaluate a single neural network"""
        self.env.reset_car()
        state = self.env.get_state()
        total_fitness = 0
        steps = 0
        
        start_time = time.time()
        
        for step in range(max_steps):
            if render:
                self.env.handle_events()
                if not self.env.running:
                    break
            
            # Neural network decides action
            action = neural_network.forward(state)
            steering, throttle = action[0], action[1]
            
            # Take action in environment
            next_state, reward, done, info = self.env.step([steering, throttle])
            
            state = next_state
            total_fitness += reward
            steps += 1
            
            # Render if requested
            if render:
                self.env.render()
                self.env.clock.tick(60)  # 60 FPS
            
            if done:
                break
        
        # Bonus for surviving longer
        survival_bonus = steps * 0.1
        total_fitness += survival_bonus
        
        # Bonus for completing lap
        if self.env.completed_lap:
            total_fitness += 500 * self.env.laps  # Scale bonus by number of laps
        
        return total_fitness, steps, self.env.completed_lap
    
    def evaluate_population(self):
        """Evaluate entire population"""
        fitness_scores = []
        
        print(f"\n🧬 Evaluating Generation {self.generation}")
        print("-" * 50)
        
        for i, individual in enumerate(self.population):
            fitness, steps, completed_lap = self.evaluate_individual(
                individual, 
                render=(self.render and i == 0)  # Only render best individual
            )
            fitness_scores.append(fitness)
            
            status = "🏁" if completed_lap else "❌"
            print(f"Individual {i+1:2d}: Fitness = {fitness:7.1f}, Steps = {steps:3d} {status}")
            
            # Update best fitness
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                print(f"🎯 New best fitness: {fitness:.1f}")
        
        return fitness_scores
    
    def selection(self, fitness_scores):
        """Select parents using tournament selection"""
        parents = []
        tournament_size = 3
        
        for _ in range(self.population_size // 2):
            # Tournament selection
            tournament_indices = random.sample(range(self.population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_index])
        
        return parents
    
    def create_next_generation(self, fitness_scores):
        """Create next generation through selection, crossover, and mutation"""
        # Keep best individual (elitism)
        best_index = np.argmax(fitness_scores)
        elite = self.population[best_index]
        
        # Select parents
        parents = self.selection(fitness_scores)
        
        # Create new generation
        new_population = [elite]  # Keep elite
        
        # Create offspring through crossover
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = parent1.crossover(parent2)
            
            # Mutate children
            child1.mutate(mutation_rate=0.1, mutation_strength=0.2)
            child2.mutate(mutation_rate=0.1, mutation_strength=0.2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.population_size]
    
    def save_best_individual(self, filename):
        """Save the best individual"""
        fitness_scores = [
            self.evaluate_individual(individual, max_steps=2500)[0] 
            for individual in self.population
        ]
        
        best_index = np.argmax(fitness_scores)
        best_individual = self.population[best_index]
        
        # Save weights
        weights = best_individual.get_weights().tolist()
        
        data = {
            'weights': weights,
            'fitness': fitness_scores[best_index],
            'generation': self.generation,
            'architecture': {
                'input_size': best_individual.input_size,
                'hidden_size': best_individual.hidden_size,
                'output_size': best_individual.output_size
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Best individual saved to {filename}")
    
    def load_individual(self, filename):
        """Load a saved individual"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Create network with saved architecture
        arch = data['architecture']
        network = SimpleNeuralNetwork(
            arch['input_size'], 
            arch['hidden_size'], 
            arch['output_size']
        )
        
        # Load weights
        network.set_weights(np.array(data['weights']))
        
        print(f"📂 Individual loaded from {filename}")
        print(f"   Fitness: {data['fitness']:.1f}, Generation: {data['generation']}")
        
        return network
    
    def train(self, generations=50):
        """Main training loop"""
        print("🚗 Starting Genetic Algorithm Training")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {generations}")
        print("-" * 60)
        
        for generation in range(generations):
            self.generation = generation
            
            # Evaluate population
            fitness_scores = self.evaluate_population()
            
            # Statistics
            avg_fitness = np.mean(fitness_scores)
            max_fitness = np.max(fitness_scores)
            min_fitness = np.min(fitness_scores)
            
            self.fitness_history.append({
                'generation': generation,
                'avg_fitness': avg_fitness,
                'max_fitness': max_fitness,
                'min_fitness': min_fitness
            })
            
            print(f"\nGeneration {generation} Summary:")
            print(f"  Average Fitness: {avg_fitness:7.1f}")
            print(f"  Best Fitness:    {max_fitness:7.1f}")
            print(f"  Worst Fitness:   {min_fitness:7.1f}")
            
            # Save best individual every 10 generations
            if generation % 10 == 0:
                self.save_best_individual(f'genetic_best_gen_{generation}.json')
            
            # Create next generation
            if generation < generations - 1:
                self.create_next_generation(fitness_scores)
        
        # Final save
        self.save_best_individual('genetic_final_best.json')
        print(f"\n🏆 Training completed! Best fitness: {self.best_fitness:.1f}")
    
    def test_individual(self, network, episodes=5):
        """Test a trained individual"""
        print("🧪 Testing individual...")
        
        for episode in range(episodes):
            print(f"\nTest Episode {episode + 1}/{episodes}")
            fitness, steps, completed_lap = self.evaluate_individual(
                network, max_steps=2000, render=True
            )
            
            status = "🏁 Completed!" if completed_lap else "❌ Failed"
            print(f"Result: {status} | Fitness: {fitness:.1f} | Steps: {steps}")

def main():
    print("🧬 Genetic Algorithm Car Training")
    print("1. Train new population")
    print("2. Test saved individual")
    print("3. Continue training from checkpoint")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice == '1':
        # Train new population
        population_size = int(input("Population size (default 20): ") or "20")
        generations = int(input("Number of generations (default 50): ") or "50")
        render = input("Enable rendering for best individual? (y/n): ").lower() == 'y'
        
        trainer = GeneticCarTrainer(population_size=population_size, render=render)
        trainer.train(generations=generations)
        
    elif choice == '2':
        # Test saved individual
        filename = input("Enter saved file (e.g., genetic_final_best.json): ")
        try:
            trainer = GeneticCarTrainer(render=True)
            network = trainer.load_individual(filename)
            trainer.test_individual(network, episodes=5)
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error loading file: {e}")
    
    elif choice == '3':
        # Continue training (simplified - just load best and create new population around it)
        filename = input("Enter checkpoint file: ")
        try:
            trainer = GeneticCarTrainer(render=True)
            best_network = trainer.load_individual(filename)
            
            # Create new population based on best individual
            trainer.population = [best_network]  # Keep the best
            for _ in range(trainer.population_size - 1):
                new_individual = SimpleNeuralNetwork()
                new_individual.set_weights(best_network.get_weights())
                new_individual.mutate(mutation_rate=0.3, mutation_strength=0.3)
                trainer.population.append(new_individual)
            
            generations = int(input("Additional generations (default 25): ") or "25")
            trainer.train(generations=generations)
            
        except FileNotFoundError:
            print(f"Checkpoint not found: {filename}")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()