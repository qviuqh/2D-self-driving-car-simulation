import numpy as np
import random
import pygame
import time
import json
from car_environment import CarEnvironment

class SimpleNeuralNetwork:
    def __init__(self, input_size=6, hidden_size=8, output_size=4):
        """
        Simple neural network for genetic algorithm
        input_size: 5 (speed, angle, 3 sensors)
        output_size: 4 (up, down, left, right)
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
        # Output layer - using sigmoid for 4 directional controls
        output = 1 / (1 + np.exp(-(np.dot(h, self.w2) + self.b2)))  # Sigmoid activation
        return output
    
    def get_action(self, state):
        """Convert neural network output to steering and throttle"""
        output = self.forward(state)
        
        # 4 outputs: [up, down, left, right]
        up, down, left, right = output
        
        # Convert to steering (-1 to 1) and throttle (-1 to 1)
        steering = (right - left)  # Right positive, left negative
        throttle = (up - down)     # Up positive, down negative
        
        # Clamp values to [-1, 1]
        steering = np.clip(steering, -1, 1)
        throttle = np.clip(throttle, -1, 1)
        
        return [steering, throttle]
    
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
    def __init__(self, population_size=20, render=True, max_laps=5):
        self.population_size = population_size
        self.render = render
        self.max_laps = max_laps
        self.env = CarEnvironment()
        
        # Create initial population
        self.population = [
            SimpleNeuralNetwork() for _ in range(population_size)
        ]
        
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_laps_completed = 0
        self.fitness_history = []
        
        # Track best multi-lap performer
        self.best_multilap_individual = None
        self.best_multilap_fitness = -float('inf')
        
    def evaluate_individual(self, neural_network, max_steps=10000, render=False):
        """Evaluate a single neural network with multi-lap tracking"""
        self.env.reset_car()
        state = self.env.get_state()
        total_fitness = 0
        steps = 0
        
        # Multi-lap tracking
        laps_completed = 0
        prev_completed_lap = False
        lap_completion_times = []
        lap_start_step = 0
        
        start_time = time.time()
        
        for step in range(max_steps):
            if render:
                self.env.handle_events()
                if not self.env.running:
                    break
            
            # Neural network decides action
            action = neural_network.get_action(state)
            steering, throttle = action[0], action[1]
            
            # Take action in environment
            next_state, reward, done, info = self.env.step([steering, throttle])
            
            state = next_state
            total_fitness += reward
            steps += 1
            
            # Check for lap completion
            current_completed_lap = self.env.completed_lap
            if current_completed_lap and not prev_completed_lap:
                laps_completed += 1
                lap_time = steps - lap_start_step
                lap_completion_times.append(lap_time)
                lap_start_step = steps
                
                # Massive bonus for each completed lap
                # lap_bonus = 1000 * (laps_completed ** 1.5)  # Exponential bonus
                lap_bonus = 200 * laps_completed
                total_fitness += lap_bonus
                
                print(f"    🏁 Lap {laps_completed} completed in {lap_time} steps! Bonus: +{lap_bonus:.0f}")
                
                # Reset the lap flag in environment (if your environment supports it)
                if hasattr(self.env, 'reset_lap_flag'):
                    self.env.reset_lap_flag()
                
                # Stop if reached max laps
                if laps_completed >= self.max_laps:
                    print(f"    🏆 Maximum {self.max_laps} laps completed!")
                    break
            
            prev_completed_lap = current_completed_lap
            
            # Render if requested
            if render:
                self.env.render()
                
                # Display lap info on screen
                if hasattr(self.env, 'screen'):
                    font = pygame.font.Font(None, 36)
                    lap_text = font.render(f"Laps: {laps_completed}/{self.max_laps}", True, (255, 255, 255))
                    self.env.screen.blit(lap_text, (10, 10))
                    steps_text = font.render(f"Steps: {steps}", True, (255, 255, 255))
                    self.env.screen.blit(steps_text, (10, 50))
                    fitness_text = font.render(f"Fitness: {total_fitness:.0f}", True, (255, 255, 255))
                    self.env.screen.blit(fitness_text, (10, 90))
                
                self.env.clock.tick(60)  # 60 FPS
            
            if done:
                break
        
        # Additional bonuses
        survival_bonus = steps * 0.05  # Reduced survival bonus
        total_fitness += survival_bonus
        
        # Consistency bonus for multiple laps
        if laps_completed > 1:
            avg_lap_time = np.mean(lap_completion_times)
            consistency_bonus = 200 * laps_completed * (1 / (1 + np.std(lap_completion_times) / avg_lap_time))
            total_fitness += consistency_bonus
        
        # Penalty for getting stuck (very low fitness with many steps)
        if steps > 3000 and total_fitness < 100:
            total_fitness -= 500
        
        return total_fitness, steps, laps_completed, lap_completion_times
    
    def evaluate_population(self):
        """Evaluate entire population"""
        fitness_scores = []
        lap_counts = []
        
        print(f"\n🧬 Evaluating Generation {self.generation}")
        print("-" * 70)
        
        for i, individual in enumerate(self.population):
            fitness, steps, laps, lap_times = self.evaluate_individual(
                individual, 
                max_steps=8000,  # Increased for multi-lap attempts
                render=(self.render and i == 0)  # Only render best individual
            )
            fitness_scores.append(fitness)
            lap_counts.append(laps)
            
            # Status indicators
            if laps >= self.max_laps:
                status = "🏆"
            elif laps >= 2:
                status = "🥈"
            elif laps >= 1:
                status = "🥉"
            else:
                status = "❌"
            
            print(f"Individual {i+1:2d}: Fitness = {fitness:8.1f}, Steps = {steps:4d}, Laps = {laps} {status}")
            
            # Track best multi-lap performer
            if laps > self.best_laps_completed or (laps == self.best_laps_completed and fitness > self.best_multilap_fitness):
                self.best_laps_completed = laps
                self.best_multilap_fitness = fitness
                self.best_multilap_individual = individual
                print(f"🎯 New best multi-lap performer: {laps} laps, fitness {fitness:.1f}")
            
            # Update overall best fitness
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                print(f"⭐ New best fitness: {fitness:.1f}")
        
        return fitness_scores, lap_counts
    
    def selection(self, fitness_scores, lap_counts):
        """Enhanced selection considering both fitness and lap completion"""
        parents = []
        tournament_size = 3
        
        for _ in range(self.population_size // 2):
            # Tournament selection with lap bias
            tournament_indices = random.sample(range(self.population_size), tournament_size)
            
            # Create weighted scores (laps are heavily weighted)
            weighted_scores = []
            for idx in tournament_indices:
                lap_weight = lap_counts[idx] * 100  # Heavy weight for laps
                fitness_weight = fitness_scores[idx]
                weighted_scores.append(lap_weight + fitness_weight)
            
            winner_index = tournament_indices[np.argmax(weighted_scores)]
            parents.append(self.population[winner_index])
        
        return parents
    
    def create_next_generation(self, fitness_scores, lap_counts):
        """Create next generation with emphasis on multi-lap performers"""
        # Find individuals with most laps completed
        max_laps = max(lap_counts)
        multi_lap_indices = [i for i, laps in enumerate(lap_counts) if laps >= max(1, max_laps)]
        
        # Elitism - keep best multi-lap performers
        elite_count = min(3, len(multi_lap_indices))
        elite_indices = sorted(multi_lap_indices, 
                             key=lambda i: (lap_counts[i], fitness_scores[i]), 
                             reverse=True)[:elite_count]
        
        new_population = [self.population[i] for i in elite_indices]
        
        # Select parents with lap consideration
        parents = self.selection(fitness_scores, lap_counts)
        
        # Create offspring through crossover
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = parent1.crossover(parent2)
            
            # Adaptive mutation - less mutation for good performers
            max_parent_laps = max([lap_counts[self.population.index(p)] for p in [parent1, parent2] if p in self.population])
            if max_parent_laps >= 2:
                mutation_rate, mutation_strength = 0.05, 0.1  # Conservative mutation
            else:
                mutation_rate, mutation_strength = 0.15, 0.25  # More aggressive mutation
            
            child1.mutate(mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            child2.mutate(mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.population_size]
    
    def save_best_individual(self, filename, save_multilap=True):
        """Save the best individual with lap information"""
        if save_multilap and self.best_multilap_individual is not None:
            best_individual = self.best_multilap_individual
            best_fitness = self.best_multilap_fitness
            best_laps = self.best_laps_completed
        else:
            fitness_scores = []
            lap_counts = []
            for individual in self.population:
                fitness, steps, laps, lap_times = self.evaluate_individual(individual, max_steps=5000)
                fitness_scores.append(fitness)
                lap_counts.append(laps)
            
            best_index = np.argmax(fitness_scores)
            best_individual = self.population[best_index]
            best_fitness = fitness_scores[best_index]
            best_laps = lap_counts[best_index]
        
        # Save weights
        weights = best_individual.get_weights().tolist()
        
        data = {
            'weights': weights,
            'fitness': best_fitness,
            'laps_completed': best_laps,
            'generation': self.generation,
            'architecture': {
                'input_size': best_individual.input_size,
                'hidden_size': best_individual.hidden_size,
                'output_size': best_individual.output_size
            },
            'control_type': '4_directional'
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Best individual saved to {filename}")
        print(f"   Fitness: {best_fitness:.1f}, Laps: {best_laps}")
    
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
        if 'laps_completed' in data:
            print(f"   Laps completed: {data['laps_completed']}")
        
        return network
    
    def train(self, generations=50):
        """Main training loop with multi-lap focus"""
        print("🚗 Starting Enhanced Genetic Algorithm Training")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {generations}")
        print(f"Maximum laps per individual: {self.max_laps}")
        print("-" * 70)
        
        for generation in range(generations):
            self.generation = generation
            
            # Evaluate population
            fitness_scores, lap_counts = self.evaluate_population()
            
            # Statistics
            avg_fitness = np.mean(fitness_scores)
            max_fitness = np.max(fitness_scores)
            min_fitness = np.min(fitness_scores)
            avg_laps = np.mean(lap_counts)
            max_laps = np.max(lap_counts)
            
            self.fitness_history.append({
                'generation': generation,
                'avg_fitness': avg_fitness,
                'max_fitness': max_fitness,
                'min_fitness': min_fitness,
                'avg_laps': avg_laps,
                'max_laps': max_laps,
                'individuals_with_laps': sum(1 for laps in lap_counts if laps > 0)
            })
            
            print(f"\nGeneration {generation} Summary:")
            print(f"  Average Fitness: {avg_fitness:8.1f}")
            print(f"  Best Fitness:    {max_fitness:8.1f}")
            print(f"  Worst Fitness:   {min_fitness:8.1f}")
            print(f"  Average Laps:    {avg_laps:8.2f}")
            print(f"  Max Laps:        {max_laps:8.0f}")
            print(f"  Individuals completing ≥1 lap: {sum(1 for laps in lap_counts if laps >= 1)}/{self.population_size}")
            
            # Save best individual every 10 generations
            if generation % 10 == 0:
                self.save_best_individual(f'genetic_multilap_gen_{generation}.json')
            
            # Early stopping if we achieve max laps
            if max_laps >= self.max_laps:
                print(f"🎉 Individual completed maximum {self.max_laps} laps! Early success!")
                self.save_best_individual('genetic_multilap_champion.json')
            
            # Create next generation
            if generation < generations - 1:
                self.create_next_generation(fitness_scores, lap_counts)
        
        # Final save
        self.save_best_individual('genetic_multilap_final.json')
        print(f"\n🏆 Training completed!")
        print(f"Best overall fitness: {self.best_fitness:.1f}")
        print(f"Best laps completed: {self.best_laps_completed}")
    
    def test_individual(self, network, episodes=3):
        """Test a trained individual"""
        print("🧪 Testing individual...")
        
        total_laps = 0
        for episode in range(episodes):
            print(f"\nTest Episode {episode + 1}/{episodes}")
            fitness, steps, laps, lap_times = self.evaluate_individual(
                network, max_steps=10000, render=True
            )
            
            total_laps += laps
            avg_lap_time = np.mean(lap_times) if lap_times else 0
            
            print(f"Result: {laps} laps | Fitness: {fitness:.1f} | Steps: {steps}")
            if lap_times:
                print(f"Lap times: {lap_times}")
                print(f"Average lap time: {avg_lap_time:.1f} steps")
        
        print(f"\nOverall: {total_laps} total laps across {episodes} episodes")
        print(f"Average laps per episode: {total_laps/episodes:.2f}")

def main():
    print("🧬 Enhanced Genetic Algorithm Car Training (4-Direction + Multi-Lap)")
    print("1. Train new population")
    print("2. Test saved individual")
    print("3. Continue training from checkpoint")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice == '1':
        # Train new population
        population_size = int(input("Population size (default 20): ") or "20")
        generations = int(input("Number of generations (default 50): ") or "50")
        max_laps = int(input("Maximum laps per individual (default 5): ") or "5")
        render = input("Enable rendering for best individual? (y/n): ").lower() == 'y'
        
        trainer = GeneticCarTrainer(population_size=population_size, render=render, max_laps=max_laps)
        trainer.train(generations=generations)
        
    elif choice == '2':
        # Test saved individual
        filename = input("Enter saved file (e.g., genetic_multilap_final.json): ")
        try:
            trainer = GeneticCarTrainer(render=True)
            network = trainer.load_individual(filename)
            trainer.test_individual(network, episodes=3)
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error loading file: {e}")
    
    elif choice == '3':
        # Continue training
        filename = input("Enter checkpoint file: ")
        try:
            max_laps = int(input("Maximum laps per individual (default 5): ") or "5")
            trainer = GeneticCarTrainer(render=True, max_laps=max_laps)
            best_network = trainer.load_individual(filename)
            
            # Create new population based on best individual
            trainer.population = [best_network]  # Keep the best
            for _ in range(trainer.population_size - 1):
                new_individual = SimpleNeuralNetwork()
                new_individual.set_weights(best_network.get_weights())
                new_individual.mutate(mutation_rate=0.2, mutation_strength=0.2)
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