import pygame
import numpy as np
import math
import os
from PIL import Image
import gym
from gym import spaces

class CarEnvironment:
    def __init__(self, width=800, height=600, map_path="static/track_map.png", car_path="static/car_blue_3.png"):
        pygame.init()
        
        # Display settings
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Self-Driving Car Environment")
        
        # Load images
        self.map_path = map_path
        self.car_path = car_path
        
        # Camera
        self.camera = {
            'x': 0,
            'y': 0,
            'width': width,
            'height': height
        }
        
        # Car properties
        self.car = {
            'x': 600,
            'y': 1650,
            'width': 50,
            'height': 30,
            'radius': 15,
            'angle': 0,
            'velocity': {'x': 0, 'y': 0},
            'speed': 0,
            'max_speed': 10,
            'acceleration': 0,
            'max_acceleration': 0.5,
            'friction': 0.96,
            'turn_speed': 0.065,
            'color': (52, 152, 219)
        }
        
        self.load_images()
        
        # Sensors
        self.sensors = {
            'front': {'angle': 0, 'distance': 300, 'detection': float('inf')},
            'left': {'angle': -40 * math.pi / 180, 'distance': 250, 'detection': float('inf')},
            'right': {'angle': 40 * math.pi / 180, 'distance': 250, 'detection': float('inf')},
            'left_1': {'angle': -20 * math.pi / 180, 'distance': 250, 'detection': float('inf')},
            'right_1': {'angle': 20 * math.pi / 180, 'distance': 250, 'detection': float('inf')}
        }
        
        # Control keys
        self.keys = {
            'up': False,
            'down': False,
            'left': False,
            'right': False
        }
        
        # Game state
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_initialized = True
        
        # Initialize camera
        self.update_camera()
        
        # For gym-like interface
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # [steering, throttle]
        self.observation_space = spaces.Box(low=0, high=1000, shape=(5,), dtype=np.float32)  # [speed, angle, 3 sensors, position]
        
        # Lap completion
        self.completed_lap = False
        self.checkpoint_hit = False
        self.no_progress_steps = 0 

    def load_images(self):
        """Load map and car images, create fallbacks if not found"""
        try:
            # Load map
            if os.path.exists(self.map_path):
                self.map_image = pygame.image.load(self.map_path).convert()
            else:
                self.map_image = self.create_fallback_map()
                
            # Load car
            if os.path.exists(self.car_path):
                self.car_image = pygame.image.load(self.car_path).convert_alpha()
            else:
                self.car_image = self.create_fallback_car()
                
            # Scale car image
            self.car_image = pygame.transform.scale(self.car_image, (self.car['width'], self.car['height']))
            
            # Create map surface for pixel detection
            self.map_array = pygame.surfarray.array3d(self.map_image)
            
        except Exception as e:
            print(f"Error loading images: {e}")
    
    def is_on_road(self, x, y):
        """Check if point is on road (gray color)"""
        if x < 0 or x >= self.map_image.get_width() or y < 0 or y >= self.map_image.get_height():
            return False
        
        try:
            # Get pixel color
            color = self.map_image.get_at((int(x), int(y)))
            r, g, b = color[:3]
            
            # Road is RGB(94,94,94) with tolerance ±10
            return (abs(r - 94) <= 10 and abs(g - 94) <= 10 and abs(b - 94) <= 10)
        except:
            return False
    
    def is_white_line(self, x, y):
        """Check if point is on white line"""
        if x < 0 or x >= self.map_image.get_width() or y < 0 or y >= self.map_image.get_height():
            return False
        
        try:
            color = self.map_image.get_at((int(x), int(y)))
            r, g, b = color[:3]
            return r > 220 and g > 220 and b > 220
        except:
            return False
    
    def update_camera(self):
        """Update camera position"""
        self.camera['x'] = self.car['x'] - self.camera['width'] / 2
        self.camera['y'] = self.car['y'] - self.camera['height'] / 2
        
        # Clamp camera to map bounds
        self.camera['x'] = max(0, min(self.camera['x'], self.map_image.get_width() - self.camera['width']))
        self.camera['y'] = max(0, min(self.camera['y'], self.map_image.get_height() - self.camera['height']))
    
    def update_sensors(self):
        """Update sensor readings"""
        for sensor_key, sensor in self.sensors.items():
            sensor_angle = self.car['angle'] + sensor['angle']
            sensor['detection'] = sensor['distance']
            
            # Scan along sensor ray
            for d in range(10, int(sensor['distance']) + 1, 5):
                px = self.car['x'] + math.cos(sensor_angle) * d
                py = self.car['y'] + math.sin(sensor_angle) * d
                
                # Check bounds
                if px < 0 or px >= self.map_image.get_width() or py < 0 or py >= self.map_image.get_height():
                    sensor['detection'] = d - 5
                    break
                
                # Check white lines
                if self.is_white_line(px, py):
                    sensor['detection'] = d
                    break
                
                # Check off road
                if not self.is_on_road(px, py):
                    sensor['detection'] = max(10, d - 5)
                    break
    
    def update_car_physics(self):
        """Update car physics and movement"""
        # Calculate acceleration based on speed
        max_a = self.car['max_acceleration']
        v = abs(self.car['speed'])
        v_max = self.car['max_speed']
        
        def custom_acceleration(v):
            return max_a * (1 - v / v_max)
        
        # Handle acceleration
        if self.keys['up']:
            self.car['acceleration'] += (custom_acceleration(v) - self.car['acceleration']) * 0.2
        elif self.keys['down']:
            self.car['acceleration'] += ((-custom_acceleration(v) * 0.5) - self.car['acceleration']) * 0.2
        else:
            self.car['acceleration'] += (0 - self.car['acceleration']) * 0.02
        
        # Update speed
        self.car['speed'] += self.car['acceleration']
        self.car['speed'] *= self.car['friction']
        self.car['speed'] = min(self.car['max_speed'], self.car['speed'])
        
        # Handle steering
        if abs(self.car['speed']) > 0.1:
            if self.keys['left']:
                self.car['angle'] -= self.car['turn_speed'] * abs(self.car['speed']) / self.car['max_speed']
            if self.keys['right']:
                self.car['angle'] += self.car['turn_speed'] * abs(self.car['speed']) / self.car['max_speed']
        
        # Update velocity
        self.car['velocity']['x'] = math.cos(self.car['angle']) * self.car['speed']
        self.car['velocity']['y'] = math.sin(self.car['angle']) * self.car['speed']
        
        # Update position
        self.car['x'] += self.car['velocity']['x']
        self.car['y'] += self.car['velocity']['y']
        
        # Boundary collision
        if self.car['x'] < self.car['radius']:
            self.car['x'] = self.car['radius']
            self.car['velocity']['x'] *= -0.5
            self.car['speed'] *= 0.5
        if self.car['x'] > self.map_image.get_width() - self.car['radius']:
            self.car['x'] = self.map_image.get_width() - self.car['radius']
            self.car['velocity']['x'] *= -0.5
            self.car['speed'] *= 0.5
        if self.car['y'] < self.car['radius']:
            self.car['y'] = self.car['radius']
            self.car['velocity']['y'] *= -0.5
            self.car['speed'] *= 0.5
        if self.car['y'] > self.map_image.get_height() - self.car['radius']:
            self.car['y'] = self.map_image.get_height() - self.car['radius']
            self.car['velocity']['y'] *= -0.5
            self.car['speed'] *= 0.5
        
        # Off-road reaction
        if not self.is_on_road(self.car['x'], self.car['y']):
            self.car['x'] -= self.car['velocity']['x'] * 0.5
            self.car['y'] -= self.car['velocity']['y'] * 0.5
            self.car['speed'] *= -0.3
    
    def draw_background(self):
        """Draw the map background"""
        # Calculate visible area
        src_rect = pygame.Rect(
            self.camera['x'], self.camera['y'],
            self.camera['width'], self.camera['height']
        )
        
        # Draw map portion
        self.screen.blit(self.map_image, (0, 0), src_rect)
    
    def draw_car(self):
        """Draw the car"""
        # Calculate car position on screen
        car_x = self.car['x'] - self.camera['x']
        car_y = self.car['y'] - self.camera['y']
        
        # Rotate car image
        rotated_car = pygame.transform.rotate(self.car_image, -self.car['angle'] * 180 / math.pi)
        car_rect = rotated_car.get_rect(center=(car_x, car_y))
        
        self.screen.blit(rotated_car, car_rect)
    
    def draw_sensors(self):
        """Draw sensor rays"""
        for sensor_key, sensor in self.sensors.items():
            sensor_angle = self.car['angle'] + sensor['angle']
            
            # Start point
            start_x = self.car['x'] - self.camera['x']
            start_y = self.car['y'] - self.camera['y']
            
            # End point
            end_x = start_x + math.cos(sensor_angle) * sensor['detection']
            end_y = start_y + math.sin(sensor_angle) * sensor['detection']
            
            # Color based on detection
            color = (231, 76, 60) if sensor['detection'] < sensor['distance'] else (140, 140, 140)
            
            # Draw ray
            pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 2)
            
            # Draw detection point
            if sensor['detection'] < sensor['distance']:
                pygame.draw.circle(self.screen, (231, 76, 60), (int(end_x), int(end_y)), 4)
    
    def draw_ui(self):
        """Draw UI information"""
        font = pygame.font.Font(None, 36)
        
        # Speed
        speed_text = font.render(f"Speed: {abs(self.car['speed'] * 10):.1f} km/h", True, (255, 255, 255))
        self.screen.blit(speed_text, (10, 10))
        
        # Angle
        angle_text = font.render(f"Angle: {self.car['angle'] * 180 / math.pi:.1f}°", True, (255, 255, 255))
        self.screen.blit(angle_text, (10, 50))
        
        # Position
        pos_text = font.render(f"Position: ({self.car['x']:.0f}, {self.car['y']:.0f})", True, (255, 255, 255))
        self.screen.blit(pos_text, (10, 90))
        
        # Sensors
        y_offset = 130
        for sensor_key, sensor in self.sensors.items():
            detection_str = f"{sensor['detection']:.0f}px" if sensor['detection'] != float('inf') else "∞"
            sensor_text = font.render(f"{sensor_key}: {detection_str}", True, (255, 255, 255))
            self.screen.blit(sensor_text, (10, y_offset))
            y_offset += 40
    
    def reset_car(self):
        """Reset car to starting position"""
        self.car['x'] = 600
        self.car['y'] = 1650
        self.car['angle'] = 0
        self.car['velocity'] = {'x': 0, 'y': 0}
        self.car['speed'] = 0
        self.car['acceleration'] = 0
        self.completed_lap = False
        self.checkpoint_hit = False
        self.no_progress_steps = 0
        self.last_y = self.car['y']  # Thêm dòng này
        self.prev_angle = self.car['angle']  # Thêm dòng này
    
    def get_state(self):
        """
        The function `get_state` returns a numpy array containing normalized values representing the
        car's speed, sensor readings, and maximum speed.
        :return: The `get_state` method returns a numpy array containing normalized values for the car's
        speed, sensor readings in front, left, right, left_1, and right_1 directions. The values are
        normalized based on certain maximum values (max_speed, 300, 250) to ensure they fall within a
        specific range for processing or analysis.
        """
        return np.array([
            self.car['speed'] / self.car['max_speed'],  # Normalize speed
            self.sensors['front']['detection'] / 300,  # Normalize sensor readings
            self.sensors['left']['detection'] / 250,
            self.sensors['right']['detection'] / 250,
            self.sensors['left_1']['detection'] / 250,
            self.sensors['right_1']['detection'] / 250
        ], dtype=np.float32)
    
    def step(self, action=None):
        """Improved step function with better reward shaping"""
        if action is not None:
            # action = [steering, throttle]
            steering, throttle = action
            
            # Map action to keys với threshold nhạy hơn
            self.keys['up'] = throttle > 0.05
            self.keys['down'] = throttle < -0.05
            self.keys['left'] = steering < -0.05
            self.keys['right'] = steering > 0.05
        
        # Store previous state for reward calculation
        prev_x, prev_y = self.car['x'], self.car['y']
        prev_speed = self.car['speed']
        
        # Update physics
        self.update_car_physics()
        self.update_camera()
        self.update_sensors()
        
        # Initialize reward
        reward = 0
        done = False
        
        # 1. Basic survival reward
        reward += 0.1
        
        # 2. Speed reward - encourage forward movement
        speed_reward = self.car['speed'] * 5  # Scale speed reward
        if self.car['speed'] > 0.1:
            reward += speed_reward
        else:
            reward -= 1  # Penalty for being too slow
        
        # 3. Road adherence
        if not self.is_on_road(self.car['x'], self.car['y']):
            reward -= 50  # Heavy penalty for going off road
            done = True
        else:
            reward += 2  # Reward for staying on road
        
        # 4. Sensor-based safety rewards
        min_sensor = min(
            self.sensors['front']['detection'],
            self.sensors['left']['detection'], 
            self.sensors['right']['detection']
        )
        
        if min_sensor < 15:  # Very close to obstacle
            reward -= 10
        elif min_sensor < 30:  # Close to obstacle
            reward -= 3
        elif min_sensor > 80:  # Safe distance
            reward += 1
        
        # 5. Forward progress reward (encourage moving toward goal)
        progress_reward = 0
        if hasattr(self, 'last_y'):
            # Reward for moving "forward" on track (decreasing Y coordinate)
            if self.car['y'] < self.last_y:  # Moving forward
                progress_reward = (self.last_y - self.car['y']) * 0.1
                reward += progress_reward
            elif self.car['y'] > self.last_y:  # Moving backward
                reward -= (self.car['y'] - self.last_y) * 0.05
        self.last_y = self.car['y']
        
        # 6. Smooth driving reward (penalize excessive turning)
        if hasattr(self, 'prev_angle'):
            angle_change = abs(self.car['angle'] - self.prev_angle)
            if angle_change > 0.1:  # Sharp turn
                reward -= angle_change * 2
        self.prev_angle = self.car['angle']
        
        # 7. Checkpoint system
        if self.car['y'] < 800 and not self.checkpoint_hit:
            self.checkpoint_hit = True
            reward += 30  # Checkpoint bonus
            print("🎯 Checkpoint reached!")
        
        # 8. Lap completion
        if (1585 < self.car['y'] < 1715 and 
            580 < self.car['x'] < 620 and 
            self.checkpoint_hit):
            self.completed_lap = True
            reward += 100  # Huge bonus for completing lap
            done = True
            print("🏁 Lap completed!")
        
        # 9. Timeout prevention
        if self.car['speed'] < 0.1:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        
        if self.no_progress_steps > 200:  # Increased tolerance
            reward -= 20
            done = True
        
        # 10. Encourage staying in center of road
        # Check if car is roughly in center (this is track-specific)
        if 400 < self.car['x'] < 600:  # Rough center for this track
            reward += 0.5
        
        # Clip reward to reasonable range
        reward = np.clip(reward, -100, 100)
        
        # Thêm reward cho việc giữ làn đường
        center_line_x = 550  # Tọa độ x của vạch giữa
        distance_to_center = abs(self.car['x'] - center_line_x)
        lane_keeping_reward = 1.0 / (1.0 + distance_to_center)
        reward += lane_keeping_reward
        
        # Tăng reward cho tốc độ ổn định
        optimal_speed = 6.0  # Tốc độ lý tưởng
        speed_reward = -abs(self.car['speed'] - optimal_speed)
        reward += speed_reward
        
        return self.get_state(), reward, done, {
            'speed': self.car['speed'],
            'progress': progress_reward,
            'sensors': min_sensor
        }
    
    def render(self):
        """Render the environment"""
        self.screen.fill((0, 0, 0))
        
        self.draw_background()
        self.draw_sensors()
        self.draw_car()
        self.draw_ui()
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.keys['up'] = True
                elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.keys['down'] = True
                elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.keys['left'] = True
                elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.keys['right'] = True
                elif event.key == pygame.K_r:
                    self.reset_car()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.keys['up'] = False
                elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.keys['down'] = False
                elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.keys['left'] = False
                elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.keys['right'] = False
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update_car_physics()
            self.update_camera()
            self.update_sensors()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()

# Example usage
if __name__ == "__main__":
    env = CarEnvironment()
    env.run()