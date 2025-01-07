import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
from MAML4 import MazeMAML
from tqdm import tqdm
import time
from MAMLawareGAN import MAMLAwareGAN  

#makes the dataset
def generate_complex_maze(width, height, loop_probability=0.3, multiple_paths=True):
    
    maze = np.ones((height, width), dtype=int)
    
    start = (1, 1)
    end = (height - 2, width - 2)
    stack = [start]
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    visited = set()
    i=0
    while stack:
        i+=1
        #print(i)
        current_cell = stack[-1]
        visited.add(current_cell)
        maze[current_cell] = 0  
        
        #find the adjacent cells
        neighbors = []
        for direction in directions:
            next_cell = (current_cell[0] + direction[0], current_cell[1] + direction[1])
            if 0 <= next_cell[0] < height and 0 <= next_cell[1] < width:
                neighbors.append(next_cell)
        
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        
        if unvisited_neighbors:
            next_cell = unvisited_neighbors[np.random.randint(0, len(unvisited_neighbors))]
            maze[(current_cell[0] + next_cell[0]) // 2, (current_cell[1] + next_cell[1]) // 2] = 0
            stack.append(next_cell)
        else:
           
            stack.pop()
            
            #this creates loops and multiple paths
            if np.random.random() < loop_probability:
                for neighbour in neighbors:
                    if maze[neighbour] == 0 and neighbour not in stack:
                        maze[(current_cell[0] + neighbour[0]) // 2, (current_cell[1] + neighbour[1]) // 2] = 0
                        if multiple_paths:
                            stack.append(neighbour)  #continue from this cell to create multiple paths
                        break
    
    
    ensure_path(maze,start,end)
    maze[start] = 2
    maze[end] = 3
    return maze

def ensure_path(maze, start, end):
    #A* algorithm to find a path from start to end
    def heuristic(a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    def get_neighbors(pos):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if 0 <= new_pos[0] < maze.shape[0] and 0 <= new_pos[1] < maze.shape[1]:
                neighbors.append(new_pos)
        return neighbors
    
    open_set = {start}
    closed_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    i=0
    while open_set:
        i+=1
        #print(i)
        current = min(open_set, key=lambda pos: f_score[pos])
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            for pos in path:
                maze[pos] = 0
            return
        
        open_set.remove(current)
        closed_set.add(current)
        
        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue
            
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score[neighbor]:
                continue
            
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
    
    x, y = start
    while (x, y) != end:
        if x < end[0]:
            x += 1
        elif x > end[0]:
            x -= 1
        elif y < end[1]:
            y += 1
        elif y > end[1]:
            y -= 1
        maze[x, y] = 0
class goodEvaulator:
    def __init__(self):
        self.gan_maml = MazeMAML(maze_size=11, vision_range=2, load_checkpoint=False)
        self.standard_maml = MazeMAML(maze_size=11, vision_range=2, load_checkpoint=False)
        self.adaptive_gan = MAMLAwareGAN()  # Your adaptive GAN
        
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 6))
        self.setup_plot()
        
    def load_all_models(self, gan_maml_path, standard_maml_path, adaptive_gan_path):
        try:
            #load MAML agents
            gan_checkpoint = torch.load(gan_maml_path)
            self.gan_maml.policy.load_state_dict(gan_checkpoint['model_state_dict'])
            
            std_checkpoint = torch.load(standard_maml_path)
            self.standard_maml.policy.load_state_dict(std_checkpoint['model_state_dict'])
            
            #load adaptive GAN
            if os.path.exists(adaptive_gan_path):
                checkpoint = torch.load(adaptive_gan_path)
                self.adaptive_gan.generator.load_state_dict(checkpoint['generator_state_dict'])
                print("Loaded adaptive GAN checkpoint")
            else:
                print("No adaptive GAN checkpoint found, will use regular maze generation")
            
            print("Successfully loaded all available checkpoints!")
            return True
        except Exception as e:
            print(f"Error loading checkpoints: {e}")
            return False
    
    def generate_the_mazes(self, num_mazes=50):
        regular_mazes = [generate_complex_maze(11,11) for _ in range(num_mazes)]
        
        try:
            adaptive_mazes = [self.adaptive_gan.generate_maze() for _ in range(num_mazes)]
            has_adaptive = True
        except:
            print("Could not generate adaptive mazes, will test only on regular mazes")
            adaptive_mazes = []
            has_adaptive = False
        
        return regular_mazes, adaptive_mazes, has_adaptive

    def run_comprehensive_comparison(self, num_mazes=50, max_steps=100):
        print("\nStarting comparison...")
        
        #generate mazes
        regular_mazes, adaptive_mazes, has_adaptive = self.generate_the_mazes(num_mazes)
        
        results = {
            'regular_mazes': {
                'gan_maml': defaultdict(list),
                'standard_maml': defaultdict(list)
            }
        }
        
        if has_adaptive:
            results['adaptive_mazes'] = {
                'gan_maml': defaultdict(list),
                'standard_maml': defaultdict(list)
            }
        

        print("\nTesting on regular mazes...")
        self.run_the_mazes(regular_mazes, results['regular_mazes'], 
                          max_steps, "Regular Maze")
        

        if has_adaptive:
            print("\nTesting on adaptive mazes...")
            self.run_the_mazes(adaptive_mazes, results['adaptive_mazes'], 
                              max_steps, "Adaptive Maze")
        
        self.save_comprehensive_results(results)
        return results
    
    def run_the_mazes(self, mazes, results, max_steps, maze_type):
        for maze_idx, maze in enumerate(mazes):
            gan_info = {'visited': {(1, 1): 1}, 'current_pos': (1, 1), 
                       'steps': 0, 'success': False}
            std_info = {'visited': {(1, 1): 1}, 'current_pos': (1, 1), 
                       'steps': 0, 'success': False}
            
            for step in range(max_steps):
                if not gan_info['success']:
                    gan_trajectory = self.gan_maml.collect_trajectory_with_info(maze)
                    gan_info.update({
                        'visited': gan_trajectory['visited'],
                        'current_pos': gan_trajectory['path'][-1] if gan_trajectory['path'] else (1, 1),
                        'steps': step + 1,
                        'success': gan_trajectory['success']
                    })
                
                if not std_info['success']:
                    std_trajectory = self.standard_maml.collect_trajectory_with_info(maze)
                    std_info.update({
                        'visited': std_trajectory['visited'],
                        'current_pos': std_trajectory['path'][-1] if std_trajectory['path'] else (1, 1),
                        'steps': step + 1,
                        'success': std_trajectory['success']
                    })
                
                self.visualize_step(maze, gan_info, std_info, maze_idx + 1, maze_type)
                
                if gan_info['success'] and std_info['success']:
                    time.sleep(0.5)
                    break
                if step == max_steps - 1:
                    time.sleep(0.5)
            
            for agent_type, info in [('gan_maml', gan_info), ('standard_maml', std_info)]:
                results[agent_type]['success_rates'].append(int(info['success']))
                results[agent_type]['steps'].append(info['steps'])
                results[agent_type]['unique_positions'].append(len(info['visited']))
                if info['success']:
                    results[agent_type]['success_path_lengths'].append(len(info['visited']))
    
    def save_comprehensive_results(self, results):
        """Save and display comprehensive results with step comparisons"""
        print("\n=== Comprehensive Results ===")
        
        for maze_type, agents in results.items():
            print(f"\n{maze_type.upper()}")
            
            gan_metrics = agents['gan_maml']
            std_metrics = agents['standard_maml']
            
            for agent_type, metrics in agents.items():
                success_rate = np.mean(metrics['success_rates'])
                avg_steps = np.mean(metrics['steps'])
                avg_path = np.mean(metrics['success_path_lengths']) if metrics['success_path_lengths'] else float('inf')
                
                print(f"\n{agent_type.upper()}:")
                print(f"Success Rate: {success_rate:.1%}")
                print(f"Average Steps: {avg_steps:.1f}")
                print(f"Average Success Path Length: {avg_path:.1f}")
            
            total_mazes = len(gan_metrics['steps'])
            gan_better = 0
            equal_steps = 0
            both_success = 0
            both_success_gan_better = 0
            
            for i in range(total_mazes):
                if gan_metrics['success_rates'][i] and std_metrics['success_rates'][i]:
                    both_success += 1
                    if gan_metrics['steps'][i] < std_metrics['steps'][i]:
                        gan_better += 1
                        both_success_gan_better += 1
                    elif gan_metrics['steps'][i] == std_metrics['steps'][i]:
                        equal_steps += 1
            
            print("\nStep Comparison:")
            print(f"Total Mazes: {total_mazes}")
            print(f"Both Successful: {both_success}")
            print(f"GAN-MAML took fewer steps: {gan_better} ({(gan_better/total_mazes)*100:.1f}% of all mazes)")
            if both_success > 0:
                print(f"GAN-MAML better when both succeed: {both_success_gan_better} ({(both_success_gan_better/both_success)*100:.1f}% of mutually successful mazes)")
            print(f"Equal steps: {equal_steps} ({(equal_steps/total_mazes)*100:.1f}%)")
            
            step_differences = []
            for i in range(total_mazes):
                if gan_metrics['success_rates'][i] and std_metrics['success_rates'][i]:
                    diff = std_metrics['steps'][i] - gan_metrics['steps'][i]
                    step_differences.append(diff)
            
            if step_differences:
                avg_diff = np.mean(step_differences)
                print(f"Average step difference (when both succeed): {avg_diff:.1f} steps")
                print(f"Positive means GAN-MAML was faster by that many steps")
        
        os.makedirs('evaluation_results', exist_ok=True)
        with open('evaluation_results/comprehensive_results.json', 'w') as f:
            json_results = {}
            for maze_type, agents in results.items():
                json_results[maze_type] = {}
                for agent_type, metrics in agents.items():
                    json_results[maze_type][agent_type] = {
                        k: [float(v) if isinstance(v, np.float32) else v 
                        for v in vals] if isinstance(vals, list) else vals
                        for k, vals in metrics.items()
                    }
            json.dump(json_results, f, indent=4)
            
        plt.figure(figsize=(10, 6))
        plt.hist(step_differences, bins=20, alpha=0.75)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Distribution of Step Differences\n(Positive = GAN-MAML was faster)')
        plt.xlabel('Standard MAML steps - GAN-MAML steps')
        plt.ylabel('Count')
        plt.savefig('evaluation_results/step_differences.png')
        plt.close()
    
    def setup_plot(self):
        """Initialize the plot layout"""
        self.axes[0].set_title('GAN-MAML Agent')
        self.axes[1].set_title('Standard MAML Agent')
        
        self.gan_text = self.fig.text(0.25, 0.02, '', ha='center', fontfamily='monospace')
        self.std_text = self.fig.text(0.75, 0.02, '', ha='center', fontfamily='monospace')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    def visualize_step(self, maze, gan_info, std_info, maze_count, maze_type):
        """Update the visualization with current state"""
        self.axes[0].clear()
        self.axes[1].clear()
        
        gan_maze = self.create_visualization_maze(maze, gan_info)
        std_maze = self.create_visualization_maze(maze, std_info)
        
        self.axes[0].imshow(gan_maze, cmap='viridis')
        self.axes[0].set_title(f'GAN-MAML Agent\n{maze_type} {maze_count}')
        self.axes[0].axis('off')
        
        self.axes[1].imshow(std_maze, cmap='viridis')
        self.axes[1].set_title(f'Standard MAML Agent\n{maze_type} {maze_count}')
        self.axes[1].axis('off')
        
        self.gan_text.set_text(self.format_info_text(gan_info))
        self.std_text.set_text(self.format_info_text(std_info))
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def create_visualization_maze(self, base_maze, info):
        maze_vis = base_maze.copy()
        
        if 'visited' in info:
            max_visits = max(info['visited'].values()) if info['visited'] else 1
            for pos, count in info['visited'].items():
                x, y = pos
                if maze_vis[x, y] != 1:
                    intensity = count / max_visits
                    maze_vis[x, y] = 5 + intensity * 4
        
        if 'current_pos' in info:
            x, y = info['current_pos']
            if maze_vis[x, y] != 1:
                maze_vis[x, y] = 10
        
        return maze_vis
    
    def format_info_text(self, info):
        """Format information text for display"""
        return '\n'.join([
            f"Steps: {info.get('steps', 0)}",
            f"Visits: {len(info.get('visited', {}))}",
            f"Status: {'Success' if info.get('success', False) else 'In Progress'}"
        ])

if __name__ == "__main__":
    evaluator = goodEvaulator()
    
    gan_maml_path = "checkpoints/maml2gan_vision2.pth"
    standard_maml_path = "checkpoints/control_maml.pth"
    adaptive_gan_path = "checkpoints/gan_phase2_best.pth" 
    if evaluator.load_all_models(gan_maml_path, standard_maml_path, adaptive_gan_path):
        results = evaluator.run_comprehensive_comparison(
            num_mazes=50, 
            max_steps=100  
        )