import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
import time
from collections import deque


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_DIM = 100
MAZE_SIZE = 11
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.name = 'generator'
        
        self.model = nn.Sequential(
            nn.Linear(NOISE_DIM, 3*3*256),
            nn.BatchNorm1d(3*3*256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Unflatten(1, (256, 3, 3)),
            
            # 3x3 -> 6x6
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 6x6 -> 12x12
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 12x12 -> 11x11 (using conv layer to reduce size)
            nn.Conv2d(64, 1, 2, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

def visualize_maze(maze):

    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap='viridis')
    plt.axis('off')
    plt.title('Generated Maze')
    plt.show()


class PolicyNetwork(nn.Module):
    def __init__(self, input_size=11, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
        self.temperature = 4.0
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        logits = self.net(x)

        probs = F.softmax(logits / self.temperature, dim=-1)

        return probs / probs.sum()

class MazeMAML:
    """
    MAML implementation for maze solving
    Follows the MAML algorithm:
    1. Sample batch of tasks (mazes)
    2. For each task:
        - Collect pre-adaptation trajectories
        - Perform inner loop update
        - Collect post-adaptation trajectories
    3. Perform meta-update across all tasks
    """
    def __init__(self, maze_size=11, vision_range=2, load_checkpoint=True):
        self.maze_size = maze_size  
        self.vision_range = vision_range
        
        # Calculate input size
        view_size = (2 * vision_range + 1) ** 2  # Local view size
        self.input_size = view_size + 4  # +2 for position, +2 for goal direction
        
        print(f"Initializing with input size: {self.input_size}")
        print(f"Vision range: {vision_range}")
        print(f"View size: {view_size}")
        
        # Initialize policy with correct input size
        self.policy = PolicyNetwork(input_size=self.input_size)
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=0.0001)
        self.inner_lr = 0.03
        
        # Load the GAN
        self.generator = Generator().to(DEVICE)
        gan_checkpoint = torch.load('checkpoints/generator_checkpoint.pth')
        self.generator.load_state_dict(gan_checkpoint['model_state_dict'])
        self.generator.eval()
        
        self.buffer_size = 100
        self.replay_buffer = []
        
        self.checkpoint_path = f'checkpoints/maml2gan_vision{vision_range}.pth'
        self.start_epoch = 0
        
        if load_checkpoint:
            self.load_checkpoint()
    
    def generate_maze(self):
        """Generate a maze using the loaded GAN with proper size checks"""
        with torch.no_grad():
            noise = torch.randn(1, NOISE_DIM, device=DEVICE)
            generated_image = self.generator(noise).cpu().squeeze().numpy()
            # Ensure proper size and values
            maze = np.round((generated_image * 2) + 2).astype('int32')
            # Ensure start and end positions are clear
            maze[1, 1] = 0  # Start position
            maze[self.maze_size-2, self.maze_size-2] = 0  # Goal position
            return maze

    def save_checkpoint(self, epoch, meta_loss, success_rate=None):

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'loss': meta_loss,
            'success_rate': success_rate,
            'vision_range': self.vision_range,
            'input_size': self.input_size,  
            'hyperparameters': {
                'temperature': self.policy.temperature,
                'inner_lr': self.inner_lr
            }
        }
        torch.save(checkpoint_data, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")
        print(f"Input size saved: {self.input_size}")

    def load_checkpoint(self):
        """Load checkpoint with proper input size compatibility check"""
        try:
            checkpoint = torch.load(self.checkpoint_path)
            print(f"Found checkpoint with input size: {checkpoint['input_size']}")
            print(f"Current input size: {self.input_size}")
            
            if (checkpoint['vision_range'] == self.vision_range and 
                checkpoint['input_size'] == self.input_size): 
                
                self.policy.load_state_dict(checkpoint['model_state_dict'])
                self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch']
                
                # Load hyperparameters if they exist
                if 'hyperparameters' in checkpoint:
                    self.policy.temperature = checkpoint['hyperparameters']['temperature']
                    self.inner_lr = checkpoint['hyperparameters']['inner_lr']
                
                print(f"Loaded checkpoint from epoch {self.start_epoch}")
                if 'success_rate' in checkpoint:
                    print(f"Previous success rate: {checkpoint['success_rate']:.2%}")
                    
                return True
            else:
                print("\nCheckpoint incompatibility:")
                print(f"Saved vision range: {checkpoint['vision_range']}, Current: {self.vision_range}")
                print(f"Saved input size: {checkpoint['input_size']}, Current: {self.input_size}")
                print("Starting fresh.")
                return False
                
        except FileNotFoundError:
            print("No checkpoint found. Starting fresh.")
            return False

    def get_state(self, maze, pos):
        x, y = pos
        goal_pos = (self.maze_size-2, self.maze_size-2)
        pos_norm = np.array(pos, dtype=np.float32) / self.maze_size
        
        # Calculate goal direction (normalized to [-1, 1] range)
        goal_direction = np.array([
            (goal_pos[0] - x) / self.maze_size,  # x distance to goal
            (goal_pos[1] - y) / self.maze_size   # y distance to goal
        ], dtype=np.float32)
        
        # Get local view
        view_size = 2 * self.vision_range + 1
        local_view = np.ones((view_size, view_size))  # Initialize with walls (1)
        
        # Fill in the visible portion of the maze
        for i in range(-self.vision_range, self.vision_range + 1):
            for j in range(-self.vision_range, self.vision_range + 1):
                abs_x = x + i
                abs_y = y + j
                if (0 <= abs_x < self.maze_size and 
                    0 <= abs_y < self.maze_size):
                    # Convert to local view coordinates
                    local_x = i + self.vision_range
                    local_y = j + self.vision_range
                    local_view[local_x, local_y] = maze[abs_x, abs_y] / 4.0
        
        # Concatenate position, goal direction, and flattened local view
        return np.concatenate([pos_norm, goal_direction, local_view.flatten()])

    def collect_trajectory(self, maze, policy=None):
        if policy is None:
            policy = self.policy
            
        start_pos = (1, 1)
        goal_pos = (self.maze_size-2, self.maze_size-2)
        current_pos = start_pos
        
        trajectory = []
        visited = {start_pos: 1}
        path = [current_pos]
        done = False
        recent_positions = deque(maxlen=4)  # Track recent positions to detect oscillation
        stuck_count = 0  # Counter for stuck detection
        
        step_reward = -0.005
        goal_reward = 15.0
        progress_weight = 1.0
        directional_weight = 0.5
        base_revisit_penalty = -0.3
        oscillation_penalty = -1.0  # New penalty for oscillating behavior
        
        max_steps = self.maze_size * 3
        initial_dist = abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])
        
        exploration_steps = min(20, max_steps // 4)
        
        for step in range(max_steps):
            state = self.get_state(maze, current_pos)
            recent_positions.append(current_pos)
            
            # Detect oscillation
            is_oscillating = False
            if len(recent_positions) >= 4:
                if list(recent_positions)[-4:] == list(recent_positions)[-2:] * 2:
                    is_oscillating = True
                    stuck_count += 1
                else:
                    stuck_count = max(0, stuck_count - 1)  # Gradually reduce stuck count
            
            with torch.no_grad():
                probs = policy(state)
                moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                valid_moves = []
                valid_probs = []
                
                goal_dx = goal_pos[0] - current_pos[0]
                goal_dy = goal_pos[1] - current_pos[1]
                
                # Collect all valid moves first
                for idx, (dx, dy) in enumerate(moves):
                    new_x, new_y = current_pos[0] + dx, current_pos[1] + dy
                    new_pos = (new_x, new_y)
                    
                    if (0 <= new_x < self.maze_size and 
                        0 <= new_y < self.maze_size and 
                        maze[new_x, new_y] != 1):
                        
                        # Base probability from policy
                        base_prob = probs[idx].item()
                        
                        # Different strategies based on situation
                        if is_oscillating or stuck_count > 2:
                            # When stuck, heavily favor unexplored positions
                            visit_count = visited.get(new_pos, 0)
                            if visit_count == 0:  # Unexplored position
                                base_prob *= 3.0
                            elif new_pos in recent_positions:  # Recent position
                                base_prob *= 0.1  # Strongly discourage recent positions
                        else:
                            # Normal exploration/exploitation
                            if step < exploration_steps:
                                visit_count = visited.get(new_pos, 0)
                                exploration_bonus = 2.0 if visit_count == 0 else 0.5
                                base_prob *= exploration_bonus
                            else:
                                # Direction bonus
                                direction_bonus = 0
                                if (dx > 0 and goal_dx > 0) or (dx < 0 and goal_dx < 0):
                                    direction_bonus += 0.3
                                if (dy > 0 and goal_dy > 0) or (dy < 0 and goal_dy < 0):
                                    direction_bonus += 0.3
                                
                                visit_count = visited.get(new_pos, 0)
                                visit_penalty = 0.7 ** visit_count
                                base_prob *= (1 + direction_bonus) * visit_penalty
                        
                        valid_moves.append((dx, dy))
                        valid_probs.append(max(1e-10, base_prob))
                
                if not valid_moves:
                    break
                
                # Normalize probabilities
                valid_probs = np.array(valid_probs)
                prob_sum = valid_probs.sum()
                if prob_sum > 0:
                    valid_probs = valid_probs / prob_sum
                else:
                    valid_probs = np.ones_like(valid_probs) / len(valid_probs)
                
                valid_probs = np.nan_to_num(valid_probs, nan=1.0/len(valid_probs))
                valid_probs = valid_probs / valid_probs.sum()
                
                # Select move
                try:
                    if stuck_count > 4:  # If severely stuck, force random unexplored move
                        unexplored_moves = [(i, move) for i, move in enumerate(valid_moves) 
                                        if (current_pos[0] + move[0], current_pos[1] + move[1]) not in recent_positions]
                        if unexplored_moves:
                            move_idx = random.choice(unexplored_moves)[0]
                        else:
                            move_idx = np.random.choice(len(valid_moves), p=valid_probs)
                    else:
                        move_idx = np.random.choice(len(valid_moves), p=valid_probs)
                    
                    dx, dy = valid_moves[move_idx]
                    action = moves.index((dx, dy))
                except ValueError as e:
                    print(f"Probability error: {valid_probs}")
                    move_idx = random.randrange(len(valid_moves))
                    dx, dy = valid_moves[move_idx]
                    action = moves.index((dx, dy))
                
                new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                reward = step_reward
                
                # Progress reward
                old_dist = abs(goal_pos[0] - current_pos[0]) + abs(goal_pos[1] - current_pos[1])
                new_dist = abs(goal_pos[0] - new_pos[0]) + abs(goal_pos[1] - new_pos[1])
                
                progress_reward = (old_dist - new_dist) * progress_weight
                overall_progress = 1.0 - (new_dist / initial_dist)
                progress_reward += overall_progress * 0.5
                
                reward += progress_reward
                
                # Additional penalties
                if is_oscillating:
                    reward += oscillation_penalty
                
                visit_count = visited.get(new_pos, 0)
                if visit_count > 0:
                    reward += base_revisit_penalty * (visit_count ** 0.5)
                
                visited[new_pos] = visit_count + 1
                
                if new_pos == goal_pos:
                    path_efficiency = len(set(path)) / len(path)
                    time_bonus = (max_steps - step) / max_steps
                    reward = goal_reward * (path_efficiency + time_bonus)
                    done = True
                
                current_pos = new_pos
                path.append(current_pos)
                trajectory.append((state, action, reward))
                
                if done:
                    break
        
        if not done:
            final_dist = abs(goal_pos[0] - current_pos[0]) + abs(goal_pos[1] - current_pos[1])
            final_reward = -final_dist * 0.5
            if trajectory:
                trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], final_reward)
        
        return trajectory, done, path, visited


    def collect_trajectory_with_info(self, maze):
        #collect trajectory and return info needed for GAN adaptation
        trajectory, success, path, visited = self.collect_trajectory(maze)
        start = (1, 1)
        end = (MAZE_SIZE-2, MAZE_SIZE-2)
        optimal_path_length = self.find_shortest_path_length(maze, start, end)
        trajectory_info = {
            'success': success,
            'path': path,
            'visited': visited,
            'optimal_path_length': optimal_path_length,
            'adaptation_steps': len(trajectory)
        }
        
        return trajectory_info

    #helper function to find optimal path length
    def find_shortest_path_length(self, maze, start, end):
        queue = deque([(start, 0)])  # (position, distance)
        visited = {start}
        
        while queue:
            (x, y), dist = queue.popleft()
            if (x, y) == end:
                return dist
                
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x, next_y = x + dx, y + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < maze.shape[0] and 
                    0 <= next_y < maze.shape[1] and 
                    maze[next_x, next_y] != 1 and
                    next_pos not in visited):
                    queue.append((next_pos, dist + 1))
                    visited.add(next_pos)
        
        return float('inf')
    def meta_update(self, task_trajectories):
        meta_loss = 0
        successful_adaptations = 0
        
        if self.replay_buffer:
            replay_samples = random.sample(self.replay_buffer, 
                                        min(3, len(self.replay_buffer)))
            task_trajectories.extend(replay_samples)
        
        for maze, trajectories in task_trajectories:
            task_policy = PolicyNetwork(input_size=self.input_size)
            task_policy.load_state_dict(self.policy.state_dict())
            
            inner_optimizer = optim.SGD(task_policy.parameters(), lr=self.inner_lr)
            
            for adapt_step in range(3):
                adapt_loss = 0
                for trajectory in trajectories:
                    returns = self._compute_returns([t[2] for t in trajectory], gamma=0.98)
                    adapt_loss += self._compute_loss(trajectory, returns, task_policy)
                
                inner_optimizer.zero_grad()
                adapt_loss.backward(retain_graph=True)
                inner_optimizer.step()
        
            post_trajectory, success, path, _ = self.collect_trajectory(maze, task_policy)
            if success:
                successful_adaptations += 1
                if len(self.replay_buffer) < self.buffer_size:
                    self.replay_buffer.append((maze, [post_trajectory]))
            
            returns = self._compute_returns([t[2] for t in post_trajectory], gamma=0.98)
            task_meta_loss = self._compute_loss(post_trajectory, returns, task_policy)
            meta_loss += task_meta_loss
        
        meta_loss /= len(task_trajectories)
        
        #update base policy
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.meta_optimizer.step()
        
        return meta_loss.item(), successful_adaptations

    def _compute_returns(self, rewards, gamma=0.95):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        if len(returns) > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def _compute_loss(self, trajectory, returns, policy):
        loss = 0
        for (state, action, _), R in zip(trajectory, returns):
            probs = policy(state)
            log_prob = torch.log(probs[action] + 1e-8)
            loss -= log_prob * R
        return loss


    #solves the maze with a cool visualization
    def solve_maze(self, maze, policy=None, visualize=True):
        if policy is None:
            policy = self.policy
        
        start_pos = (1, 1)
        goal_pos = (self.maze_size-2, self.maze_size-2)
        current_pos = start_pos
        path = [current_pos]
        visited = {current_pos: 1}
        
        if visualize:
            plt.ion()
            if len(plt.get_fignums()) == 0:
                plt.figure(figsize=(15, 8))
        
        max_steps = self.maze_size * 3
        
        try:
            for step in range(max_steps):
                state = self.get_state(maze, current_pos)
                
                with torch.no_grad():
                    probs = policy(state)
                    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    valid_moves = []
                    valid_probs = []
                    
                    #gets all valid moves and their probabilities
                    for idx, (dx, dy) in enumerate(moves):
                        new_x, new_y = current_pos[0] + dx, current_pos[1] + dy
                        if (0 <= new_x < self.maze_size and 
                            0 <= new_y < self.maze_size and 
                            maze[new_x, new_y] != 1):
                            valid_moves.append((dx, dy))
                            valid_probs.append(max(1e-10, probs[idx].item()))
                    
                    if not valid_moves:
                        break
                    
                    valid_probs = np.array(valid_probs)
                    prob_sum = valid_probs.sum()
                    if prob_sum > 0:
                        valid_probs = valid_probs / prob_sum
                    else:
                        valid_probs = np.ones_like(valid_probs) / len(valid_probs)
                    
                    valid_probs = np.nan_to_num(valid_probs, nan=1.0/len(valid_probs))
                    valid_probs = valid_probs / valid_probs.sum()
                    
                    try:
                        move_idx = np.random.choice(len(valid_moves), p=valid_probs)
                        dx, dy = valid_moves[move_idx]
                    except ValueError as e:
                        print(f"Probability error: {valid_probs}")
                        # Fall back to random choice if probabilities are invalid
                        move_idx = random.randrange(len(valid_moves))
                        dx, dy = valid_moves[move_idx]
                    
                    new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    current_pos = new_pos
                    path.append(current_pos)
                    visited[current_pos] = visited.get(current_pos, 0) + 1
                    
                    if visualize:
                        info = {
                            'step': step + 1,
                            'raw_probs': probs.numpy(),
                            'masked_probs': valid_probs,
                            'chosen_action': moves.index((dx, dy)),
                            'valid_moves': valid_moves,
                        }
                        self.visualize_path_with_info(maze, path, visited, False, info, policy)
                    
                    if current_pos == goal_pos:
                        if visualize:
                            info['success'] = True
                            self.visualize_path_with_info(maze, path, visited, True, info, policy)
                        print(f"Goal reached in {step+1} steps!")
                        return path, True
            
            if visualize:
                info = {
                    'step': max_steps,
                    'raw_probs': probs.numpy(),
                    'masked_probs': valid_probs,
                    'chosen_action': moves.index((dx, dy)),
                    'valid_moves': valid_moves,
                    'success': False
                }
                self.visualize_path_with_info(maze, path, visited, False, info, policy)
            print(f"Failed to reach goal. Steps taken: {len(path)}")
            return path, False
            
        except Exception as e:
            print(f"Error during maze solving: {e}")
            return path, False

    def visualize_path_with_info(self, maze, path, visited, success, info, policy=None):
        """Simplified and robust visualization"""
        try:
            plt.clf()
            fig = plt.gcf()
            fig.set_size_inches(15, 8)
            
            ax_maze = fig.add_subplot(121)  
            ax_info = fig.add_subplot(122)  
            
        
            maze_vis = maze.copy()
            
            #create heatmap of visited positions
            max_visits = max(visited.values()) if visited else 1
            for pos, count in visited.items():
                x, y = pos
                if maze_vis[x, y] != 1:
                    intensity = count / max_visits
                    maze_vis[x, y] = 5 + min(intensity * 4, 4)
        
            for pos in path:
                x, y = pos
                if maze_vis[x, y] != 1:
                    maze_vis[x, y] = 10
            
            #woah colors
            colors = ['white', 'black', 'green', 'red', 'gray',
                    'lightpink', 'pink', 'red', 'darkred', 'purple',
                    'blue', 'yellow']
            cmap = plt.cm.colors.ListedColormap(colors)
            
            ax_maze.imshow(maze_vis, cmap=cmap)
            ax_maze.set_title("Agent Navigation")
            ax_maze.axis('off')
            
            info_text = [
                f"Status: {'Success' if success else 'In Progress'}",
                f"Path Length: {len(path)}",
                f"Unique Positions: {len(visited)}",
                f"Max Revisits: {max(visited.values())}"
            ]
            
            if isinstance(info, dict):
                if 'step' in info:
                    info_text.insert(0, f"Step: {info['step']}")
                if 'raw_probs' in info:
                    info_text.extend([
                        "\nRaw Action Probabilities:",
                        f"Up:    {info['raw_probs'][0]:.4f}",
                        f"Right: {info['raw_probs'][1]:.4f}",
                        f"Down:  {info['raw_probs'][2]:.4f}",
                        f"Left:  {info['raw_probs'][3]:.4f}",
                    ])
                if 'chosen_action' in info:
                    info_text.append(f"\nChosen: {['Up', 'Right', 'Down', 'Left'][info['chosen_action']]}")
            
            ax_info.text(0, 1, "\n".join(info_text),
                        verticalalignment='top',
                        fontfamily='monospace',
                        transform=ax_info.transAxes)
            ax_info.axis('off')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"Visualization error: {e}")


    def train(self, num_epochs=1000, tasks_per_epoch=10, trajectories_per_task=3):
        """Enhanced training loop"""
        best_success_rate = 0
        no_improvement_count=0
        success_rates = []
        losses = []
        
        self.policy.temperature = 8.0  
        
        for epoch in range(self.start_epoch, num_epochs):
            task_trajectories = []
            epoch_successes = 0
            
            #grradually reduce temperature as it trains
            if epoch > 200:
                self.policy.temperature = max(3.0, 8.0 - (epoch - 200) * 0.01)
            
            for _ in range(tasks_per_epoch):
                maze = self.generate_maze()
                trajectories = []
                
                num_trajectories = max(3, trajectories_per_task - epoch // 200)
                
                for _ in range(num_trajectories):
                    trajectory, success, path, visited = self.collect_trajectory(maze)
                    if success:
                        epoch_successes += 1
                    trajectories.append(trajectory)
                    
                task_trajectories.append((maze, trajectories))
            
            meta_loss, adaptation_successes = self.meta_update(task_trajectories)
            success_rate = adaptation_successes / tasks_per_epoch
            #CHECKPOINTS BABY
            if (epoch + 1) % 50 == 0:  
                print("\nRunning test episodes...")
                self.test_agent(num_mazes=1, visualize=True)  
                plt.close('all')

                success_rates.append(success_rate)
                losses.append(meta_loss)
            
            if (epoch + 1) % 10 == 0:
                pre_adapt_rate = epoch_successes/(tasks_per_epoch*num_trajectories)
                print(f"\nEpoch {epoch+1}")
                print(f"Meta Loss: {meta_loss:.4f}")
                print(f"Success Rate: {success_rate:.2%}")
                print(f"Pre-adaptation Success Rate: {pre_adapt_rate:.2%}")
                print(f"Temperature: {self.policy.temperature:.2f}")
                print(f"Replay Buffer Size: {len(self.replay_buffer)}")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                self.save_checkpoint(epoch + 1, meta_loss, success_rate)
                print(f"\nNew best success rate: {success_rate:.2%}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
                if (epoch + 1) % 50 == 0:
                    self.save_checkpoint(epoch + 1, meta_loss, success_rate)
                    print(f"\nPeriodic checkpoint saved at epoch {epoch + 1}")
                    


    def test_agent(self, num_mazes=5, visualize=True):
        """Test agent with robust visualization"""
        print("\nTesting on new mazes:")
        successes = 0
        total_steps = 0
        
        try:
            plt.ion()
            for i in range(num_mazes):
                plt.close('all')  
                plt.figure(figsize=(15, 8))  
                
                test_maze = self.generate_maze()
                print(f"\nMaze {i+1}:")
                
                test_policy = PolicyNetwork(input_size=self.input_size)
                test_policy.load_state_dict(self.policy.state_dict())
                
                print("Adaptation phase...")
                adapt_trajectories = []
                adapt_successes = 0
                
                for j in range(3):
                    trajectory, success, path, visited = self.collect_trajectory(test_maze)
                    if success:
                        adapt_successes += 1
                        print(f"Adaptation episode {j+1}: success")
                    else:
                        print(f"Adaptation episode {j+1}: failed")
                    adapt_trajectories.append(trajectory)
                
                print(f"Adaptation success rate: {adapt_successes/3:.2%}")
                
                optimizer = optim.SGD(test_policy.parameters(), lr=self.inner_lr)
                for trajectory in adapt_trajectories:
                    returns = self._compute_returns([t[2] for t in trajectory])
                    loss = self._compute_loss(trajectory, returns, test_policy)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                print("\ntesting adapted policy right neow")
                if visualize:
                    path, success = self.solve_maze(test_maze, test_policy, visualize=True)
                else:
                    trajectory, success, path, visited = self.collect_trajectory(test_maze, test_policy)
                
                if success:
                    successes += 1
                    total_steps += len(path)
                    print(f"easy success. Path length: {len(path)}")
                else:
                    print("Failed to reach goal")
                
                plt.pause(2)
            
            plt.ioff()
            avg_steps = total_steps / successes if successes > 0 else float('inf')
            print(f"\nTest Results:")
            print(f"Success Rate: {successes/num_mazes:.2%}")
            print(f"Average Steps (successful runs): {avg_steps:.1f}")
            
        except Exception as e:
            print(f"Error during testing: {e}")
if __name__ == "__main__":
    print("making MAML agent")
    

    maml = MazeMAML(vision_range=2, load_checkpoint=True)
    

    training_params = {
        'num_epochs': 1000,
        'tasks_per_epoch': 8,  
        'trajectories_per_task': 2
    }
    

    print("\nStarting training right neow")
    maml.train(**training_params)
    
    #testing
    print("\nTraining complete. Testing agent right neow")
    test_maze = maml.generate_maze()
    print("\nTesting on new maze:")
    visualize_maze(test_maze)
    path, success = maml.solve_maze(test_maze, visualize=True)
    print(f"Path length: {len(path)}, Success: {success}")
    plt.show()
