import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import torch.nn.functional as F
from collections import deque
MAZE_SIZE = 11
BATCH_SIZE = 32
EPOCHS = 400
NOISE_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = 'checkpoints'
from MAML4 import MazeMAML
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.name = 'generator'
        
        #initial size needs to be adjusted to reach 11x11
        self.model = nn.Sequential(
            #start with 3x3
            nn.Linear(NOISE_DIM, 3*3*256),
            nn.BatchNorm1d(3*3*256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Unflatten(1, (256, 3, 3)),
            
            #3x3 -> 6x6
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            #6x6 -> 12x12
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 12x12 -> 11x11 (using conv layer to reduce size)
            nn.Conv2d(64, 1, 2, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.name = 'discriminator'
        self.model = nn.Sequential(
            # 11x11 -> 5x5
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # 5x5 -> 2x2
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 1)
        )

    def forward(self, x):
        return self.model(x)

#generates maze dataset for pretrianing 
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
        
        current_cell = stack[-1]
        visited.add(current_cell)
        maze[current_cell] = 0  
        

        neighbours = []
        for direction in directions:
            next_cell = (current_cell[0] + direction[0], current_cell[1] + direction[1])
            if 0 <= next_cell[0] < height and 0 <= next_cell[1] < width:
                neighbours.append(next_cell)
        
        unvisited_neighbours = [n for n in neighbours if n not in visited]
        
        if unvisited_neighbours:

            next_cell = unvisited_neighbours[np.random.randint(0, len(unvisited_neighbours))]

            maze[(current_cell[0] + next_cell[0]) // 2, (current_cell[1] + next_cell[1]) // 2] = 0
            stack.append(next_cell)
        else:

            stack.pop()
            
            #makes loops and stuff for decision making
            if np.random.random() < loop_probability:
                for neighbour in neighbours:
                    if maze[neighbour] == 0 and neighbour not in stack:
                        maze[(current_cell[0] + neighbour[0]) // 2, (current_cell[1] + neighbour[1]) // 2] = 0
                        if multiple_paths:
                            stack.append(neighbour) 
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


def visualize_maze(maze):
    vis_maze = np.zeros(maze.shape, dtype=str)
    vis_maze[maze == 0] = '_'  # Empty space
    vis_maze[maze == 1] = '██'  # Wall
    vis_maze[maze == 4] = '░░'  # Slow tile
    vis_maze[maze == 2] = 'S '  # Start
    vis_maze[maze == 3] = 'E '  # End
    for row in vis_maze:
        print(''.join(row))



def create_dataset(num_mazes=1000):
    mazes = []
    for _ in range(num_mazes):
        maze = generate_complex_maze(MAZE_SIZE, MAZE_SIZE)
        maze[maze > 1] = 0
        mazes.append(maze)
    return np.array(mazes)


def normalize_maze(maze):
    return (maze.astype('float32') - 2) / 2


def denormalize_maze(maze):
    return np.round((maze * 2) + 2).astype('int32')
class MAMLAwareGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, generator, discriminator, real_mazes, noise, 
                maml_performance, current_difficulty):
        batch_size = real_mazes.size(0)
        

        fake_mazes = generator(noise)
        
        #GAN losses
        d_real = discriminator(real_mazes)
        d_fake = discriminator(fake_mazes.detach())
        
        #generator loss
        g_loss = self.base_loss(d_fake, torch.ones_like(d_fake))
        

        solvability_loss = self.compute_solvability_loss(fake_mazes)
        

        difficulty_loss = self.compute_difficulty_loss(fake_mazes, current_difficulty)
        


        if maml_performance < 0.4: 
            total_loss = g_loss + 0.8 * solvability_loss + 0.05 * difficulty_loss
        elif maml_performance > 0.6: 
            total_loss = g_loss + 0.4 * solvability_loss + 0.2 * difficulty_loss
        else:  
            total_loss = g_loss + 0.6 * solvability_loss + 0.1 * difficulty_loss
        
        return total_loss

    def compute_solvability_loss(self, mazes):
        mazes_np = mazes.detach().cpu().numpy()
        batch_size = mazes_np.shape[0]
        
        total_penalty = 0
        for i in range(batch_size):
            maze = mazes_np[i, 0]
            wall_density = np.mean(maze == 1)
            density_penalty = float(wall_density > 0.5) 
            paths = self.check_path_quality(maze)
            
            total_penalty += density_penalty + paths
        
        return torch.tensor(total_penalty / batch_size, device=mazes.device)
    
    def check_path_quality(self, maze):
        visited = np.zeros_like(maze, dtype=bool)
        narrow_paths = 0
        stack = [(1, 1)]
        
        while stack:
            x, y = stack.pop()
            if not visited[x, y]:
                visited[x, y] = True
                #check for narrow corridors
                walls_around = 0
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    next_x, next_y = x + dx, y + dy
                    if (0 <= next_x < maze.shape[0] and 
                        0 <= next_y < maze.shape[1]):
                        if maze[next_x, next_y] == 1:
                            walls_around += 1
                        elif not visited[next_x, next_y]:
                            stack.append((next_x, next_y))
                
                if walls_around >= 3: 
                    narrow_paths += 0.1
        
        return narrow_paths
    def compute_difficulty_loss(self, mazes, target_difficulty):
        #simple difficulty estimation using wall density
        maze_np = mazes.detach().cpu().numpy()
        current_difficulty = torch.tensor(np.mean(maze_np == 1), 
                                        device=mazes.device)
        
        return F.mse_loss(current_difficulty, 
                         torch.tensor(target_difficulty, device=mazes.device))
class MAMLPerformanceTracker:
    def __init__(self):
        self.success_rates = []
        self.path_lengths = []
        
    def update(self, trajectory_info):
        success = trajectory_info['success']
        path_length = len(trajectory_info['path'])
        
        self.success_rates.append(float(success))
        self.path_lengths.append(path_length)
    
        if len(self.success_rates) > 100:
            self.success_rates.pop(0)
            self.path_lengths.pop(0)
            
    def get_difficulty_score(self):
        if not self.success_rates:
            return 0.5
            
        recent_success_rate = np.mean(self.success_rates[-20:])
        return recent_success_rate
class MAMLAwareGAN:
    def __init__(self, pretrained_generator=None, pretrained_discriminator=None):

        self.generator = pretrained_generator if pretrained_generator else Generator().to(DEVICE)
        self.discriminator = pretrained_discriminator if pretrained_discriminator else Discriminator().to(DEVICE)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        

        self.base_criterion = nn.BCEWithLogitsLoss()

        self.maml_criterion = MAMLAwareGANLoss()
        
        self.performance_tracker = MAMLPerformanceTracker()
        self.current_difficulty = 0.2  
        self.min_difficulty = 0.1
        self.max_difficulty = 0.5  
        self.phase2_checkpoint_path = 'checkpoints/gan_phase2.pth'
        self.best_performance = 0.0
        self.start_epoch = 0
        self.total_meta_epochs = 0 
        self.maml_checkpoint_path = 'checkpoints/maml2gan_vision2.pth'
    def save_phase2_checkpoint(self, epoch, maml_agent, performance_score):
        """Save checkpoint for phase 2 training"""
        checkpoint_data = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'performance_score': performance_score,
            'difficulty': self.current_difficulty
        }
        torch.save(checkpoint_data, self.phase2_checkpoint_path)
        
        if performance_score > self.best_performance:
            self.best_performance = performance_score
            best_path = self.phase2_checkpoint_path.replace('.pth', '_best.pth')
            torch.save(checkpoint_data, best_path)
            print(f"\nNew best GAN model saved! Performance: {performance_score:.2%}")
    def load_phase2_checkpoint(self):
        """Load phase 2 checkpoint"""
        try:
            if os.path.exists(self.phase2_checkpoint_path):
                checkpoint = torch.load(self.phase2_checkpoint_path)
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                self.current_difficulty = checkpoint['difficulty']
                self.best_performance = checkpoint.get('performance_score', 0.0)
                self.start_epoch = checkpoint['epoch']
                print(f"Loaded phase 2 checkpoint from epoch {checkpoint['epoch']}")
                print(f"Previous performance: {checkpoint['performance_score']:.2%}")
                return True
        except Exception as e:
            print(f"Error loading phase 2 checkpoint: {e}")
        return False

    def pretrain(self, num_epochs=200):
        """Phase 1: Standard GAN training on maze generation"""
        print("Starting GAN pretraining...")
        
        for epoch in range(num_epochs):
            mazes = create_dataset(1000)
            dataset = torch.utils.data.DataLoader(
                torch.from_numpy(normalize_maze(mazes)).unsqueeze(1),
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            
            for batch in dataset:
                real_images = batch.to(DEVICE)
                g_loss, d_loss = self.train_step_phase1(real_images)
            
            if (epoch + 1) % 1 == 0:
                print(f"Pretrain Epoch {epoch+1}/{num_epochs}")
                print(f"G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
                self.generate_sample_maze(epoch)  # Visualize progress
                
        print("Pretraining complete! Saving checkpoint...")
        self.save_pretrained_models()
        
    def train_step_phase1(self, real_images):
        """Standard GAN training step"""
        batch_size = real_images.size(0)
        real_label = torch.ones(batch_size, 1, device=DEVICE)
        fake_label = torch.zeros(batch_size, 1, device=DEVICE)
        
        #train discriminator
        self.d_optimizer.zero_grad()
        real_output = self.discriminator(real_images)
        d_loss_real = self.base_criterion(real_output, real_label)
        
        noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
        fake_images = self.generator(noise)
        fake_output = self.discriminator(fake_images.detach())
        d_loss_fake = self.base_criterion(fake_output, fake_label)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        #train generator
        self.g_optimizer.zero_grad()
        fake_output = self.discriminator(fake_images)
        g_loss = self.base_criterion(fake_output, real_label)
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item()
    def train_maml_epoch(self, maml_agent, params):
        """Single epoch of MAML training"""
        epoch_successes = 0
        task_trajectories = []
        
        for _ in range(params['tasks_per_epoch']):
            maze = self.generate_maze()
            trajectories = []
            
            #collect trajectories
            for _ in range(params['trajectories_per_task']):
                trajectory, success, path, visited = maml_agent.collect_trajectory(maze)
                if success:
                    epoch_successes += 1
                trajectories.append(trajectory)
            
            task_trajectories.append((maze, trajectories))
        
        #boom meta update
        meta_loss, _ = maml_agent.meta_update(task_trajectories)
        
        return epoch_successes

    def adapt_to_maml(self, maml_agent, num_epochs=400, max_meta_epochs=1000):
        """Improved MAML-aware training with epoch limit"""
        print("Starting MAML-aware training...")
        self.load_phase2_checkpoint()
        maml_training_params = {
            'num_epochs': 15,
            'tasks_per_epoch': 6,
            'trajectories_per_task': 4
        }
        
        running_success_rate = 0.0
        running_steps = []
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nPhase 2 - Epoch {epoch+1}/{num_epochs}")
            print(f"Total Meta-Learning Epochs: {self.total_meta_epochs}/{max_meta_epochs}")
            
            if self.total_meta_epochs >= max_meta_epochs:
                print(f"\nReached maximum meta-learning epochs ({max_meta_epochs})")
                print("Saving final checkpoint and stopping...")
                self.save_phase2_checkpoint(epoch + 1, maml_agent, running_success_rate)
                break
        
            epoch_successes = 0
            epoch_steps = []
            
            for maml_epoch in range(maml_training_params['num_epochs']):
                if self.total_meta_epochs >= max_meta_epochs:
                    break
                
                task_trajectories = []
                
                #generate curriculum of mazes
                for _ in range(maml_training_params['tasks_per_epoch']):
                    if np.random.random() < 0.3:
                        old_diff = self.current_difficulty
                        self.current_difficulty = max(self.min_difficulty, 
                                                    self.current_difficulty - 0.1)
                        maze = self.generate_maze()
                        self.current_difficulty = old_diff
                    else:
                        maze = self.generate_maze()
                    
                    trajectories = []
                    task_success = 0
                    
                    for _ in range(maml_training_params['trajectories_per_task']):
                        trajectory, success, path, visited = maml_agent.collect_trajectory(maze)
                        if success:
                            task_success += 1
                            epoch_steps.append(len(path))
                        trajectories.append(trajectory)
                    
                    epoch_successes += task_success
                    task_trajectories.append((maze, trajectories))
                
                #boom more meta loss
                meta_loss, adaptation_successes = maml_agent.meta_update(task_trajectories)
                self.total_meta_epochs += 1 
                #checkpoints!
                if (maml_epoch + 1) % 5 == 0:
                    print(f"MAML Epoch {maml_epoch+1}")
                    print(f"Meta Loss: {meta_loss:.4f}")
                    print(f"Total Meta-Learning Epochs: {self.total_meta_epochs}")
                    

                    if self.total_meta_epochs % 50 == 0:
                        print("Saving periodic checkpoint...")
                        self.save_phase2_checkpoint(epoch + 1, maml_agent, running_success_rate)
                        maml_agent.save_checkpoint(
                            epoch=self.total_meta_epochs,
                            meta_loss=meta_loss,
                            success_rate=running_success_rate
                        )
            

            if self.total_meta_epochs >= max_meta_epochs:
                print(f"\nReached maximum meta-learning epochs ({max_meta_epochs})")
                print("Saving final checkpoint and stopping...")
                self.save_phase2_checkpoint(epoch + 1, maml_agent, running_success_rate)
                break
            

            num_trials = (maml_training_params['tasks_per_epoch'] * 
                         maml_training_params['trajectories_per_task'])
            success_rate = epoch_successes / num_trials
            

            running_success_rate = 0.9 * running_success_rate + 0.1 * success_rate
            running_steps.extend(epoch_steps)
            if len(running_steps) > 100:
                running_steps = running_steps[-100:]
            
            #difficulty adjustment
            if running_success_rate > 0.6:
                self.current_difficulty = min(self.current_difficulty + 0.02, 
                                           self.max_difficulty)
            elif running_success_rate < 0.4:
                self.current_difficulty = max(self.current_difficulty - 0.03, 
                                           self.min_difficulty)
            
            #generate validation mazes and update GAN
            print("\nValidating and updating GAN...")
            validation_mazes = create_dataset(50)
            dataset = torch.utils.data.DataLoader(
                torch.from_numpy(normalize_maze(validation_mazes)).unsqueeze(1),
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            
            total_loss = 0
            num_batches = 0
            for batch in dataset:
                real_images = batch.to(DEVICE)
                loss = self.train_step_phase2(real_images)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            

            print("\nEpoch Summary:")
            print(f"Current Difficulty: {self.current_difficulty:.2f}")
            print(f"Success Rate: {success_rate:.2%}")
            print(f"Running Success Rate: {running_success_rate:.2%}")
            print(f"Average Steps: {np.mean(running_steps):.1f}")
            print(f"GAN Loss: {avg_loss:.4f}")
            print(f"Total Meta-Learning Epochs: {self.total_meta_epochs}")
            

            self.save_phase2_checkpoint(epoch + 1, maml_agent, running_success_rate)
            
            if (epoch + 1) % 5 == 0:
                self.generate_sample_maze(epoch)
                maml_agent.test_agent(num_mazes=1, visualize=True)
                plt.pause(1)
                plt.close('all')
        
    def train_step_phase2(self, real_images):
        """MAML-aware training step"""
        batch_size = real_images.size(0)
        
        #generate noise with difficulty encoding
        noise = self.generate_controlled_noise()
        

        performance_score = self.performance_tracker.get_difficulty_score()  # Add this line
        
        #MAML-aware loss computation
        total_loss = self.maml_criterion(
            self.generator,
            self.discriminator,
            real_images,
            noise,
            performance_score,  
            self.current_difficulty
        )
        
        #update the generator with MAML awareness
        self.g_optimizer.zero_grad()
        total_loss.backward()
        self.g_optimizer.step()
        
        return total_loss.item()
    
    def generate_controlled_noise(self):
        """Generate noise with embedded difficulty information"""
        batch_size = BATCH_SIZE
        base_noise = torch.randn(batch_size, NOISE_DIM-1, device=DEVICE)
        difficulty_channel = torch.full((batch_size, 1), self.current_difficulty, device=DEVICE)
        return torch.cat([base_noise, difficulty_channel], dim=1)
    def generate_maze(self):
        """Generate a maze with current difficulty"""
        self.generator.eval()
        with torch.no_grad():
            noise = self.generate_controlled_noise()
            generated_image = self.generator(noise[0].unsqueeze(0)).cpu().squeeze().numpy()
            maze = denormalize_maze(generated_image)
            maze[1, 1] = 2  
            maze[MAZE_SIZE-2, MAZE_SIZE-2] = 3  
        self.generator.train()
        return maze
    def generate_sample_maze(self, epoch):
        """Generate and visualize a sample maze"""
        self.generator.eval()
        with torch.no_grad():
            noise = self.generate_controlled_noise()
            generated_image = self.generator(noise[0].unsqueeze(0)).cpu().squeeze().numpy()
            maze = denormalize_maze(generated_image)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(maze, cmap='viridis')
            plt.axis('off')
            plt.title(f"Generated Maze (Difficulty: {self.current_difficulty:.2f})")
            plt.savefig(f'maze_sample_epoch_{epoch}.png')
            plt.close()
        self.generator.train()
    
    def save_pretrained_models(self):
        """Save pretrained models"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, 'pretrained_gan.pth')


if __name__ == "__main__":

    gan = MAMLAwareGAN()
    

    maml_agent = MazeMAML(
        maze_size=11, 
        vision_range=2, 
        load_checkpoint=True  
    )
    
    print(f"\nMAML Agent starting from epoch: {maml_agent.start_epoch}")
    print(f"Current policy temperature: {maml_agent.policy.temperature}")
    print(f"Inner learning rate: {maml_agent.inner_lr}")
    
    if os.path.exists('pretrained_gan.pth'):
        print("Loading pretrained GAN model")
        checkpoint = torch.load('pretrained_gan.pth')
        gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        print("Pretrained GAN model loaded successfully")
        

        gan.total_meta_epochs = maml_agent.start_epoch
        print(f"Synced GAN total_meta_epochs to: {gan.total_meta_epochs}")
        
        #phase 2: MAML-aware training
        gan.adapt_to_maml(maml_agent, num_epochs=300)
    else:
        print("No pretrained GAN model found. Starting pretraining...")
        #phase 1: Pretrain GAN
        gan.pretrain(num_epochs=200)
        
        #phase 2: MAML-aware training
        gan.adapt_to_maml(maml_agent, num_epochs=300)