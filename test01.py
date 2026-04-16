import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def plot_data(data, title=None):
    # plot initial data
    plt.figure(figsize=(6,6))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6, c='purple')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def get_spiral_data(n_points=1000):
    theta = np.sqrt(np.random.rand(n_points)) * 720 * (np.pi / 180)
    r = 0.5 * theta 
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.stack([x, y], axis=1) + np.random.randn(n_points, 2) * 0.3
    data_max = np.max(np.abs(data))
    data = data / data_max
    return data.astype(np.float32)

def forward_diffusion_step(x, t, betas):
    # sample gaussian noise
    noise = torch.randn_like(x)
    # get beta for current timestep
    b_t = betas[t]
    # add noise to x (forward diffusion)
    x_t = torch.sqrt(1-b_t) * x + torch.sqrt(b_t) * noise
    
    return x_t
    
def forward_diffusion(x, diffusion_steps, betas):
    x_values = [x]
    for i in range(diffusion_steps):
        x_t = forward_diffusion_step(x, i, betas)
        x_values.append(x_t)
        x = x_t
    return x, x_values

def sample_t(x_0, t, alpha_bars):
    # sample noise
    epsilon = torch.randn_like(x_0)
    # get x_t using closed form equation
    x_t = torch.sqrt(alpha_bars[t]) * x_0 + torch.sqrt(1-alpha_bars[t]) * epsilon
    
    return x_t, epsilon

@torch.no_grad() # Disables gradient tracking inside this function.
def sample(model, x_t, alpha_t, alpha_bar_t, sigma_t, t):
    epsilon = model(x_t, t) # The model input is x_t and t and the output is the predicted noise
    if t.numel() > 1:
        is_zero_timestep = (t[0] == 0)
    else:
        is_zero_timestep = (t.item() == 0)
        
    if is_zero_timestep:
        z = torch.zeros_like(x_t)
    else:
        z = torch.randn_like(x_t)
    x_prev = 1/torch.sqrt(alpha_t) * (x_t - (1-alpha_t)/torch.sqrt(1-alpha_bar_t) * epsilon) + sigma_t * z
    
    return x_prev

@torch.no_grad()
def reverse_diffusion(model, x_t, timesteps, device):  
    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    alphas = 1-betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    batch_size = x_t.shape[0]  # Get the batch size
    
    model.eval() # Set the model to evaluation mode (disables dropout, batch norm, etc.)
    for t in range(timesteps-1, -1, -1):
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        sigma_t = torch.sqrt(betas[t]) # The standard deviation of the noise is the square root of the betas
        # Expand t to match batch size
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        x_prev = sample(model, x_t, alpha_t, alpha_bar_t, sigma_t, t_batch)
        x_t = x_prev
    return x_t

## neural network model
class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def forward(self, t):
            device = t.device
            emb = torch.zeros(t.shape[0], self.embedding_dim, device=device)
            
            for i in range(self.embedding_dim // 2):
                # We ensure the constant is on the correct device too
                const = torch.tensor(10000.0, device=device) 
                omega_i = torch.exp(-(2*i/self.embedding_dim) * torch.log(const))
                
                emb[:, 2*i] = torch.sin(omega_i * t)
                emb[:, 2*i+1] = torch.cos(omega_i * t)
            
            return emb
        
class DenoiserNetwork(nn.Module):
    def __init__(self, sample_dim=2, time_embedding_dim=64, hidden_dim=256):
        super().__init__()
        
        self.time_embedder = SinusoidalEmbedding(time_embedding_dim)
        
        # Project time embedding to hidden dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),  # SiLU (Swish) often works better than ReLU for diffusion
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(sample_dim, hidden_dim)
        
        # Main network with residual connections and normalization
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(3)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, sample_dim),
        )
        
    def forward(self, x, t):
        # x: (batch_size, 2) noisy samples
        # t: (batch_size,) timesteps
        
        # Embed time
        t_emb = self.time_embedder(t)  # (batch, time_embedding_dim)
        t_emb = self.time_mlp(t_emb)   # (batch, hidden_dim)
        
        # Project input
        h = self.input_proj(x)  # (batch, hidden_dim)
        
        # Add time embedding (additive conditioning)
        h = h + t_emb
        
        # Residual blocks
        for block in self.blocks:
            h = h + block(h)  # Residual connection
        
        # Output
        return self.output_proj(h)
    
    
class SpiralDataset(Dataset):
    def __init__(self, n_points=1000, timesteps = 200):
        self.data = get_spiral_data(n_points)
        # plot initial data
        plot_data(self.data, title="Training Data: 2D Spiral")
        self.timesteps = timesteps
        self.alpha_bars = torch.cumprod(1-torch.linspace(1e-4, 0.02, timesteps), dim=0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # sample a random timestep
        t = torch.randint(0, self.timesteps, (1,)).item()
        # get the data point
        x_0 = torch.tensor(self.data[index]).float()
        # get x_t and epsilon using the forward diffusion trick
        x_t, epsilon = sample_t(x_0, torch.tensor(t), self.alpha_bars)
        
        return x_t, t, epsilon
    


if __name__ == "__main__":
    # data = get_spiral_data()
    # plot_data(data, title="Spiral Data")   
    
    
    ### 
    # diffusion_steps = 6
    # # schedule beta values
    # betas = torch.linspace(0.0001, 0.02, diffusion_steps)

    # x_0 = torch.tensor(data).float()
    # x_noisy, x_values = forward_diffusion(x_0, diffusion_steps, betas)

    # # plot progressive noising
    # fig, axs = plt.subplots(2, 3, figsize=(12,8))
    # for i in range(diffusion_steps):
    #     ax = axs[i//3, i%3]
    #     ax.scatter(x_values[i][:, 0].numpy(), x_values[i][:, 1].numpy(), s=10, alpha=0.6, c='purple')
    #     ax.set_xlim(-1.2, 1.2)
    #     ax.set_ylim(-1.2, 1.2)
    #     ax.set_title(f"Step {i}")
    #     ax.grid(True, alpha=0.3)
    # plt.show()
    
    ### 
    # # Try playing with different time steps t
    # n_steps = 100
    # betas = torch.linspace(1e-4, 0.02, n_steps)
    # alphas = 1-betas
    # alpha_bars = torch.cumprod(alphas, dim=0)

    # t = 10
    # x_t, epsilon = sample_t(x_0, t, alpha_bars)

    # plot_data(x_t.numpy() - epsilon.numpy(), title=f"Forward Diffusion Trick at Step {t}")
    
    
    ## training
    timesteps = 300
    n_points = 2000

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Instantiate Dataset & DataLoader
    dataset = SpiralDataset(n_points=n_points, timesteps=timesteps) # Make sure the fix is in here!
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # 3. Instantiate Model
    model = DenoiserNetwork(sample_dim=2, time_embedding_dim=32, hidden_dim=256)
    model.to(device)

    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    
    num_epochs = 1000
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for x_t, t, epsilon in dataloader:
            x_t = x_t.to(device)
            t = t.to(device)
            epsilon = epsilon.to(device)
            
            pred_epsilon = model(x_t, t)
            loss = loss_fn(pred_epsilon, epsilon)
            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} loss: {avg_loss}")
        
    ### generate data
    # 1. Generate pure noise
    x_T = torch.randn(n_points, 2).to(device)

    # 2. Run the reverse process
    # This might take a few seconds as it loops 300 times
    generated_data = reverse_diffusion(model, x_T, timesteps, device=device)

    # 3. Plot the result
    generated_data = generated_data.cpu().numpy()

    plot_data(generated_data, title="Generated Data after Reverse Diffusion")