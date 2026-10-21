import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from jepa_model import ActionJEPA
from teleop_env import MockAntiGravityEnv

def generate_teleop_dataset(num_episodes=50, max_steps=100, k_predict=5):
    """
    Generates offline teleoperation data to train the JEPA model.
    Records (x_t, a_{t:t+k}, x_{t+k}).
    """
    print("Generating offline demonstration dataset...")
    env = MockAntiGravityEnv()
    
    x_t_list, a_list, x_t_k_list = [], [], []
    
    for _ in range(num_episodes):
        obs = env.reset()
        episode_obs = [obs]
        episode_actions = []
        
        for _ in range(max_steps):
            # PD controller acting as a human operator (moving towards target)
            action = (env.target - env.state) * 5.0
            action = np.clip(action, -1.0, 1.0)
            
            obs, _, _, _ = env.step(action)
            episode_obs.append(obs)
            episode_actions.append(action)
            
        # Compile valid (t, t+k) pairs
        for t in range(max_steps - k_predict):
            x_t_list.append(episode_obs[t])
            # Aggregate actions over the prediction horizon k
            a_seq = episode_actions[t:t+k_predict]
            a_list.append(a_seq)
            x_t_k_list.append(episode_obs[t + k_predict])
            
    x_t_tensor = torch.tensor(np.array(x_t_list), dtype=torch.float32)
    a_tensor = torch.tensor(np.array(a_list), dtype=torch.float32)
    x_t_k_tensor = torch.tensor(np.array(x_t_k_list), dtype=torch.float32)
    
    return TensorDataset(x_t_tensor, a_tensor, x_t_k_tensor)

def train_jepa():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training JEPA on {device}...")
    
    dataset = generate_teleop_dataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = ActionJEPA(in_channels=3, latent_dim=128, action_dim=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for x_t, actions, x_t_k in dataloader:
            x_t, actions, x_t_k = x_t.to(device), actions.to(device), x_t_k.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: JEPA handles encoding, prediction, and target encoding internally
            z_hat, z_target, loss = model(x_t, actions, x_t_k)
            
            loss.backward()
            optimizer.step()
            
            # Crucial: Update EMA target encoder
            model.update_target_encoder()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | JEPA Smooth-L1 Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), "jepa_teleop.pth")
    print("Saved JEPA model to jepa_teleop.pth")

if __name__ == "__main__":
    train_jepa()
