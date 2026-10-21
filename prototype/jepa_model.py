import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class VisualEncoder(nn.Module):
    """
    Extracts abstract latent representations z_t from high-dimensional visual frames.
    Using a lightweight CNN for low-latency inference on the edge.
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class ActionConditionedPredictor(nn.Module):
    """
    Predicts future latent state \hat{z}_{t+k} given current latent z_t and 
    a sequence of planned/executed actions a_{t:t+k}.
    """
    def __init__(self, latent_dim=256, action_dim=7, hidden_dim=512):
        super().__init__()
        # Input: concatenated [z_t, aggregated_actions]
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t, actions):
        # Flatten temporal actions into a single embedding (or use an RNN/Transformer)
        # For this prototype, we assume actions are temporally aggregated/summed or passed step-by-step
        if actions.dim() == 3: # (B, K, action_dim)
            # Simplistic prototype approach: mean action over the K steps, or an RNN
            actions = actions.mean(dim=1)
            
        x = torch.cat([z_t, actions], dim=-1)
        return self.net(x)

class ActionJEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture for Teleoperation.
    Maintains a Context Encoder (online), a Target Encoder (EMA), and a Predictor.
    """
    def __init__(self, in_channels=3, latent_dim=256, action_dim=7, ema_tau=0.99):
        super().__init__()
        self.ema_tau = ema_tau
        
        self.context_encoder = VisualEncoder(in_channels, latent_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = ActionConditionedPredictor(latent_dim, action_dim)
        
        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update for the target encoder."""
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_p.data.mul_(self.ema_tau).add_(ctx_p.data, alpha=1.0 - self.ema_tau)

    def forward(self, x_t, actions, x_t_plus_k=None):
        """
        x_t: Observation at time t (B, C, H, W)
        actions: Actions applied from t to t+k (B, K, action_dim)
        x_t_plus_k: Observation at time t+k (B, C, H, W) - Ground truth (Optional, for training)
        """
        # Encode current state
        z_t = self.context_encoder(x_t)
        
        # Predict future state in latent space
        z_hat_t_plus_k = self.predictor(z_t, actions)
        
        loss = None
        target_z = None
        if x_t_plus_k is not None:
            # Generate target representation using EMA encoder (no gradients)
            with torch.no_grad():
                target_z = self.target_encoder(x_t_plus_k)
            
            # JEPA Loss: Smooth L1 in feature space
            loss = F.smooth_l1_loss(z_hat_t_plus_k, target_z)
            
        return z_hat_t_plus_k, target_z, loss
        
    def compute_anomaly_score(self, x_t, actions, x_t_plus_k):
        """
        Demonstration Quality Validator:
        Computes cosine distance between predicted intent and actual executed future.
        High distance indicates severe jitter or operator divergence.
        """
        self.eval()
        with torch.no_grad():
            z_hat, target_z, _ = self.forward(x_t, actions, x_t_plus_k)
            # Cosine distance: 1 - cosine_similarity
            cos_sim = F.cosine_similarity(z_hat, target_z, dim=-1)
            anomaly_score = 1.0 - cos_sim
        return anomaly_score

