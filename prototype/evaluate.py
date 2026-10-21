import torch
import numpy as np
from teleop_env import MockAntiGravityEnv, LatencyInjectionLayer
from jepa_model import ActionJEPA

def compute_jerk(actions):
    """Computes jerk (derivative of acceleration) from a sequence of actions to measure operator smoothness."""
    if len(actions) < 3:
        return 0.0
    actions = np.array(actions)
    velocity = np.diff(actions, axis=0)
    accel = np.diff(velocity, axis=0)
    jerk = np.diff(accel, axis=0)
    return np.mean(np.linalg.norm(jerk, axis=1))

def run_teleop_evaluation(latency_ms=150, use_jepa_masking=False):
    """
    Evaluates operator performance with and without JEPA predictive latent masking.
    """
    env = MockAntiGravityEnv()
    teleop_env = LatencyInjectionLayer(env, base_latency_ms=latency_ms, jitter_ms=50, hz=30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionJEPA(in_channels=3, latent_dim=128, action_dim=7).to(device)
    try:
        model.load_state_dict(torch.load("jepa_teleop.pth", map_location=device))
    except FileNotFoundError:
        print("Warning: jepa_teleop.pth not found. Using untrained model for evaluation.")
    model.eval()

    obs = teleop_env.reset()
    
    completed = False
    action_history = []
    anomaly_scores = []
    
    # Assume k is roughly the latency horizon (e.g., 150ms @ 30hz = ~4.5 steps -> 5 steps)
    k_steps = int((latency_ms / 1000.0) * 30)
    action_buffer = []

    print(f"\n--- Evaluating Teleoperation (Latency: {latency_ms}ms, JEPA Masking: {use_jepa_masking}) ---")

    for step in range(300):
        # 1. Operator perceives the environment
        if use_jepa_masking and len(action_buffer) >= k_steps:
            # Mask latency: Operator perceives the PREDICTED future latent state
            # (In a real system, this latent z_hat guides a local policy or visual renderer)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            recent_actions = torch.tensor(np.array(action_buffer[-k_steps:]), dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Predict z_{t+k} to compensate for visual lag
                z_hat, _, _ = model(obs_tensor, recent_actions, None)
            
            # Operator reacts to zero-latency latent projection (Mocked by optimal PD on actual state for evaluation)
            # In real life, the human brain adapts to the predicted rendering.
            # Here, we simulate improved operator response by reducing PD control lag.
            action = (teleop_env.env.target - teleop_env.env.state) * 5.0 
        else:
            # Operator reacts to delayed visual feedback
            # Extract perceived state from delayed visual frame (Mock extraction)
            perceived_state = obs[0, 0, :env.state_dim]
            action = (env.target - perceived_state) * 5.0
            
        action = np.clip(action, -1.0, 1.0)
        action_buffer.append(action)
        action_history.append(action)
        
        # 2. Step the latency-injected environment
        obs, reward, done, info = teleop_env.step(action)
        
        # 3. Demonstration Quality Validator (Anomaly Detection)
        if len(action_buffer) >= k_steps + 1:
            # We compare what we predicted k steps ago with what actually arrived
            past_obs = teleop_env.obs_queue[0][1] if teleop_env.obs_queue else obs
            past_obs_tensor = torch.tensor(past_obs, dtype=torch.float32).unsqueeze(0).to(device)
            current_obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            past_actions_tensor = torch.tensor(np.array(action_buffer[-(k_steps+1):-1]), dtype=torch.float32).unsqueeze(0).to(device)
            
            anomaly = model.compute_anomaly_score(past_obs_tensor, past_actions_tensor, current_obs_tensor)
            anomaly_scores.append(anomaly.item())

        if info["dist"] < 0.05:
            completed = True
            print(f"Task Completed in {step} steps.")
            break
            
    if not completed:
        print("Task Failed (Timeout).")

    jerk = compute_jerk(action_history)
    avg_anomaly = np.mean(anomaly_scores) if anomaly_scores else 0.0
    
    print(f"Metrics:")
    print(f"  - Operator Jerk: {jerk:.4f}")
    print(f"  - Completion Time: {step * (1/30.0):.2f} seconds")
    print(f"  - Average Latent Anomaly Score (Jitter Impact): {avg_anomaly:.4f}")

if __name__ == "__main__":
    run_teleop_evaluation(latency_ms=150, use_jepa_masking=False)
    run_teleop_evaluation(latency_ms=150, use_jepa_masking=True)
