import numpy as np
import collections
import time

class MockAntiGravityEnv:
    """
    Simulates a 7-DoF contact-rich pick-and-place task.
    Returns dummy visual frames (e.g., 64x64 RGB).
    """
    def __init__(self, state_dim=7, img_size=64):
        self.state_dim = state_dim
        self.img_size = img_size
        self.state = np.zeros(self.state_dim)
        self.target = np.ones(self.state_dim) * 0.5
        self.time_step = 0

    def reset(self):
        self.state = np.random.uniform(-0.1, 0.1, size=self.state_dim)
        self.time_step = 0
        return self._render()

    def step(self, action):
        # Simulated physics step
        self.state += action * 0.05
        # Add some noise (simulating contact dynamics)
        self.state += np.random.normal(0, 0.005, size=self.state_dim)
        self.time_step += 1
        
        obs = self._render()
        
        # Reward based on distance to target
        dist = np.linalg.norm(self.state - self.target)
        reward = -dist
        done = dist < 0.05 or self.time_step >= 200
        
        return obs, reward, done, {"dist": dist}

    def _render(self):
        # Mock rendering: normally this would be PyBullet/IsaacGym rendering
        # Here we just generate a dummy (3, H, W) float32 frame based on state
        frame = np.random.rand(3, self.img_size, self.img_size).astype(np.float32)
        # Embed state directly into the frame pixels for the encoder to "learn"
        frame[0, 0, :self.state_dim] = self.state
        return frame

class LatencyInjectionLayer:
    """
    Simulates cross-continental public internet latency and jitter.
    Maintains queues for upstream (action) and downstream (video) packets.
    """
    def __init__(self, env, base_latency_ms=100, jitter_ms=50, hz=30):
        self.env = env
        self.dt = 1.0 / hz
        self.base_latency_steps = int((base_latency_ms / 1000.0) / self.dt)
        self.jitter_steps = int((jitter_ms / 1000.0) / self.dt)
        
        self.action_queue = collections.deque()
        self.obs_queue = collections.deque()
        
        self.current_step = 0

    def reset(self):
        obs = self.env.reset()
        self.action_queue.clear()
        self.obs_queue.clear()
        self.current_step = 0
        
        # Prime the observation queue
        for _ in range(self.base_latency_steps):
            self.obs_queue.append(obs)
            
        return self.obs_queue.popleft()

    def step(self, action):
        self.current_step += 1
        
        # 1. Action travels UPSTREAM to Robot
        # Simulate jitter on action arrival
        delay = self.base_latency_steps + np.random.randint(-self.jitter_steps, self.jitter_steps + 1)
        delay = max(0, delay)
        
        self.action_queue.append((self.current_step + delay, action))
        
        # Process actions that have arrived at the robot
        executed_action = np.zeros_like(action)
        # Take the oldest action that has "arrived"
        if self.action_queue and self.current_step >= self.action_queue[0][0]:
            _, executed_action = self.action_queue.popleft()
            
        # 2. Step the actual robot environment
        obs, reward, done, info = self.env.step(executed_action)
        
        # 3. Video travels DOWNSTREAM to Operator
        # Simulate jitter on video arrival
        obs_delay = self.base_latency_steps + np.random.randint(-self.jitter_steps, self.jitter_steps + 1)
        obs_delay = max(0, obs_delay)
        
        self.obs_queue.append((self.current_step + obs_delay, obs))
        
        # Retrieve the visual frame that has arrived at the operator
        operator_obs = obs # Fallback
        # Sort queue by arrival time and pop if ready
        # (In a real system, out-of-order packets cause stutter. Here we enforce order.)
        ready_obs = [x for x in self.obs_queue if self.current_step >= x[0]]
        if ready_obs:
            operator_obs = ready_obs[-1][1] # Get latest available frame
            # Remove all up to this point
            while self.obs_queue and self.obs_queue[0][0] <= ready_obs[-1][0]:
                self.obs_queue.popleft()
                
        return operator_obs, reward, done, info
