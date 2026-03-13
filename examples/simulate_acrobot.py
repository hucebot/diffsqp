import torch
import numpy as np

from diffsqp.dynamics import AcrobotDynamics
from diffsqp.utils.animate import AcrobotAnimator

torch.set_default_dtype(torch.double)

# 1. Setup Parameters
nB = 2
dt = 0.001
tf = 5.0  # Reduced time for faster testing
steps = int(tf / dt)

model = AcrobotDynamics(m1=0.1, m2=0.1, l1=0.3, l2=0.3, grav=9.81)

# 2. Initial State
x = torch.tensor([[0.0, 0.0, 0.0, 0.0], [torch.pi, 0.0, 0.0, 0.1]])
u = torch.zeros((nB, 1))
u = torch.tensor([[0.0]]).repeat(nB, 1)

# 3. Storage for results
state_history = [x.clone().numpy()]
control_history = [u.clone().numpy()]
time_history = [0.0]

# 4. Simulation Loop
for i in range(steps):
    x = model.f(x, u, dt)

    state_history.append(x.clone().numpy())
    control_history.append(u.clone().numpy())
    time_history.append((i + 1) * dt)

# 5. Concatenate and Plot
states = np.array(state_history)
controls = np.array(control_history)
t = np.array(time_history)

# # 4. Animate!
anim = AcrobotAnimator(states, model.l1, model.l2, dt, nB)
anim.animate(step_size=5)
