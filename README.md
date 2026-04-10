# EKF MC Sim

A 2D rigid body simulation framework for generating Monte Carlo trajectory datasets, intended for EKF (Extended Kalman Filter) testing and development.

## Running the Demo

```bash
# Run all demos, opens results in browser
python run_demo.py

# Run a specific demo
python run_demo.py circle_center
python run_demo.py circle_tangent
python run_demo.py sinusoid
python run_demo.py random_walk

# Save output as HTML files instead of opening in browser
python run_demo.py --html
python run_demo.py sinusoid --html
```

Output HTML files are saved to `./output/` when using `--html`.

## What the Sim Does

Each demo runs **2000 Monte Carlo trials** over an 8-second simulation window and generates five plots: trajectory, animation, MC path overlay, IMU measurements, and ±1σ spread bands.

### Core Classes (`sim.py`)

**`SE2`** — Vectorized SE(2) Lie group transforms. Handles rigid body poses (x, y, θ) with composition, inversion, and exp/log maps for all n trials at once.

**`NoiseConfig`** — Configures Gaussian noise for IMU (accelerometer + gyro) and GPS-style position measurements.

**`RigidBodyTrajectory`** — Container for all simulation outputs across n trials and nt timesteps: true poses, velocities, accelerations, forces/torques, and noisy IMU/position measurements.

**`RigidBodySim`** — Base simulator. Integrates the equations of motion using RK4, then injects noise into the measurements. Subclasses override `force_body_B`, `force_world_W`, and `torque` to define motion profiles.

### Motion Profiles

| Class | Description |
|---|---|
| `CircleCenterFacing` | Circular orbit, body always faces the center |
| `CircleTangentFacing` | Circular orbit, body faces the direction of travel |
| `SinusoidForward` | Forward travel along a sinusoidal path, heading tracks velocity |
| `RandomWalk` | Stochastic forces and torques at each timestep |
