# Simulation Framework planning
This is a document to outline the framework used to simulate trajectories in SE(2) for testing of the EKF. All of the operations/classes should be vectorized to a length n so that monte carlo trials can be easily run. n is the number of monte carlo trials and nt is the number of timesteps.

Transformations in code should be notated by A_T_B: this is the transfomation from B to A. This allows cancelation to occur like A_T_C = A_T_B * B_T_C. In documentation it should be notated like: $T_{AC} = T_{AB} T_{BC}$

## Classes

### SE2
Class for storing and defining vectors of SE(2) values as well as several vectorized operations on these.
The main state or values stored here are:

(x, y, and theta)

These should be stored as individual vectors of size n where n is the number of parallel instances

Multiplication (composition) of the SE(2) operators should be done with the * operator

Operations needed to implement:
* forward multiplication
    * with * and with .apply(2nd transform)
* logarithmic map
    * returns the logarithmic map of the given transform $\xi$
* exponential map
    * given the twist $\xi$ get the transform

### RigidBodyTrajectory
Stores all of the information needed from the trajectory of a rigid body. This should be vectorizable.
* The mass and inertia values of the rigid body. (length n each)
* the noise config used
* timestamps (n, nt)
* poses (nt): array of se2 instances se2 by default contains the n
* velocity (n, nt, 2)
* acceleration (n, nt, 2)
* force (n, nt, 2)
* angular_velocity (n, nt)
* angular_acceleration (n, nt)
* torque (n, nt)
* accel_meas (n, nt, 2): the noised values of the acceleration
* angular_velocity_meas (n, nt): the noised values of the angular velocity
* pos_meas (n, nt, 2): the position measured with noise

### NoiseConfig
Contains the information about the noise in the simulation. For now this is just in the measurements however eventually we may add process noise to the sim.

* IMU cov: 3x3 cov matrix for IMU measurements (accel and angular vel)
* pos_cov: 2x2 cov matrix for position measurement

there should also be an optional seed for deterministic reruns

### RigidBodySim
Base class representing the rigid body simulation. Should by default take in arguments for mass and moment of inertia. It should also take in a noise config and a dt value. The following functions should be deffined but not implemented (will be implemented for child classes):
* force_body: callback to get force in the body frame
* force_world: callback to get force in the world frame
* torque_body: callback to get force in the body frame
* torque_world: callback to get force in the world frame

There should be a simulate function that when called will run the vectorized simulation it will fully populate the RigidBodyTrajectory class/datastructure. It will start by simulating the loop forward with rk4. it will then add noise based on the noise config and then populate the measurements in the trajectory.


Forward step math:
Form the state vector of a concatenation of the pose variables and the rate variables
$[x, y, \theta, \dot x, \dot y, \dot \theta]$. Feed this to the force and torque callbacks. Get state derivative $[\dot x, \dot y, \dot \theta, \ddot x, \ddot y, \ddot \theta]$. Step the state forward like this with rk4.


Make the following implementations by defining the force callbacks.

Implementations:
* center facing circle
* circle facing direction of travel
* forward going sinusoid facing the direction of travel
* pure random walk with some randomized force and torque at each timestep
