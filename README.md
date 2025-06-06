# RL_triPendulum_Swingup

![License](https://img.shields.io/github/license/yuki-koyama/elasty)
<img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat">


**Communication**

<a style="text-decoration: none" href="https://twitter.com/hogelungfish" target="_blank">
    <img src="https://img.shields.io/badge/twitter-%40hogelungfish-1da1f2.svg" alt="Twitter">
</a>
<p>

Reinforced learning for the swing-up problem of the tri-pendulum.

SAC (Soft-Actor Critic)[^1] is employed as the RL algorithm ([Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) [^2]).
Reinforcement learning (RL) is conducted on the [MuJoCo environment](https://mujoco.readthedocs.io/en/stable/overview.html). 

![rl-video-episode-10000](https://github.com/user-attachments/assets/33ba400b-c4fd-4b84-9645-57075a8a7324)




## Preparation before analysis

<details><summary><b>Show instructions</b></summary>

Requirement:

* __python=3.11__

* For Reinforcement learning algorithm: __Stable Baselines3=2.3.2__

* For simulation environment: __gymnasium=0.29.1__

* For plot: __matplotlib=3.9.2__

* For matrix calculation: __numpy=1.26.0__

* For the Matlab file (*.mat) generation: __scipy=1.13.1__

* For simulation environment: __mujoco=3.2.4__

* For data logging: __tensorboard=2.18.0__

* For GPGPU: __PyTorch__

* others

```batch
conda create -n py311mujoco python=3.11
conda activate py311mujoco
conda install numpy=1.26.0

# install PyTorch

pip install stable-baselines3
pip install gymnasium
pip install mujoco
pip install tensorboard

# install others

git clone https://github.com/KRproject-tech/RL_triPendulum_Swingup
cd RL_triPendulum_Swingup
python exe.py

# wait until 5*1e+6 steps
```
To confirm the learning process, the following command is used in the another terminal;

```batch
cd RL_triPendulum_Swingup
tensorboard --logdir logs/SAC_1
```

After that, we can confirm the learning process in the web browser at `http://localhost:6006/`.

<img width="941" alt="image" src="https://github.com/user-attachments/assets/ce9bb3a0-f932-4cfb-bb5f-9be683aebe23" />

The convergence of reinforcement learning depends on the initial value.
Therefore, __some trials of reinforcement learning may be necessary to achieve the inverted situation.__

</details>



## Definitions for the dynamics of the pendulum

The dynamics of the pendulum are defined in the MJCF (MuJoCo Format) file named `inverted_tri_pendulum.xml`.

<details><summary><b>Show inverted_tri_pendulum.xml</b></summary>
    
````xml
<!-- Cartpole Model

    The state space is populated with joints in the order that they are
    defined in this file. The actuators also operate on joints.

    State-Space (name/joint/parameter):
        - cart      slider      position (m)
        - pole      hinge       angle (rad)
        - cart      slider      velocity (m/s)
        - pole      hinge       angular velocity (rad/s)

    Actuators (name/actuator/parameter):
        - cart      motor       force x (N)

-->
<mujoco model="cartpole">
  <compiler coordinate="local" inertiafromgeom="true"/>
  <custom>
    <numeric data="2" name="frame_skip"/>
  </custom>
  <default>
    <joint damping="0.01"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="1e-5 0 -9.81" integrator="RK4" timestep="0.01"/>
  <size nstack="3000"/>
  <worldbody>
    <geom name="floor" pos="0 0 -3.0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 2" type="capsule"/>
    <body name="cart" pos="0 0 0">
      <joint axis="1 0 0" limited="true" margin="0.01" name="slider" pos="0 0 0" range="-2 2" type="slide"/>
      <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
      <body name="pole" pos="0 0 0">
        <joint axis="0 1 0" name="hinge" pos="0 0 0" type="hinge"/>
        <geom fromto="0 0 0 0 0 -0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
        <body name="pole2" pos="0 0 -0.6">
          <joint axis="0 1 0" name="hinge2" pos="0 0 0" type="hinge"/>
          <geom fromto="0 0 0 0 0 -0.6" name="cpole2" rgba="0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
          <body name="pole3" pos="0 0 -0.6">
            <joint axis="0 1 0" name="hinge3" pos="0 0 0" type="hinge"/>
            <geom fromto="0 0 0 0 0 -0.6" name="cpole3" rgba="0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
            <site name="tip" pos="0 0 -0.6" size="0.01 0.01"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-2 2" gear="500" joint="slider" name="slide"/>
  </actuator>
</mujoco>
````
        
</details>



## actions

Force to the moving base of the tri-pendulum along sliding direction from -2N to 2N, namely;

$$
a := f_x.
$$

## observations

Observations are defined in `inverted_tri_pendulum_swingup.py`.

Observations of the rotational angle between links should be limited to achieve fast convergence of reinforcement learning.
Therefore, the following definition of observations is employed;

````python
def _get_obs(self):
    return np.concatenate(
        [
            self.data.qpos[:1],                                     # cart x pos [m]
            np.sin( self.data.qpos[1:]),                            # link angles [rad]   
            np.cos( self.data.qpos[1:]),                            # link angles [rad]    
            np.clip( self.data.qvel[:1], -10, 10),                  # cart x pos vel [m/s]  
            np.clip( self.data.qvel[1:], -10*np.pi, 10*np.pi),      # link angles vel [rad/s]      
        ]
    ).ravel()
````
Namely,

$$
\bf{o} := 
\left[
\begin{array}{c}
x\\
\sin(\theta_1)\\
\sin(\theta_2)\\
\sin(\theta_3)\\
\cos(\theta_1)\\
\cos(\theta_2)\\
\cos(\theta_3)\\
\rm{clip}(v, -10, 10)\\
\rm{clip}(\omega_1, -10\pi, 10\pi)\\
\rm{clip}(\omega_2, -10\pi, 10\pi)\\
\rm{clip}(\omega_3, -10\pi, 10\pi)\\
\end{array}
\right]
$$

## Rewards[^3]

Rewards are defined in `inverted_tri_pendulum_swingup.py`.

````python
J_xpos = np.exp( -(x_pos/2.0)**2 )

# theta1 = 0 -> below position
J_theta1 = ( 1 + np.cos( theta1 - PI ) )/2.0
J_theta2 = ( 1 + np.cos( theta2 ) )/2.0
J_theta3 = ( 1 + np.cos( theta3 ) )/2.0

J_omega1 = np.exp( -5.0*( omega1/(2*PI) )**2)
J_omega2 = np.exp( -1.0*( omega2/(2*PI) )**2)
J_omega3 = np.exp( -1.0*( omega3/(2*PI) )**2)

r =  J_xpos*J_theta1*J_theta2*J_theta3*np.amin([ J_omega1, J_omega2, J_omega3 ])
````

Namely, 

$$
r := J_x \cdot J_{\theta_1} \cdot J_{\theta_2} \cdot J_{\theta_3} \min \\{ J_{\omega_1}, J_{\omega_2}, J_{\omega_3} \\},
$$

where,

```math
\begin{eqnarray}
J_x &:=& \exp \left( -\left(\frac{x}{2}\right)^2 \right), \\
J_{\theta_1} &:=& \frac{1 + \cos(\theta_1 - \pi)}{2},\\
J_{\theta_2} &:=& \frac{1 + \cos(\theta_2)}{2},\\
J_{\theta_3} &:=& \frac{1 + \cos(\theta_3)}{2},\\
J_{\omega_1} &:=& \exp \left( -5\left(\frac{\omega_1}{2\pi}\right)^2 \right),\\
J_{\omega_2} &:=& \exp \left( -\left(\frac{\omega_2}{2\pi}\right)^2 \right),\\
J_{\omega_3} &:=& \exp \left( -\left(\frac{\omega_3}{2\pi}\right)^2 \right).
\end{eqnarray}
```

## Garallys

<img width="1620" alt="image" src="https://github.com/user-attachments/assets/6140a730-db76-4bb8-84da-ccafa6e4d44c" />

<img width="1593" alt="image" src="https://github.com/user-attachments/assets/cab653a3-5cab-40fd-9d96-5273f84f3e90" />


### References  

[^1]: T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, ArXiv abs/1801.01290 (2018).
[^2]: A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann, Stable-baselines3: Reliable reinforcement learning implementations, Journal of Machine Learning Research 22 (268) (2021) 1–8.
[^3]: Jongchan Baek, Changhyeon Lee, Young Sam Lee,∗, Soo Jeon, Soohee Han, Reinforcement learning to achieve real-time control of triple inverted pendulum, Engineering Applications of Artificial Intelligence, Vol. 128, 2024.
