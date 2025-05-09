# RL_triPendulum_Swingup
Reinforced learning for the swing-up problem of the tri-pendulum.

SAC (Soft-Actor Critic)[^1] is employed as the RL algorithm (Stable Baselines3 [^2]).
Reinforcement learning (RL) is conducted on the MuJoCo environment. 

![rl-video-episode-10000](https://github.com/user-attachments/assets/33ba400b-c4fd-4b84-9645-57075a8a7324)


**Communication**

<a style="text-decoration: none" href="https://twitter.com/hogelungfish" target="_blank">
    <img src="https://img.shields.io/badge/twitter-%40hogelungfish-1da1f2.svg" alt="Twitter">
</a>
<p>

## Preparation before analysis

Requierment:

Stable Baselines3=2.3.2

gymnasium=0.29.1

matplotlib=3.9.2

numpy=1.26.0

scipy=1.13.1

mujoco=3.2.4

tensorboard=2.18.0

PyTorch

しらんけど

## actions

Force to the moving base of the tri-pendulum along sliding direction from -2N to 2N.

## observations

Observations of the rotational angle between links should be limited to achieve fast convergence of reinforcement learning.
Therefore, the following definition of observations is employed;

````
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


## Rewards[^3]

````
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


編集中


### References  

[^1]: T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, ArXiv abs/1801.01290 (2018).
[^2]: A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann, Stable-baselines3: Reliable reinforcement learning implementations, Journal of Machine Learning Research 22 (268) (2021) 1–8.
[^3]: Jongchan Baek, Changhyeon Lee, Young Sam Lee,∗, Soo Jeon, Soohee Han, Reinforcement learning to achieve real-time control of triple inverted pendulum, Engineering Applications of Artificial Intelligence, Vol. 128, 2024.
