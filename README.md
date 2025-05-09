# RL_triPendulum_Swingup
Reinforced learning for the swing-up problem of the tri-pendulum.

SAC is employed as the RL algorithm (Stable Baselines3).
RL is conducted on the MuJoCo environment. 

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

''''
def _get_obs(self):
    return np.concatenate(
        [
            # self.data.qpos[:1],             # cart x pos
            # np.sin(self.data.qpos[1:]),     # link angles
            # np.cos(self.data.qpos[1:]),
            # np.clip(self.data.qvel, -10, 10),
            # np.clip(self.data.qfrc_constraint, -10, 10),
            self.data.qpos[:1],                                     # cart x pos [m]
            np.sin( self.data.qpos[1:]),                            # link angles [rad]   
            np.cos( self.data.qpos[1:]),                            # link angles [rad]    
            np.clip( self.data.qvel[:1], -10, 10),                  # cart x pos vel [m/s]  
            np.clip( self.data.qvel[1:], -10*np.pi, 10*np.pi),      # link angles vel [rad/s]      
        ]
    ).ravel()
''''

## Rewards

編集中
