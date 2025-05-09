# File access
import os
import shutil
import random
import time as Time


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium import register

register(
    id='Inverted_tri_pendulum_swingup-v1',
    entry_point='inverted_tri_pendulum_swingup:InvertedTriPendulumEnv_swingup'
)

# for PPO
from stable_baselines3 import PPO
# for SAC
from stable_baselines3 import SAC
# for DDPG
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# math Library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Pytorch
import torch

######################################################

# algorithm = 'PPO'
algorithm = 'SAC'
# algorithm = 'DDPG'

GPU_use = False



# 動画を保存するディレクトリを先に定義
video_dir = './videos'


# 削除
if os.path.exists( video_dir[2:]) :
    shutil.rmtree( video_dir[2:])
os.mkdir( video_dir[2:])

# ログフォルダの生成
log_dir = './logs/'


if os.path.exists( log_dir[2:]) :
    shutil.rmtree( log_dir[2:])
os.makedirs( log_dir[2:])




########################################################
time_start = Time.time()

# 学習環境の準備
env = gym.make('Inverted_tri_pendulum_swingup-v1', render_mode="rgb_array")

# 10エピソードごとにビデオを記録するように環境をラップ
env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: x % 100 == 0)

env = make_vec_env(lambda: env)
# env = VecNormalize(env) # これがある状態で学習させると，ロード時にうまく機能しない


# モデルの準備
print("Is CUDA available?", torch.cuda.is_available())
print("Is GPGPU?", GPU_use)

if algorithm == 'PPO' :
    ### PPO

    if torch.cuda.is_available() and GPU_use :
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, device="cuda")
    else :
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, device="cpu")

    # 学習の実行
    model.learn(total_timesteps=10*128000)

elif algorithm == 'SAC' :
    ### SAC
    
    if torch.cuda.is_available() and GPU_use :
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cuda")
    else :
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu")

    model.learn(total_timesteps=500*10000, log_interval=10)
    model.save("sac_model")

    del model # remove to demonstrate saving and loading

    model = SAC.load("sac_model")

else :
    ### DDPG

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    if torch.cuda.is_available() and GPU_use :
        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=log_dir, device="cuda")
    else :
        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=log_dir, device="cpu")

    model.learn(total_timesteps=100*10000, log_interval=10)
    model.save("ddpg_model")

    del model # remove to demonstrate saving and loading

    model = DDPG.load("ddpg_model")


######################################################
# 推論の実行
h_time = []
h_rewards = []
h_state = []
h_action = []

state = env.reset()
while True:

    # time [s]
    time = env.envs[0].unwrapped.time
    print( '{:.2f}'.format(time))

    # 学習環境の描画
    env.render()

    # モデルの推論

    action, _ = model.predict(state, deterministic=True)

    # 1ステップ実行
    state, rewards, done, info = env.step(action)

    h_time.append( time)

    h_rewards.append( rewards);
    h_state.append( state);
    h_action.append( action);


    # エピソード完了
    if done:
        break

# 学習環境の解放
env.close()


########################################################
# 報酬の変化をプロット
plt.figure(figsize=(10, 5))
plt.plot( h_time, h_rewards)
plt.title('Rewards over time')
plt.xlabel('Time')
plt.ylabel('Total Reward')
plt.grid()

# PDFとして保存
with PdfPages('rewards_plot.pdf') as pdf:
    pdf.savefig()  # 現在の図をPDFに保存
    plt.close()  # 図を閉じる


# 観測値の変化をプロット
plt.figure(figsize=(10, 5))
plt.plot( h_time, np.squeeze( h_state))
plt.title('state over time')
plt.xlabel('Time')
plt.ylabel('staste')
plt.grid()


# PDFとして保存
with PdfPages('staste_plot.pdf') as pdf:
    pdf.savefig()  # 現在の図をPDFに保存
    plt.close()  # 図を閉じる


# 行動の変化をプロット
plt.figure(figsize=(10, 5))
plt.plot( h_time, np.squeeze( h_action))
plt.title('action over time')
plt.xlabel('Time')
plt.ylabel('action')
plt.grid()

# PDFとして保存
with PdfPages('actions_plot.pdf') as pdf:
    pdf.savefig()  # 現在の図をPDFに保存
    plt.close()  # 図を閉じる


print("報酬の変化を'rewards_plot.pdf'として保存しました。")
print(f"Computing Time : {Time.time()-time_start:.2f}s")
