# -*- coding: utf-8 -*-

import chainer
# v0.7.0 で動作確認
import chainerrl
from chainerrl import replay_buffer
from chainerrl import experiments
from chainerrl import links
from chainerrl import explorers
from chainerrl.q_functions import DistributionalDuelingDQN
import gym
import random
import cv2
import datetime as dt
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import numpy as np
import sqlite3 as lite
import pandas.io.sql as psql
from sklearn.model_selection import train_test_split
import logging
import sys
import os
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
# from chainerrl_visualizer import launch_visualizer

plotly.io.orca.ensure_server()

# gym の atri の画像サイズ
RAW_IMAGE_HEIGHT = 210
RAW_IMAGE_WIDTH = 160

# ChainerRL標準のサイズ
OBSERVATION_IMAGE_HEIGHT = 84
OBSERVATION_IMAGE_WIDTH = 84

MAX_TIME = 45
IMAGE_TIME_STEPS = 45
TEST_STEP = 500
STEPS = 30000
PER_FRAME = 4  # gymの仕様の4フレームを環境にする
GPU_DEVICE = -1  # 利用しない場合は -1

WRITE_OBSERVATION_IMAGE = True
WRITE_RAW_IMAGE = True

cols = ['date', 'open', 'high', 'low', 'close', 'volume']
trade_cols = ['buy_sell', 'open_price',
              'close_price', 'reward', 'holding_count']
episode_cols = ['time', 'train_profit',
                'train_win_rate', 'test_profit', 'test_win_rate']


class Position:
    def __init__(self, buy_sell, open_price):
        # 買いか売りか
        self.buy_sell = buy_sell
        # 取得価格
        self.open_price = open_price
        # 目標価格
        self.target_price = self._target_price()
        # ロスカット価格
        self.loss_cut_price = self._loss_cut_price()
        # 決済価格
        self.close_price = None
        self.reward = None
        self.holding_times = 0

    def cal_reward(self, close_price):
        self.close_price = close_price
        if (self.buy_sell == 'Buy'):
            diff = self.close_price - self.target_price
        if (self.buy_sell == 'Sell'):
            diff = self.target_price - self.close_price

        if (diff > 0):
            self.reward = 1
        else:
            self.reward = -1

    def count_up(self):
        self.holding_times += 1

    def to_pd(self):
        return pd.DataFrame(
            [[
                self.buy_sell,
                self.open_price,
                self.close_price,
                self.reward,
                self.holding_times
            ]],
            columns=trade_cols)

    def _target_price(self):
        # BitMEXのスプレッドが0.5ドルで固定
        # 手数料はTaker手数料 0.075% Maker手数料 0.05%
        # 目標は 0.1%
        if (self.buy_sell == 'Buy'):
            return self.open_price * 1.001
        if (self.buy_sell == 'Sell'):
            return self.open_price * 0.999

    def _loss_cut_price(self):
        # 0.2% 分予想から外れたらロスカット
        if (self.buy_sell == 'Buy'):
            return self.open_price * 0.998
        if (self.buy_sell == 'Sell'):
            return self.open_price * 1.002

    def is_over_loss(self, price):
        if (self.buy_sell == 'Buy'):
            return self.loss_cut_price > price
        if (self.buy_sell == 'Sell'):
            return self.loss_cut_price < price


class Trade(gym.core.Env):
    def __init__(self, df, test=False):
        self.test = test
        self.df = df.reset_index(drop=True)
        self.df_row = len(self.df)
        self.position = None
        self.start_time = self._start_time()
        self.time = 0
        self.before_action = None
        self.consecutive_times = 0
        # 0: buy, 1: sell, 2: close, 3: wait
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(OBSERVATION_IMAGE_HEIGHT, OBSERVATION_IMAGE_WIDTH),
            dtype=np.uint8)
        if (self.test):
            # テスト時には取引を記録してCSVで出力する
            self.trades_df = pd.DataFrame([], columns=trade_cols)

    def step(self, action):
        reward = 0
        done = False
        current_price = self.current_df().iloc[-1]['close']

        if (action == self.before_action):
            self.consecutive_times += 1
        else:
            self.consecutive_times = 0

        if (self.position is None):
            if (action == 0):
                self.position = Position('Buy', current_price)
            if (action == 1):
                self.position = Position('Sell', current_price)

        if (self.position is not None):
            self.position.count_up()
            if (action == 2):
                self.position.cal_reward(current_price)
                reward += self.position.reward

                if (self.test):
                    self.trades_df = self.trades_df.append(
                        self.position.to_pd(), ignore_index=True
                    )
                else:
                    done = True
                self.position = None
                if (reward > 0):
                    print('win:', self.time)
                else:
                    print('lose:', self.time)
            else:
                if (self.position.is_over_loss(current_price)):
                    reward += -2
                    if (not self.test):
                        done = True
                    self.position = None
                    print('loss cut:', self.time)

        observation = self._observation()
        info = {}
        self.time += 1
        if (self.test):
            if (self.time == TEST_STEP):
                done = True
                self.trades_df.to_csv(
                    'csv/trades' + dt.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')
                self.trades_df = pd.DataFrame([], columns=trade_cols)
        else:
            if (self.time == MAX_TIME):
                done = True
                reward -= 2
                print('over:', self.time)
        self.before_action = action
        return observation, reward, done, info

    def reset(self):
        self.time = 0
        self.start_time = self._start_time()
        self.position = None
        observation = self._observation()
        return observation

    def render(self, mode):
        return np.array(cv2.cvtColor(self.current_chart(0, raw=True), cv2.COLOR_BGR2RGB))

    def close(self):
        pass

    def seed(self):
        pass

    def _observation(self):
        if WRITE_OBSERVATION_IMAGE:
            cv2.imwrite("charts/plot-observation.png", cv2.cvtColor(
                self.current_chart(0, raw=False), cv2.COLOR_BGR2GRAY))

        observation = np.array([
            cv2.cvtColor(self.current_chart(3), cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(self.current_chart(2), cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(self.current_chart(1), cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(self.current_chart(0), cv2.COLOR_BGR2GRAY),
        ])
        return observation

    def current_df(self, delay=0):
        current_time = self.start_time + self.time - delay

        return self.df[current_time:current_time + IMAGE_TIME_STEPS]

    def current_chart(self, delay=0, raw=False):
        df = self.current_df(delay)
        fig = plotly.subplots.make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.8, 0.2],
            print_grid=False
        )
        fig.update_layout(height=RAW_IMAGE_HEIGHT, width=RAW_IMAGE_WIDTH)
        fig['layout'].update(title='')
        fig['layout']['title']['font'].update(size=8)
        fig['layout']['showlegend'] = False
        fig['layout']['xaxis']['rangeslider'].update(visible=False)
        fig['layout']['xaxis2']['rangeslider'].update(visible=False)
        fig['layout']['xaxis']['showticklabels'] = False
        fig['layout']['xaxis2']['showticklabels'] = False
        fig['layout']['yaxis']['showticklabels'] = False
        fig['layout']['yaxis2']['showticklabels'] = False
        fig['layout']['xaxis']['showgrid'] = False
        fig['layout']['xaxis2']['showgrid'] = False
        fig['layout']['yaxis']['showgrid'] = False
        fig['layout']['yaxis2']['showgrid'] = False
        fig['layout']['margin'].update(l=1)
        fig['layout']['margin'].update(r=1)
        fig['layout']['margin'].update(b=1)
        fig['layout']['margin'].update(t=1)
        fig['layout']['margin'].update(pad=0)
        fig['layout'].update(paper_bgcolor='black')
        fig['layout'].update(plot_bgcolor='black')

        candles = go.Candlestick(
            x=df.date,
            open=df.open,
            high=df.high,
            low=df.low,
            close=df.close,
            increasing=dict(line=dict(color=('#FD0000'))),
            decreasing=dict(line=dict(color=('#00FF00'))),
            name='Price'
        )

        fig.append_trace(candles, 1, 1)

        if (self.position is not None):
            if (self.position.buy_sell == 'Buy'):
                buy = go.Scatter(
                    x=df.date,
                    y=[self.position.target_price] * IMAGE_TIME_STEPS,
                    name='Buy',
                    line=dict(
                        color=('#00FFFF'),
                        width=2)
                )

                fig.append_trace(buy, 1, 1)

            else:
                sell = go.Scatter(
                    x=df.date,
                    y=[self.position.target_price] * IMAGE_TIME_STEPS,
                    name='Sell',
                    line=dict(
                        color=('#FFB8FF'),
                        width=2)
                )

                fig.append_trace(sell, 1, 1)

        volume = go.Bar(
            x=df.date,
            y=df.volume,
            name='Volume',
            marker=dict(
                color=('#FFFF00')
            )
        )

        fig.append_trace(volume, 2, 1)

        if raw:
            bytes_image = pio.to_image(
                fig,
                width=RAW_IMAGE_WIDTH,
                height=RAW_IMAGE_HEIGHT,
                format='png'
            )
        else:
            bytes_image = pio.to_image(
                fig,
                width=OBSERVATION_IMAGE_WIDTH,
                height=OBSERVATION_IMAGE_HEIGHT,
                format='png'
            )
        if WRITE_RAW_IMAGE:
            fig.write_image("charts/plot-raw.png")

        return cv2.imdecode(np.frombuffer(bytes_image, np.uint8), -1)

    # テスト時は最初から
    # トレーニング時はランダムで開始場所を決定する
    def _start_time(self):
        if (self.test):
            return PER_FRAME
        else:
            return random.randint(0, self.df_row - IMAGE_TIME_STEPS - MAX_TIME - PER_FRAME)


def make_env(process_idx=0, test=False):
    print('process:', process_idx, ', test:', test)
    con = lite.connect('btcusd.db', isolation_level=None,
                       detect_types=lite.PARSE_DECLTYPES | lite.PARSE_COLNAMES)
    df = psql.read_sql("select * from ticks order by date", con)

    train_df, test_df = train_test_split(
        df, test_size=TEST_STEP + IMAGE_TIME_STEPS + PER_FRAME, shuffle=False)
    if (test):
        env = Trade(test_df, test=True)
    else:
        env = Trade(train_df, test=False)
    return env


seed = 0
chainerrl.misc.set_random_seed(seed)

env = make_env(0, test=False)
eval_env = make_env(0, test=True)

action_size = env.action_space.n

n_atoms = 51
v_max = 10
v_min = -10

q_func = DistributionalDuelingDQN(action_size, n_atoms, v_min, v_max)

gpu_device = GPU_DEVICE
if GPU_DEVICE == 0:
    chainer.cuda.get_device(gpu_device).use()
    q_func.to_gpu(gpu_device)

links.to_factorized_noisy(q_func, sigma_scale=0.5)

explorer = explorers.Greedy()

opt = chainer.optimizers.Adam(6.25e-5, eps=1.5 * 10 ** -4)
opt.setup(q_func)

update_interval = 4

betasteps = STEPS / update_interval

rbuf = replay_buffer.PrioritizedReplayBuffer(
    10 ** 6, alpha=0.5, beta0=0.4, betasteps=betasteps,
    num_steps=3)


def phi(x):
    return np.asarray(x, dtype=np.float32) / 255


Agent = chainerrl.agents.CategoricalDoubleDQN
agent = Agent(
    q_func, opt, rbuf, gpu=gpu_device, gamma=0.99,
    explorer=explorer, minibatch_size=32,
    replay_start_size=10000,
    target_update_interval=20000,
    update_interval=update_interval,
    batch_accumulator='mean',
    phi=phi,
)

# 必要であれば agent をロードする
# agent.load(
#     'results/rainbow/50000_finish'
# )

experiments.train_agent_with_evaluation(
    outdir='results/rainbow',
    env=env,
    eval_env=eval_env,
    agent=agent,
    steps=STEPS,
    eval_n_steps=None,
    eval_n_episodes=3,
    eval_interval=3000,
)

ACTION_MEANINGS = {
    0: 'BUY',
    1: 'SELL',
    2: 'CLOSE',
    3: 'WAIT',
}

# 評価用
# agent.load('results/rainbow/' + str(STEPS) + '_finish')
# stats = experiments.eval_performance(
#     eval_env, agent, n_steps=None, n_episodes=3)
# print(stats)

# chainerrl_visualizer
# env = make_env(0, test=True)
# launch_visualizer(agent, env, ACTION_MEANINGS, raw_image_input=True)
