# Tetris
snowycornさんのプログラムをお借りして、対戦用環境で学習できるようにしています
Deep-Q-Networkを使用した深層強化学習をしています

Original DQN

![Deep Q Network playing Tetris](Original_mode_dqn.gif)

Original Hold Mode DQN

![Deep Q Network playing Tetris](Hold_mode_dqn.gif)

## How it works

### 強化学習

強化学習は、ある「状態」が与えられたときに、どのような「行動」をとれば「報酬」が最大になるかを判断するために用いられます。

#### 状態

| 名前  | 説明 |
| ---: | :--- |
| 穴  | フィールドにある穴の数  |
| 設置高度  | テトリミノが設置された高さ  |
| 行遷移  | 水平方向のセル遷移の数  |
| 列遷移  | 垂直方向のセル遷移の数  |
| 累積井戸  | すべての井戸の深さの合計  |
| 凸凹 | 各列の高さの差の合計  |
| 高さ合計  | 各列の高さの合計  |
| 消去ライン数 | 一度に消去したライン数  |
| 消去ブロック数  | 消去ライン数 × 10  |

#### 報酬
報酬は基本的なテトリスのスコアと同じくらいの比率で与えています。プレイヤーが生き続ける限り報酬を与え、負けると報酬が減ります。

| 名前  | 報酬 |
| ---: | :---: |
| 生存  | +1  |
| シングル  | +40  |
| ダブル  | +100  |
| トリプル  | +300  |
| テトリス  | +1200  |
| REN(1~2)  | +10  |
| REN(3~4)  | +20  |
| REN(5~6)  | +30  |
| REN(7~9)  | +40  |
| REN(10~)  | +1500  |
| ゲームオーバー  | -5  |

#### 行動

<以下原文翻訳>

前述したように、可能性のあるすべての状態を比較して、最も高い報酬を得られるものを見つけます。
最良の盤面状態をもたら行動が選択されることになる。
行動は、回転数（0〜3）と、ブロックが落ちるべき列（0〜9）のタプルである。

### Q-Learning

<以下原文翻訳>

Q-Learningを使用しなかった場合、ニューラルネットワークは将来の報酬よりも即時の報酬を得ることを好むでしょう。
この場合、たとえ障害物ができてゲームの続きが面倒になったとしても、プレイヤーは1列を消去したいと思うでしょう。
だから、Q-Learningを使うことが重要なのです。


<解釈>
Q-Learningを使うことで将来の報酬を高めるようにするため、ゲームオーバーになりにくくなる。


### Training

<以下原文翻訳>

最初は、AIがランダムに行動を選択して探索します。
エピソードごとに、過去のゲームからランダムに選んだ経験（Q-Learningも適用）を使って自己訓練します。
徐々に、探索型から探索型に移行し、ニューラルネットワークが行動を選択するようになります。


## How to run
Run `run_human.py` if you'd like to play Tetris.

Run `run_play_pierre.py.py` if you'd like to see AI with Pierre Dellacherie algorithm.

Run `Q-learning.py` if you'd like to train Q-learning agent.

Run `run_play_dqn.py` if you'd like to see the AI play Tetris without considering "Hold" action.

Run `run_train_dqn.py` if you'd like to train the AI without considering "Hold" action.

Run `run_play_dqn_hold.py` if you'd like to see the AI play Tetris when considering "Hold" action.

Run `run_train_dqn_hold.py` if you'd like to train the AI when considering "Hold" action.

Run `run_performance.py` to see how many games and frames per second it has using randomized actions.

## Links
Explanations for statistics

[Building Controllers for Tetris](https://pdfs.semanticscholar.org/e6b0/a3513e8ad6e08e9000ca2327537ac44c1c5c.pdf)

[Tetris Agent Optimization Using Harmony Search Algorithm](https://hal.inria.fr/inria-00418954/file/article.pdf)
