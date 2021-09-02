import sys
import gym
import gym.spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as pat

class OthelloEnv(gym.Env):
    #metadata = {'render.modes': ["human", "rgb_array", "ansi"]}
    FIELD_TYPE = ["void", "white", "black"] #[0, 1, 2]
    white, black = (1, 2)
    BOARD_W, BOARD_H = (8, 8)
    INIT_BOARD = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0, 0],
        [0, 0, 0, 2, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        ])
    MAX_STONES = 64


    def __init__(self):
        super(OthelloEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(64)
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 3,
            shape = self.INIT_BOARD.shape
        )
        self.reward_range = [-1., 1.]
        self.reset()


    def reset(self):
        self.board = self.INIT_BOARD.copy()
        self.steps = 0
        self.stones = 4
        self.player = self.black
        self.skip = 0
        self.next_place = self._enable_place()
        self.done = self._is_done()
        return self._observe()

    
    def board_reset(self, _board, _player):
        self.board = _board
        self.stones = np.sum(np.where(_board > 0, 1, 0))
        self.steps = 64 - self.stones
        self.player = _player
        self.next_place = self._enable_place()
        posts, _, _ = self.next_place
        if len(posts) == 0:
            self.skip = 1
            self.player = self._opp()
            posts, _, _ = self._enable_place()
            if len(posts) == 0:
                self.skip = 2
            else:
                pass
            self.player = _player
        else:
            self.skip = 0
        self.done = self._is_done()
        reward = self._get_reward()
        return reward


    def step(self, action=0):
        #go one step
        posts, turnovers, directions = self.next_place
        now_board = self.board.copy()
        if not self.done and not self._is_skip():
            self.skip = 0
            if action not in posts:
                action = random.choice(posts)

            next_board = self._turn_stones( 
                action, 
                turnovers[str(action)], 
                directions[str(action)]
                )
            placed = True
            placed_pos = action
            self.stones += 1
        else: # self.done is False:
            self.skip += 1
            next_board = self.board
            placed = False
            placed_pos = -1

        self.board = next_board
        self.steps += 1
        self.player = self._opp()

        #get enable places info in next board
        self.next_place = self._enable_place()
        #observation = self._observe()
        self.done = self._is_done()
        reward = self._get_reward()
        
        return now_board, placed_pos, next_board, reward, self.done, self.player #next player


    def render(self, mode="human", close=False):
        if mode == "human":
            self._render_human()
            return 0
        elif mode == "rgb_array":
            return self.board
        else:
            #not use mode = "ansi"
            return 0


    def close(self):
        pass


    def seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)


    def _enable_place(self):
        _turnovers = {}
        _directions = {}
        _positions = []
        for y in range(self.BOARD_H):
            for x in range(self.BOARD_W):
                if self.board[y][x] == 0:
                    _tr, _dn = self._search(x, y)
                    if len(_tr) > 0:
                        _act = 8*y + x
                        _turnovers[str(_act)] = _tr
                        _directions[str(_act)] = _dn
                        _positions.append(_act)
        return (_positions, _turnovers, _directions)


    def _get_reward(self):
        if self.done :
            if self._is_win() == "win":
                return 1.
            elif self._is_win() == "lose":
                return -1.
            else: #drow
                return 0.
        else:
            return 0.


    def _is_done(self):
        #game is end or not
        if self.stones < self.MAX_STONES and self.skip < 2:
            return False
        else:
            return True


    def _is_skip(self):
        posts, _, _ = self.next_place
        if len(posts) == 0:
            return True
        else:
            return False


    def _is_win(self):
        my_stones = np.sum(np.where(self.board == self.player, 1, 0))
        opp_stones = np.sum(np.where(self.board == self._opp(), 1, 0))
        if my_stones > opp_stones:
            return "win"
        elif my_stones < opp_stones:
            return "lose"
        else:
            return "draw"


    def _observe(self):
        return self.board


    def _opp(self):
        return -1*self.player + 3


    def _render_human(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _pmin, _pmax = (-0.5, 7.5)
        if self.player == self.white:
            title = "white"
        else:
            title = "black"
        ax.set_title("player: "+title)
        ax.set_xlim([_pmin, _pmax])
        ax.set_ylim([_pmin, _pmax])
        for y in range(self.BOARD_H):
            ax.axhline(y+0.5, _pmin, _pmax)
            ax.axvline(y+0.5, _pmin, _pmax)
            for x in range(self.BOARD_W):
                stone = self.board[y][x]
                if stone > 0:
                    if stone == self.white:
                        face, edge = 'w', 'k' #face=white, edge=black
                    elif stone == self.black:
                        face, edge = 'k', 'w'
                    circle = pat.Circle(xy=(x, y), radius=0.4, fc=face, ec=edge, fill=True)
                    ax.add_artist(circle)
        for y in range(8):
            for x in range(8):
                ax.text(x-0.25, y+0.25, f"{x + y*8}", size=10, color='green', fontweight='bold')
        ax.invert_yaxis()
        plt.show()


    def _search(self, x, y):
        #search 8 directions from [y][x]
        adx, ady = ((-1, 0, 1), (-1, 0, 1))
        _turnover = [] 
        _direction = [] #[dx, dy]
        for dy in ady:
            for dx in adx:
                cnt = 0
                for st in range(1, self.BOARD_H+1):
                    nx = x + st*dx
                    ny = y + st*dy
                    if nx < 0 or nx >= self.BOARD_W or ny < 0 or ny >= self.BOARD_H:
                        cnt = 0
                        break
                    else:
                        if self.board[ny][nx] == 0:
                            cnt = 0
                            break
                        elif self.board[ny][nx] == self._opp():
                            cnt += 1
                        elif cnt > 0 and self.board[ny][nx] == self.player:
                            _turnover.append(cnt)
                            _direction.append([dx, dy])
                            cnt = 0
                            break
                        else:
                            cnt = 0
                            break
        return _turnover, _direction


    def _turn_stones(self, action, turnover, direction):
        x, y = action % self.BOARD_W, action // self.BOARD_H
        self.board[y][x] = self.player
        for i, (dx, dy) in enumerate(direction):
            for st in range(1, turnover[i]+1):
                nx = x + st*dx
                ny = y + st*dy 
                self.board[ny][nx] = self.player
        return self.board


if __name__ == "__main__":
    env = OthelloEnv()
    print(env.render("rgb_array"))
    posts, _, _ = env.next_place
    print(posts)
    done = False
    type = "manual" #auto or manual
    while not done:
        if type == "manual":
            action = input("input number 0-63 to put a stone: ")
            if action.isdecimal():
              action = int(action)
            else:
                action = 0
            if action == 456:
                type = "auto"
        else:
            action = 0
        nowboard, placedpos, nextboard, reward, done, player = env.step(action)
        if player == 1:
            str_player = "white"
        else:
            str_player = "black"
        print(nextboard)
        print(f"placed at {placedpos}")
        print(reward, done, env.steps, f"player: {str_player}")
        posts, _, _ = env.next_place
        print("positions to place a stone: ", posts)
    env.render()

    newboard = np.array([
        [0,0,0,0,1,1,1,1],
        [0,0,0,0,1,1,1,1],
        [0,0,0,0,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,2,1,1,1],
    ])
    print(player)
    reward = env.board_reset(newboard, player)
    env.render()
    print(f"reward: {reward}")