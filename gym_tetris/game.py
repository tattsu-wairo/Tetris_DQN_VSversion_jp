from gym_tetris.board import Board


class Game:
    def __init__(self, board: Board):
        self.board = board
        self.level = 0
        self.score = 0
        self.lines = 0
        self.drop_time = self.get_drop_speed()

        self.btb = 1
        self.attack = []
        self.attacked = []
        self.combo = 0

    def get_drop_speed(self):
        """Returns how many frames it takes for the piece to drop a cell."""
        return round(max(1.0, 8 - self.level / 2))

    def get_level_up_lines(self):
        """Returns how many lines it takes to level up."""
        return 10 * (self.level + 1)

    def get_score(self, row_count):
        """Returns the score the player receives by x rows."""
        combo = self.combo-1 # get_attack()でコンボを＋1してるので戻す
        # if row_count == 1:
        #     return 40 * (self.level + 1)
        # elif row_count == 2:
        #     return 100 * (self.level + 1)
        # elif row_count == 3:
        #     return 300 * (self.level + 1)
        # elif row_count == 4:
        #     return 1200 * (self.level + 1)
        if row_count == 1:
            return 100 + combo * 50
        elif row_count == 2:
            return 300 * + combo * 50
        elif row_count == 3:
            return 500 + combo * 50
        elif row_count == 4:
            return 800 + combo * 50
        return 0

    def get_attack(self, row_count):
        REN_BONUS_LIST = (0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5)
        basic_attack = 0
        combo_bonus = 0

        if self.combo >= len(REN_BONUS_LIST)-1:
            combo_bonus = REN_BONUS_LIST[len(REN_BONUS_LIST)-1]
        else:
            combo_bonus = REN_BONUS_LIST[self.combo]

        if row_count == 1:
            basic_attack = 0 + combo_bonus
            self.btb = 0
        elif row_count == 2:
            basic_attack = 1 + combo_bonus
            self.btb = 0
        elif row_count == 3:
            basic_attack = 2 + combo_bonus
            self.btb = 0
        elif row_count == 4:
            basic_attack = 4 + self.btb + combo_bonus
            self.btb = 1  # btbを付ける
        self.combo += 1  # RENを+1する
        if basic_attack != 0:
            self.attack.append(basic_attack)  # 火力を追加する
        return self.attack

    def _complete_rows(self, rows):
        """Remove the rows on the board and optionally adds score/level"""
        for y in rows:
            self.board.remove_row(y)
        self.attack = self.get_attack(len(rows)) # 送る火力のリストの更新
        self.score += self.get_score(len(rows))
        self.lines += len(rows)
        if self.lines >= self.get_level_up_lines():
            self.level += 1

    def _rise_rows(self, rows):
        for row in rows:
            self.board.insert_row(20,row)
    
    def send_attack(self):
        attack = self.attack.copy()
        self.attack.clear()
        return attack

    def tick(self):
        """Remove the rows on the board and optionally adds score/level"""
        rows = []

        if self.board.is_game_over():
            return rows

        self.drop_time -= 1
        if self.drop_time <= 0:
            self.board.drop_piece()
            self.drop_time = self.get_drop_speed()

        if self.board.piece is None:
            self.board.create_piece()
            self.board.attackedlist = self.attacked.copy()
            rise_rows = self.board.get_rised_rows()
            rows = self.board.get_cleared_rows()
            # 消すLINEがあるなら
            if rows:
                self._complete_rows(rows)
            else:
                self.combo = 0 # LINEを消してないのでコンボをリセット
            # お邪魔ブロックがあるなら
            if rise_rows:
                self._rise_rows(rise_rows)
                self.attacked.clear()

        return rows
