import pygame

COLORS = {
    1: [(0, 240, 240), (0, 216, 216)],
    2: [(240, 240, 0), (216, 216, 0)],
    3: [(160, 0, 240), (144, 0, 216)],
    4: [(0, 240, 0), (0, 216, 0)],
    5: [(240, 0, 0), (216, 0, 0)],
    6: [(0, 160, 240), (0, 144, 216)],
    7: [(240, 160, 0), (216, 144, 0)],
    8: [(100, 100, 100), (100, 100, 100)],
}

PIECE_OFFSETS = {
    1: (-0.5, -0.5),
    2: (-0.5, 0),
    3: (0, -1),
    4: (0, -1),
    5: (0, -1),
    6: (0, -1),
    7: (0, -1),
}

FRAMES_PER_SECOND = 30
BETWEEN_BORD_SPACE = 500  # 　プレイヤー2の盤面開始位置

"""
* pygameメモ *
==============
rect : (左上のX座標, 左上のY座標, 幅, 高さ)
"""


class View:

    def __init__(self, win, font):
        self.win = win
        self.font = font
        self.clock = pygame.time.Clock()

        # 左側のボード
        self.board_rect = (25, 25, 240, 476)
        self.board_x, self.board_y, self.board_width, self.board_height = self.board_rect
        self.hold_rect = (290, 25, 144, 72)
        self.next_rect = (290, 122, 144, 72)

        # 右側のボード
        self.board_rect_2 = (25+BETWEEN_BORD_SPACE, 25, 240, 476)
        self.board_x_2, self.board_y_2, self.board_width_2, self.board_height_2 = self.board_rect_2
        self.hold_rect_2 = (290+BETWEEN_BORD_SPACE, 25, 144, 72)
        self.next_rect_2 = (290+BETWEEN_BORD_SPACE, 122, 144, 72)

    def draw(self, game1=None, game2=None):
        """"Draws everything."""
        self.clock.tick(FRAMES_PER_SECOND)
        self.win.fill((155, 155, 155))
        self._draw_board(game1.board)
        self._draw_piece_next(game1.board.piece_next)
        self._draw_piece_holding(game1.board.piece_holding)
        self._draw_score(game1.level, game1.score, game1.lines)
        if game1.board.is_game_over():
            self._draw_game_over()
        if game2 != None:
            self._draw_board(game2.board, player=2)
            self._draw_piece_next(game2.board.piece_next, player=2)
            self._draw_piece_holding(game2.board.piece_holding, player=2)
            self._draw_score(game2.level, game2.score, game2.lines, player=2)
            if game2.board.is_game_over():
                self._draw_game_over(player=2)
        pygame.display.update()

    def _draw_board(self, board, player=1):
        """"Draws the board."""
        # プレイヤー1のボードか
        if player == 1:
            board_rect = self.board_rect
        # プレイヤー2のボードか
        else:
            board_rect = self.board_rect_2
        begin_x, begin_y, width, height = board_rect
        cell_width = width / board.columns
        cell_height = height / board.rows

        pygame.draw.rect(self.win, (5, 5, 5), board_rect)

        for board_y in range(board.rows):
            for board_x in range(board.columns):
                code = board.pieces_table[board_y][board_x]
                if code != 0:
                    x = begin_x + board_x * cell_width
                    y = begin_y + board_y * cell_height
                    self._draw_piece_cell(
                        (x, y, cell_width, cell_height), code)

        if board.piece is not None:

            ghost_y = 0
            while board.can_move_piece(0, ghost_y + 1):
                ghost_y += 1

            for board_x, board_y in board.piece.get_shape_coords():
                x = begin_x + board_x * cell_width
                y = begin_y + board_y * cell_height
                code = board.piece.shape.code
                self._draw_piece_cell((x, y, cell_width, cell_height), code)

                if ghost_y != 0:
                    self._draw_ghost_piece_cell(
                        (x, y + ghost_y * cell_height, cell_width, cell_height), code)

        self._draw_grid(board_rect, board.rows, board.columns, 2)

    def _draw_piece_cell(self, rect, code):
        """"Draws the cell of a piece."""
        x, y, width, height = rect
        pygame.draw.rect(self.win, COLORS[code][1], rect)
        pygame.draw.rect(
            self.win, COLORS[code][0], (x + 3, y + 3, width - 6, height - 6))

    def _draw_ghost_piece_cell(self, rect, code):
        """"Draws the cell of a ghost piece."""
        x, y, width, height = rect
        self._draw_rect(rect, COLORS[code][0], 50)
        self._draw_rect((x + 3, y + 3, width - 6, height - 6),
                        COLORS[code][1], 5)

    def _draw_piece_holding(self, piece_holding, player=1):
        """"Draws the held piece."""
        if player == 1:
            hold_rect = self.hold_rect
        else:
            hold_rect = self.hold_rect_2
        pygame.draw.rect(self.win, (0, 0, 0), hold_rect)
        self._draw_piece(piece_holding, hold_rect)
        text_surface = self.font.render('Hold', False, (255, 255, 255))
        self.win.blit(
            text_surface, (hold_rect[0] + 65, hold_rect[1] + 47))

    def _draw_piece_next(self, piece_next, player=1):
        """"Draws the next piece."""
        if player == 1:
            next_rect = self.next_rect
        else:
            next_rect = self.next_rect_2
        pygame.draw.rect(self.win, (0, 0, 0), next_rect)
        self._draw_piece(piece_next, next_rect)
        text_surface = self.font.render('Next', False, (255, 255, 255))
        self.win.blit(
            text_surface, (next_rect[0] + 65, next_rect[1] + 47))

    def _draw_piece(self, piece, rect):
        """"Draws a piece."""
        if piece is None:
            return
        begin_x, begin_y, width, height = rect
        cell_width = 24
        cell_height = 24
        begin_x += width / 2 + 12
        begin_y += 12

        code = piece.shape.code
        offset_x, offset_y = PIECE_OFFSETS[code]

        for board_x, board_y in piece.get_shape_coords():
            x = begin_x + (board_x + offset_x) * cell_width
            y = begin_y + (board_y + offset_y) * cell_height
            self._draw_piece_cell((x, y, cell_width, cell_height), code)

    def _draw_score(self, level, score, lines, player=1):
        """"Draws the score."""
        if player == 1:
            board_x, board_y, board_width = self.board_x, self.board_y, self.board_width
        else:
            board_x, board_y, board_width = self.board_x_2, self.board_y_2, self.board_width_2
        level_surface = self.font.render(
            'Level ' + str(level), False, (0, 0, 0))
        self.win.blit(level_surface, (board_x +
                      board_width + 25, board_y + 200))
        score_surface = self.font.render(
            'Score ' + str(score), False, (0, 0, 0))
        self.win.blit(score_surface, (board_x +
                      board_width + 25, board_y + 250))
        score_surface = self.font.render(
            'Lines  ' + str(lines), False, (0, 0, 0))
        self.win.blit(score_surface, (board_x +
                      board_width + 25, board_y + 300))
        pass

    def _draw_game_over(self, player=1):
        """"Draws game over text."""
        if player == 1:
            board_rect = self.board_rect
            board_x, board_y = self.board_x, self.board_y
        else:
            board_rect = self.board_rect_2
            board_x, board_y = self.board_x_2, self.board_y_2
        self._draw_rect(board_rect, (0, 0, 0), 150)
        text_surface = self.font.render('game over', False, (255, 255, 255))
        self.win.blit(text_surface, (board_x + 30, board_y + 5))

    def _draw_rect(self, rect, rect_color, alpha=255):
        """"Draws a rectangle."""
        surface = pygame.Surface(rect[2:])
        surface.set_alpha(alpha)
        surface.fill(rect_color)
        self.win.blit(surface, rect[:2])

    def _draw_grid(self, rect, rows, columns, line):
        """"Draws a grid."""
        begin_x, begin_y, width, height = rect
        cell_width = width / columns
        cell_height = height / rows
        half_line = line / 2

        for board_x in range(columns + 1):
            x = begin_x + board_x * cell_width - line / 2
            y = begin_y - half_line
            pygame.draw.rect(self.win, (0, 0, 0), (x, y, line, height + line))

        for board_y in range(rows + 1):
            y = begin_y + board_y * cell_height - half_line
            x = (begin_x - half_line)
            pygame.draw.rect(self.win, (0, 0, 0), (x, y, width + line, line))
