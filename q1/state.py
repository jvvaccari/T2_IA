import copy


class QuartoState:
    """Estado do tabuleiro, peças disponíveis e jogador atual."""

    LINES = (
        (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15),
        (0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
        (0, 5, 10, 15), (3, 6, 9, 12),
    )

    def __init__(self):
        self.board = [None] * 16
        self.available_pieces = list(range(16))
        self.piece_in_hand = None
        self.current_player = 1

    def get_legal_actions(self):
        """Retorna ações como (posição, próxima peça)."""
        if self.piece_in_hand is None:
            return [(None, piece) for piece in self.available_pieces]

        empty_spots = [i for i, piece in enumerate(self.board) if piece is None]
        if not self.available_pieces:
            return [(spot, None) for spot in empty_spots]

        return [
            (spot, piece)
            for spot in empty_spots
            for piece in self.available_pieces
        ]

    def apply_action(self, action):
        """Aplica uma ação em uma cópia e retorna o novo estado."""
        new_state = copy.deepcopy(self)
        position, next_piece = action

        if position is not None:
            new_state.board[position] = new_state.piece_in_hand
        if next_piece is not None:
            new_state.available_pieces.remove(next_piece)

        new_state.piece_in_hand = next_piece
        new_state.current_player = 3 - self.current_player
        return new_state

    def check_win(self):
        """Retorna o jogador vencedor ou None."""
        for line in self.LINES:
            pieces = [
                piece
                for index in line
                if isinstance(piece := self.board[index], int)
            ]
            if len(pieces) != 4:
                continue

            p0, p1, p2, p3 = pieces
            same_one = (p0 & p1 & p2 & p3) > 0
            same_zero = ((~p0 & ~p1 & ~p2 & ~p3) & 15) > 0
            if same_one or same_zero:
                return 3 - self.current_player
        return None

    def is_terminal(self):
        return self.check_win() is not None or None not in self.board
