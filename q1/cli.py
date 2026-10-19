import math

from .mcts import mcts
from .minimax import minimax
from .state import QuartoState


def print_board(state):
    print("\n" + "=" * 30)
    print("      TABULEIRO QUARTO      ")
    print("=" * 30)

    def format_piece(piece):
        return f"[{piece:04b}]" if piece is not None else "[    ]"

    for row in range(4):
        cells = [format_piece(state.board[row * 4 + column]) for column in range(4)]
        print(f"L{row}:  " + "  ".join(cells))

    print("\nPeças disponíveis:")
    available = [f"{p} ({p:04b})" for p in state.available_pieces]
    for index in range(0, len(available), 4):
        print(" | ".join(available[index:index + 4]))

    if state.piece_in_hand is not None:
        print(f"\n=> Peça a colocar: {state.piece_in_hand} ({state.piece_in_hand:04b})")
    print("=" * 30)


def human_turn(state):
    print_board(state)
    position = None
    next_piece = None
    legal_actions = state.get_legal_actions()

    if state.piece_in_hand is not None:
        while True:
            try:
                row = int(input("Digite a linha (0-3) para colocar a peça: "))
                column = int(input("Digite a coluna (0-3) para colocar a peça: "))
                position = row * 4 + column
                if 0 <= row <= 3 and 0 <= column <= 3 and state.board[position] is None:
                    break
                print("Posição inválida ou já ocupada.")
            except ValueError:
                print("Por favor, introduza um número válido.")

    if state.available_pieces:
        while True:
            try:
                next_piece = int(input("Digite o ID da próxima peça (0-15): "))
                if next_piece in state.available_pieces:
                    break
                print("Peça indisponível. Escolha da lista.")
            except ValueError:
                print("Por favor, introduza um número válido.")

    action = (position, next_piece)
    if action not in legal_actions:
        print("Ação ilegal. Tente novamente.")
        return human_turn(state)
    return action


def play_game():
    print("Bem-vindo ao QUARTO!")
    print("1. Minimax (Profundidade 2)")
    print("2. MCTS (500 iterações)")
    adversary = "MINIMAX" if input("Opção: ") == "1" else "MCTS"
    state = QuartoState()

    while not state.is_terminal():
        if state.current_player == 1:
            print("\n--- SEU TURNO ---")
            action = human_turn(state)
        else:
            print(f"\n--- TURNO DO AGENTE ({adversary}) ---")
            if adversary == "MINIMAX":
                _, action = minimax(
                    state, 2, -math.inf, math.inf, True, 2,
                )
            else:
                action = mcts(state, 500)
            if action is None:
                raise RuntimeError("O agente não encontrou uma ação legal.")
            print(f"O Agente colocou a peça na posição: {action[0]}")
            print(f"O Agente entregou-lhe a peça: {action[1]}")
        state = state.apply_action(action)

    print_board(state)
    winner = state.check_win()
    print(f"\nFIM DE JOGO! O Jogador {winner} venceu!" if winner else "\nFIM DE JOGO! Empate!")
