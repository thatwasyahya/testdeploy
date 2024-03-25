from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import torch
from utile import get_legal_moves, initialze_board

app = Flask(__name__)
CORS(app)

board = [[0 for _ in range(8)] for _ in range(8)]
current_player = -1

def input_seq_generator(board_stats_seq, length_seq):
    board_stat_init = initialze_board()

    if len(board_stats_seq) >= length_seq:
        input_seq = board_stats_seq[-length_seq:]
    else:
        input_seq = [board_stat_init]
        for i in range(length_seq - len(board_stats_seq) - 1):
            input_seq.append(board_stat_init)
        for i in range(len(board_stats_seq)):
            input_seq.append(board_stats_seq[i])
            
    return input_seq

def find_best_move(move1_prob, legal_moves):
    best_move = legal_moves[0]
    max_score = move1_prob[legal_moves[0][0], legal_moves[0][1]]
    
    for i in range(len(legal_moves)):
        if move1_prob[legal_moves[i][0], legal_moves[i][1]] > max_score:
            max_score = move1_prob[legal_moves[i][0], legal_moves[i][1]]
            best_move = legal_moves[i]
    return best_move

def make_ai_move(player_disc, player):
    global board, current_player

    if (current_player == -1 and player_disc == 'Black') or (current_player == 1 and player_disc == 'White'):
        return -1, -1

    device = torch.device("cpu")

    conf = {}
    if player == 'Easy':
        conf['player'] = 'server\\api\\Easy.pt'
    elif player == 'Medium':
        conf['player'] = 'server\\api\\Medium.pt'
    elif player == 'Hard':
        conf['player'] = 'server\\api\\Hard.pt'

    model = torch.load(conf['player'], map_location=torch.device('cpu'))
    model.eval()
    input_seq_boards = input_seq_generator(board, model.len_inpout_seq)

    if current_player == -1:
        model_input = np.array([input_seq_boards]) * -1
    else:
        model_input = np.array([input_seq_boards])
    
    move1_prob = model(torch.tensor(model_input).float().to(device))
    move1_prob = move1_prob.cpu().detach().numpy().reshape(8, 8)
    legal_moves = get_legal_moves(board, current_player)

    if len(legal_moves) > 0:
        best_move = find_best_move(move1_prob, legal_moves)
        if current_player == -1:
            print(f"Black: {best_move} < from possible move {legal_moves}")
        else:
            print(f"White: {best_move} < from possible move {legal_moves}")
        return best_move
    return -1, -1

@app.route('/api/get_board', methods=['GET'])
def get_board():
    return jsonify(board)

@app.route('/api/get_possible_moves', methods=['GET'])
def get_possible_moves():
    return jsonify(get_legal_moves(board, current_player))

@app.route('/api/make_move', methods=['POST'])
def make_move():
    global board, current_player

    data = request.get_json()
    row = data['row']
    col = data['col']

    if is_valid_move(row, col):
        board[row][col] = current_player
        flip_pieces(row, col)
        current_player = -1 if current_player == 1 else 1
        black_count, white_count = count_pieces()

        if not any(is_valid_move(row, col) for row in range(8) for col in range(8)):
            black_count, white_count = count_pieces()
            winner = "Black" if black_count > white_count else "White" if white_count > black_count else "Draw"
            return jsonify({"success": True, "winner": winner})

    return jsonify({"success": True})

@app.route('/api/make_one_move', methods=['POST'])
def make_one_move():
    global board, current_player

    data = request.get_json()
    difficulty = data['difficulty']
    player_disc = data['playerDisc']
    row, col = make_ai_move(player_disc, difficulty)
    if row == -1 or col == -1:
        row = data['row']
        col = data['col']
    if is_valid_move(row, col):
        board[row][col] = current_player
        flip_pieces(row, col)
        current_player = -1 if current_player == 1 else 1
        black_count, white_count = count_pieces()

        if not any(is_valid_move(row, col) for row in range(8) for col in range(8)):
            black_count, white_count = count_pieces()
            winner = "Black" if black_count > white_count else "White" if white_count > black_count else "Draw"
            return jsonify({"success": True, "winner": winner})

    return jsonify({"success": True})

def is_valid_move(row, col):
    global board, current_player

    if board[row][col] != 0:
        return False

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for dx, dy in directions:
        r, c = row + dx, col + dy
        to_flip = []
        while 0 <= r < 8 and 0 <= c < 8 and board[r][c] != 0 and board[r][c] != current_player:
            to_flip.append((r, c))
            r += dx
            c += dy
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == current_player and to_flip:
            return True
    return False

def flip_pieces(row, col):
    global board

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),(1, -1), (1, 0), (1, 1)]
    
    for dx, dy in directions:
        r, c = row + dx, col + dy
        to_flip = []
        while 0 <= r < 8 and 0 <= c < 8 and board[r][c] != 0 and board[r][c] != current_player:
            to_flip.append((r, c))
            r += dx
            c += dy
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == current_player:
            for r, c in to_flip:
                board[r][c] = current_player

def count_pieces():
    global board

    black_count = sum(row.count(-1) for row in board)
    white_count = sum(row.count(1) for row in board)
    return black_count, white_count 

if __name__ == "__main__":
    app.run()

