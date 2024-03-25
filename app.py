from flask import Flask, jsonify, request
from flask_cors import CORS
from utile import get_legal_moves, initialze_board

app = Flask(__name__)
CORS(app)

class ReversiGrid:
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.current_player = -1
        self.place_initial_pieces()

    def place_initial_pieces(self):
        self.board[3][3] = 1
        self.board[4][4] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1

    def is_valid_move(self, row, col):
        if self.board[row][col] != 0:
            return False

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            r, c = row + dx, col + dy
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] != 0 and self.board[r][c] != self.current_player:
                to_flip.append((r, c))
                r += dx
                c += dy
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.current_player and to_flip:
                return True

        return False

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.current_player = -1 if self.current_player == 1 else 1
            black_count, white_count = self.count_pieces()

            if not any(self.is_valid_move(row, col) for row in range(8) for col in range(8)):
                black_count, white_count = self.count_pieces()
                winner = "Black" if black_count > white_count else "White" if white_count < black_count else "Draw"
                return {"success": True, "winner": winner}

        return {"success": True}

    def make_one_move(self, player_disc, player):
        return -1, -1
    
    def count_pieces(self):
        black_count = sum(row.count(-1) for row in self.board)
        white_count = sum(row.count(1) for row in self.board)
        return black_count, white_count 

reversi_game = ReversiGrid()


@app.route('/get_board', methods=['GET'])
def get_board():
    return jsonify(reversi_game.board)

@app.route('/get_possible_moves', methods=['GET'])
def get_possible_moves():
    return jsonify(get_legal_moves(reversi_game.board, reversi_game.current_player))

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.get_json()
    row = data['row']
    col = data['col']

    result = reversi_game.make_move(row, col)
    winner = result.get("winner")
    
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    if winner:
        response.headers.add('winner', winner)

    return response

@app.route('/make_one_move', methods=['POST'])
def make_one_move():
    data = request.get_json()
    difficulty = data['difficulty']
    player_disc = data['playerDisc']
    row, col = reversi_game.make_one_move(player_disc, difficulty)
    if row == -1 or col == -1:
        row = data['row']
        col = data['col']
    result = reversi_game.make_move(row, col)
    winner = result.get("winner")
    
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    if winner:
        response.headers.add('winner', winner)

    return response

