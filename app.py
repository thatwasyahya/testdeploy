from flask import Flask, jsonify, request
from flask_cors import CORS
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.clock import Clock
import numpy as np
import copy
import time
import torch
from utile import get_legal_moves, initialze_board

BOARD_SIZE=8
app = Flask(__name__)
CORS(app)


def input_seq_generator(board_stats_seq,length_seq):
    
    board_stat_init=initialze_board()

    if len(board_stats_seq) >= length_seq:
        input_seq=board_stats_seq[-length_seq:]
    else:
        input_seq=[board_stat_init]
        #Padding starting board state before first index of sequence
        for i in range(length_seq-len(board_stats_seq)-1):
            input_seq.append(board_stat_init)
        #adding the inital of game as the end of sequence sample
        for i in range(len(board_stats_seq)):
            input_seq.append(board_stats_seq[i])
            
    return input_seq

def find_best_move(move1_prob,legal_moves):
    """
    Finds the best move based on the provided move probabilities and legal moves.

    Parameters:
    - move1_prob (numpy.ndarray): 2D array representing the probabilities of moves.
    - legal_moves (list): List of legal moves.

    Returns:
    - tuple: The best move coordinates (row, column).
    """

    # Initialize the best move with the first legal move
    best_move=legal_moves[0]
    
    # Initialize the maximum score with the probability of the first legal move
    max_score=move1_prob[legal_moves[0][0],legal_moves[0][1]]
    
    # Iterate through all legal moves to find the one with the maximum probability
    for i in range(len(legal_moves)):
        # Update the best move if the current move has a higher probability
        if move1_prob[legal_moves[i][0],legal_moves[i][1]]>max_score:
            max_score=move1_prob[legal_moves[i][0],legal_moves[i][1]]
            best_move=legal_moves[i]
    return best_move


class ReversiGrid(GridLayout):
    def __init__(self, **kwargs):
        super(ReversiGrid, self).__init__(**kwargs)
        self.cols = 8
        self.rows = 8
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.current_player = -1 # -1 is black, 1 is white
        self.create_board()
        self.place_initial_pieces()

    def create_board(self):
        for row in range(8):
            for col in range(8):
                cell = Button(background_color=(1, 1.8, 1, 1.8))  # Light green background
                cell.bind(on_press=self.make_move) # type: ignore
                
                if self.board[row][col] == -1:
                    cell.background_normal = 'black_circle.png'
                elif self.board[row][col] == 1:
                    cell.background_normal = 'white_circle.png'
                
                self.add_widget(cell)

    def place_initial_pieces(self):
        self.board[3][3] = 1
        self.board[4][4] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        for i, child in enumerate(reversed(self.children)):
            row, col = self.get_coords(child)
            if (row, col) in [(3, 3), (4, 4)]:
                child.background_normal = 'white_circle.png'
            elif (row, col) in [(3, 4), (4, 3)]:
                child.background_normal = 'black_circle.png'

    def get_coords(self, instance):
        index = self.children.index(instance)
        row = index // self.cols
        col = index % self.cols
        return row, col
    
    def is_valid_move(self, row, col):
        if self.board[row][col] != 0:
            return False  # La case est déjà occupée, le mouvement est donc invalide

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        valid_move = False

        for dx, dy in directions:
            r, c = row + dx, col + dy
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] != 0 and self.board[r][c] != self.current_player:
                to_flip.append((r, c))
                r += dx
                c += dy
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.current_player and to_flip:
                valid_move = True
                break

        return valid_move

    def update_board(self):
        for i, child in enumerate(reversed(self.children)):
            row, col = self.get_coords(child)
            if self.board[row][col] == -1:
                child.background_normal = 'black_circle.png'  
            elif self.board[row][col] == 1:
                child.background_normal = 'white_circle.png' 
    
    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.flip_pieces(row, col)
            self.update_board()
            self.current_player = -1 if self.current_player == 1 else 1
            black_count, white_count = self.count_pieces()

            # Check if there are no more valid moves
            if not any(self.is_valid_move(row, col) for row in range(8) for col in range(8)):
                black_count, white_count = self.count_pieces()
                winner = "Black" if black_count > white_count else "White" if white_count > black_count else "Draw"

                # Use Clock to schedule the UI update on the main thread
                Clock.schedule_once(lambda dt: self.show_winner_popup(winner), 0)

                # Return the result with winner information
                return {"success": True, "winner": winner}
            
        # If there are still valid moves, return success without winner information
        return {"success": True}
    
    def make_one_move(self, playerDisc, player): # player = difficulty (type of AI)
    # player: model description
    # board_stat: current 8x8 board status
    # turn: 1 or -1 - black or white turn
        # if current move is for player, skip
        if ((self.current_player == -1 and playerDisc == 'Black') or (self.current_player == 1 and playerDisc == 'White')):
            return -1, -1
        device = torch.device("cpu")

        conf = {}
        if (player == 'Easy'):
            conf['player']= 'server\\api\\Easy.pt'
        elif (player == 'Medium'):
            conf['player']= 'server\\api\\Medium.pt'
        elif (player == 'Hard'):
            conf['player']= 'server\\api\\Hard.pt'
        
        model = torch.load(conf['player'],map_location=torch.device('cpu'))
        model.eval()
        input_seq_boards = input_seq_generator(self.board,model.len_inpout_seq)
        
        
        #if black is the current player the board should be multiplay by -1
        if (self.current_player == -1):
            model_input=np.array([input_seq_boards])*-1
        else:
            model_input = np.array([input_seq_boards])
        move1_prob = model(torch.tensor(model_input).float().to(device))
        move1_prob = move1_prob.cpu().detach().numpy().reshape(8,8)
        legal_moves = get_legal_moves(self.board, self.current_player)
        if len(legal_moves) > 0:
            best_move = find_best_move(move1_prob,legal_moves)
            if (self.current_player == -1):
                print(f"Black: {best_move} < from possible move {legal_moves}")
            else:
                print(f"White: {best_move} < from possible move {legal_moves}")
            return best_move
    
    def show_winner_popup(self, winner):
        # Show the winner information on the main thread
        winner_text = f"The winner is {winner}!"
        popup_content = Label(text=winner_text)
        popup = Popup(title="Game Over", content=popup_content, size_hint=(None, None), size=(400, 200))
        popup.open()


    def flip_pieces(self, row, col):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            r, c = row + dx, col + dy
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] != 0 and self.board[r][c] != self.current_player:
                to_flip.append((r, c))
                r += dx
                c += dy
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.current_player:
                for r, c in to_flip:
                    self.board[r][c] = self.current_player

    def count_pieces(self):
        black_count = sum(row.count(-1) for row in self.board)
        white_count = sum(row.count(1) for row in self.board)
        return black_count, white_count 
            

# Crée une instance du jeu Reversi
reversi_game = ReversiGrid() 

class ReversiApp(App):
    def build(self):
        return ReversiGrid()
    
    
@app.route('/api/get_board', methods=['GET'])
def get_board():
    return jsonify(reversi_game.board)

@app.route('/api/get_possible_moves', methods=['GET'])
def get_possible_moves():
    return jsonify(get_legal_moves(reversi_game.board, reversi_game.current_player))


@app.route('/api/make_move', methods=['POST'])
def make_move():
    data = request.get_json()
    row = data['row']
    col = data['col']

    result = reversi_game.make_move(row, col)
    # Extract the winner information from the result
    winner = result.get("winner")
    
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust the origin based on your requirements
    if winner:
        # Add the winner information to the response
        response.headers.add('winner', winner) # type: ignore

    return response


@app.route('/api/make_one_move', methods=['POST'])
def make_one_move():
    data = request.get_json()
    difficulty = data['difficulty']
    playerDisc = data['playerDisc']
    row, col = reversi_game.make_one_move(playerDisc, difficulty)
    if (row == -1 or col == -1):
        row = data['row']
        col = data['col']
    result = reversi_game.make_move(row, col)
    # Extract the winner information from the result
    winner = result.get("winner")
    
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust the origin based on your requirements
    if winner:
        # Add the winner information to the response
        response.headers.add('winner', winner) # type: ignore

    return response


if __name__ == "__main__":
    reversi_game = ReversiGrid()
    app.run(debug=True, port=5000)