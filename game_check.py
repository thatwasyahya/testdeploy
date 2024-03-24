
import numpy as np
import copy
import time


import torch

from utile import get_legal_moves, is_legal_move, has_tile_to_flip, initialze_board

BOARD_SIZE=8


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

def apply_flip(best_move,board_stat,NgBlackPsWhith):
    """
    Apply tile flipping on the Othello board based on the best move.

    Parameters:
    - best_move (tuple): Coordinates (row, column) of the best move.
    - board_stat (numpy.ndarray): 2D array representing the current state of the Othello board.
    - NgBlackPsWhith (int): Indicator for the current player (Black: -1, White: 1).

    Returns:
    - numpy.ndarray: Updated Othello board after applying tile flipping.
    """
    
    MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
             (0, -1),           (0, +1),
             (+1, -1), (+1, 0), (+1, +1)]

    for direction in MOVE_DIRS:
        if has_tile_to_flip(best_move, direction,board_stat,NgBlackPsWhith):
            i = 1
            while True:
                row = best_move[0] + direction[0] * i
                col = best_move[1] + direction[1] * i
                if board_stat[row][col] == board_stat[best_move[0], best_move[1]]:
                    break
                else:
                    board_stat[row][col] = board_stat[best_move[0], best_move[1]]
                    i += 1
                    
    return board_stat


def launch_game(player1, player2):
    # Two rounds of game would be played
    # First player1 starts game, and then Player2 starts the other game
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    conf={}
    if (player1 == 'Easy'):
        conf['player1']= 'model_e2205046.pt'
    if (player2 == 'Easy'):
        conf['player2']= 'model_e2205046.pt'

    board_stat=initialze_board()

    moves_log=""
    board_stats_seq=[]
    pass2player=False

    while not np.all(board_stat) and not pass2player:

        NgBlackPsWhith=-1
        board_stats_seq.append(copy.copy(board_stat))
        if torch.cuda.is_available():
            model = torch.load(conf['player1'])
        else:
            model = torch.load(conf['player1'],map_location=torch.device('cpu'))
        model.eval()

        input_seq_boards=input_seq_generator(board_stats_seq,model.len_inpout_seq)
    	#if black is the current player the board should be multiplay by -1
        model_input=np.array([input_seq_boards])*-1
        move1_prob = model(torch.tensor(model_input).float().to(device))
        move1_prob=move1_prob.cpu().detach().numpy().reshape(8,8)

        legal_moves=get_legal_moves(board_stat,NgBlackPsWhith)

        if len(legal_moves)>0:
            
            best_move=find_best_move(move1_prob,legal_moves)
            print(f"Black: {best_move} < from possible move {legal_moves}")

            board_stat[best_move[0],best_move[1]]=NgBlackPsWhith
            moves_log+=str(best_move[0]+1)+str(best_move[1]+1)
            
            board_stat=apply_flip(best_move,board_stat,NgBlackPsWhith)

        else:
            print("Black pass")
            if moves_log[-2:]=="__":
                pass2player=True
            moves_log+="__"


        NgBlackPsWhith=+1
        board_stats_seq.append(copy.copy(board_stat))
        if torch.cuda.is_available():
            model = torch.load(conf['player2'])
        else:
            model = torch.load(conf['player2'],map_location=torch.device('cpu'))
        model.eval()

        input_seq_boards=input_seq_generator(board_stats_seq,model.len_inpout_seq)
        #if black is the current player the board should be multiplay by -1
        model_input=np.array([input_seq_boards])
        move1_prob = model(torch.tensor(model_input).float().to(device))
        move1_prob=move1_prob.cpu().detach().numpy().reshape(8,8)

        legal_moves=get_legal_moves(board_stat,NgBlackPsWhith)


        if len(legal_moves)>0:
            
            best_move = find_best_move(move1_prob,legal_moves)
            print(f"White: {best_move} < from possible move {legal_moves}")
            
            board_stat[best_move[0],best_move[1]]=NgBlackPsWhith
            moves_log+=str(best_move[0]+1)+str(best_move[1]+1)
            
            board_stat=apply_flip(best_move,board_stat,NgBlackPsWhith)

        else:
            print("White pass")
            if moves_log[-2:]=="__":
                pass2player=True
            moves_log+="__"

    board_stats_seq.append(copy.copy(board_stat))
    print("Moves log:",moves_log)

    try:
        with open ('moves.txt', 'w') as f:
            f.write(moves_log)
    except:
        with open ('moves.txt', 'a') as f:
            f.write(moves_log)

    if np.sum(board_stat)<0:
        print(f"Black {conf['player1']} is winner (with {-1*int(np.sum(board_stat))} points)")
    elif np.sum(board_stat)>0:
        print(f"White {conf['player2']} is winner (with {int(np.sum(board_stat))} points)")
    else:
        print(f"Draw")

def make_one_move(player, board_stat, turn):
    # player: model description
    # board_stat: current 8x8 board status
    # turn: 1 or -1 - black or white turn
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    conf={}
    if (player == 'Easy'):
        conf['player'] = 'server\\api\\model_e2205046.pt'

    if torch.cuda.is_available():
        model = torch.load(conf['player'])
    else:
        model = torch.load(conf['player'], map_location=torch.device('cpu'))
    model.eval()
    input_seq_boards = input_seq_generator(board_stat,model.len_inpout_seq)
    
    #if black is the current player the board should be multiplay by -1
    if (turn == -1):
        model_input=np.array([input_seq_boards])*-1
    else:
        model_input=np.array([input_seq_boards])
    move1_prob = model(torch.tensor(model_input).float().to(device))
    move1_prob = move1_prob.cpu().detach().numpy().reshape(8,8)
    legal_moves = get_legal_moves(board_stat,turn)
    if len(legal_moves)>0:
        best_move = find_best_move(move1_prob,legal_moves)
        if (turn == -1):
            print(f"Black: {best_move} < from possible move {legal_moves}")
        else:
            print(f"White: {best_move} < from possible move {legal_moves}")
        
        board_stat[best_move[0],best_move[1]]=turn
        board_stat=apply_flip(best_move,board_stat,turn)
    return best_move