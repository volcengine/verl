

mapper = {
    'K': 'White King',
    'Q': 'White Queen',
    'R': 'White Rook',
    'B': 'White Bishop',
    'N': 'White Knight',
    'P': 'White Pawn',
    'k': 'Black King',
    'q': 'Black Queen',
    'r': 'Black Rook',
    'b': 'Black Bishop',
    'n': 'Black Knight',
    'p': 'Black Pawn',
    '.': 'Empty',
}

def fen_to_board(fen_string):
    """
    Convert a FEN string to an 8x8 representation of a chess board with labels.
    
    Args:
        fen_string (str): A valid FEN string (only the board position part is used)
    
    Returns:
        str: A string representation of the chess board with single labels
    """
    # Extract the board position part of the FEN (first part before the first space)
    board_fen = fen_string.split(' ')[0]
    
    # Initialize an empty 8x8 board
    board = [['.' for _ in range(8)] for _ in range(8)]
    
    # Split the FEN board into ranks (rows)
    ranks = board_fen.split('/')
    
    # Process each rank
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0  # Column index
        
        for char in rank:
            if char.isdigit():
                # Skip empty squares
                file_idx += int(char)
            else:
                # Place the piece on the board
                board[rank_idx][file_idx] = char
                file_idx += 1
    
    # Column labels (a-h) only once at the top
    result = ' '.join('abcdefgh') + '\n'
    result += ' '.join(['-' for _ in range(8)]) + '\n'
    
    # Board with row labels (8-1) only once on the left
    for i, row in enumerate(board):
        row_label = str(8 - i)
        result += f"{' '.join(row)} | {row_label}\n"
    
    return result