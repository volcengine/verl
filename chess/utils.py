import chess

mapper = {
    'K': 'White King (K)',
    'Q': 'White Queen (Q)',
    'R': 'White Rook (R)',
    'B': 'White Bishop (B)',
    'N': 'White Knight (N)',
    'P': 'White Pawn (P)',
    'k': 'Black King (k)',
    'q': 'Black Queen (q)',
    'r': 'Black Rook (r)',
    'b': 'Black Bishop (b)',
    'n': 'Black Knight (n)',
    'p': 'Black Pawn (p)'
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



def fen_to_pieces(fen):
    
    board = chess.Board(fen)
    
    pieces = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_symbol = mapper[piece.symbol()]
            square_name = chess.square_name(square)
            if piece_symbol not in pieces:
                pieces[piece_symbol] = []
            pieces[piece_symbol].append(square_name)
        else:
            pieces['Empty'] = pieces.get('Empty', []) + [chess.square_name(square)]
    
    assert sum([len(v) for v in pieces.values()]) == 64
    
    return pieces