import re
import os
from datasets import Dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import pdb
import chess
import random
import json
from utils import mapper, fen_to_board


def get_chess_legal_dataset(args):
    
    fens = []
    vis_boards = []
    piece_squares = []
    piece_moves = []
    
    with open(args.puzzle_path, 'r') as f:
        f.readline()
        for line in tqdm(f):
            line = line.strip().split(',')
            fen = line[1]
            board = chess.Board(fen)
            legal_moves = board.legal_moves
            
            legal_moves_per_square = {}
            
            for move in legal_moves:
                from_square = chess.square_name(move.from_square)
                from_piece = board.piece_at(move.from_square).symbol()
                if from_square not in legal_moves_per_square:
                    legal_moves_per_square[from_square] = {"piece": from_piece, 
                                                           "piece_name": mapper[from_piece],
                                                           "moves": []}
                legal_moves_per_square[from_square]["moves"].append(move.uci())
            
            random_square = random.choice(list(legal_moves_per_square.keys()))

            fens.append(fen)
            vis_boards.append(fen_to_board(fen))
            piece_squares.append(random_square)
            piece_moves.append(legal_moves_per_square[random_square])
            
            if len(fens) >= args.train_size + args.test_size:
                break
            

    dataset = Dataset.from_dict({"fen": fens,
                                "vis_board": vis_boards,
                                "piece_square": piece_squares,
                                "piece_moves": piece_moves
                                })
    
    return dataset


def make_prefix_piece(dp):
    
    fen = dp['fen']
    vis_board = dp['vis_board']
    piece_square = dp['piece_square']
    
    prefix = f"""You are given a chess position in FEN: {fen}.
The visual representation of the chessboard is as follows:
{vis_board}
Give a random legal move for the piece on the square {piece_square}.
Let's think step by step and output the final answer in <answer> </answer> tags.
Example final answers: <answer>e2e4</answer>, <answer>g2f1b</answer>, <answer>e8g8</answer>, etc.
"""
    
    return prefix


def make_map_fn(split):
    
    def process_fn(sample, idx):

        data_source = f'chess_legal_{split}'
        question = make_prefix_piece(sample)
        
        solution = {
            "fen": sample['fen'],
            "vis_board": sample['vis_board'],
            "piece_square": sample['piece_square'],
            "piece_moves": sample['piece_moves'],
        }
        
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data
    
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./input_data/chess_legal')
    parser.add_argument('--train_size', type=int, default=8196*16)
    parser.add_argument('--test_size', type=int, default=512)
    parser.add_argument('--puzzle_path', type=str, default='./source_data/lichess_db_puzzle.csv')
    args = parser.parse_args()
    
    raw_dataset = get_chess_legal_dataset(args)
    
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
