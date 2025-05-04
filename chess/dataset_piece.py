"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

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


def get_chess_piece_dataset(args):
    
    fens = []
    vis_boards = []
    piece_squares = []
    piece_names = []
    
    with open(args.puzzle_path, 'r') as f:
        f.readline()
        for line in tqdm(f):
            line = line.strip().split(',')
            fen = line[1]
            board = chess.Board(fen)
            
            pieces = {}
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece is not None:
                    piece_symbol = piece.symbol()
                    square_name = chess.square_name(square)
                    if piece_symbol not in pieces:
                        pieces[piece_symbol] = []
                    pieces[piece_symbol].append(square_name)
                else:
                    pieces['.'] = pieces.get('.', []) + [chess.square_name(square)]
        
            assert sum([len(v) for v in pieces.values()]) == 64
            
            random_name = random.choice(list(pieces.keys()))
            random_square = random.choice(pieces[random_name])

            fens.append(fen)
            vis_boards.append(fen_to_board(fen))
            piece_squares.append(random_square)
            piece_names.append(mapper[random_name])
            
            if len(fens) >= args.train_size + args.test_size:
                break
            

    dataset = Dataset.from_dict({"fen": fens,
                                "vis_board": vis_boards,
                                "piece_square": piece_squares,
                                "piece_name": piece_names
                                })
    
    return dataset


def make_prefix_piece(dp):
    
    fen = dp['fen']
    vis_board = dp['vis_board']
    piece_square = dp['piece_square']
    
    prefix = f"""You are given a chess position in FEN: {fen}.
The visual representation of the chessboard is as follows:
{vis_board}
Which piece is on the square {piece_square}?
Let's think step by step and output the final answer in <answer> </answer> tags.
Example final answers: <answer>White King</answer>, <answer>Black Queen</answer>, <answer>Empty</answer>, etc.
"""
    
    return prefix


def make_map_fn(split):
    def process_fn(sample, idx):

        data_source = f'chess_piece_{split}'
        question = make_prefix_piece(sample)
        
        solution = {
            "fen": sample['fen'],
            "vis_board": sample['vis_board'],
            "piece_square": sample['piece_square'],
            "piece_name": sample['piece_name'],
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
    parser.add_argument('--local_dir', default='./input_data/chess_piece')
    parser.add_argument('--train_size', type=int, default=8192*16)
    parser.add_argument('--test_size', type=int, default=512)
    parser.add_argument('--puzzle_path', type=str, default='./source_data/lichess_db_puzzle.csv')
    args = parser.parse_args()
    
    raw_dataset = get_chess_piece_dataset(args)
    
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
