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
import pandas as pd


def make_prefix_best(sample):
    
    fen = sample['fen']
    vis_board = sample['vis_board']
    
    prefix = f"""You are given a chess position in FEN: {fen}.
The visual representation of the chessboard is as follows:
{vis_board}
Find the best move for the current position. Denote the move in UCI format.
Let's think step by step and output the final answer in <answer> </answer> tags.
Example final answers: <answer>e2e4</answer>, <answer>g2f1b</answer>, <answer>e8g8</answer>, etc.
"""
    return prefix


def make_prefix_legal(sample):
    
    fen = sample['fen']
    vis_board = sample['vis_board']
    condition = sample['condition']
    
    prefix = f"""You are given a chess position in FEN: {fen}.
The visual representation of the chessboard is as follows:
{vis_board}
Give a random legal move for the piece on the square {condition}. Denote the move in UCI format.
Let's think step by step and output the final answer in <answer> </answer> tags.
Example final answers: <answer>e2e4</answer>, <answer>g2f1b</answer>, <answer>e8g8</answer>, etc.
"""
    
    return prefix


def dataset_mapper(sample, split):
    
    data_source = sample['data_source']

    if "chess_best" in data_source:
        question = make_prefix_best(sample)
    elif "chess_legal" in data_source:
        question = make_prefix_legal(sample)
    else:
        raise ValueError("Unknown data source")


    inputs = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": sample.to_dict(),
        },
        "extra_info": {
            'split': split,
            'index': sample['index'],
        }
    }
    
    sample['inputs'] = inputs
    
    return sample


def get_chess_best_dataset(args):

    dataset_train = []
    dataset_test = []
    counter = 0
    
    with open(args.puzzle_path, 'r') as f:
        f.readline()
        for line in tqdm(f):
            line = line.strip().split(',')
            fen = line[1]
            board = chess.Board(fen)
            moves = line[2].split(' ')
            pre_move = moves[0]
            board.push(chess.Move.from_uci(pre_move))
            
            fen = board.fen()
            vis_board = fen_to_board(fen)
            best_move = moves[1]
            
            if counter < args.train_size:
                dataset_train.append(['chess_best_train', fen, vis_board, "placeholder", best_move])
            else:
                dataset_test.append(['chess_best_test', fen, vis_board, "placeholder", best_move])
            counter += 1
            
            if counter >= args.train_size + args.test_size:
                break
    
    dataset_train = pd.DataFrame(dataset_train, columns=['data_source', 'fen', 'vis_board', 'condition', 'ground_truth'])
    dataset_test = pd.DataFrame(dataset_test, columns=['data_source', 'fen', 'vis_board', 'condition', 'ground_truth'])

    return dataset_train, dataset_test


def get_chess_legal_dataset(args):
    
    dataset_train = []
    dataset_test = []
    counter = 0

    
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
                if from_square not in legal_moves_per_square:
                    legal_moves_per_square[from_square] = []
                legal_moves_per_square[from_square].append(move.uci())
            
            vis_board = fen_to_board(fen)
            random_square = random.choice(list(legal_moves_per_square.keys()))
            
            if counter < args.train_size:
                dataset_train.append(['chess_legal_train', fen, vis_board, random_square, ' '.join(legal_moves_per_square[random_square])])
            else:
                dataset_test.append(['chess_legal_test', fen, vis_board, random_square, ' '.join(legal_moves_per_square[random_square])])
            counter += 1
            
            if counter >= args.train_size + args.test_size:
                break

    dataset_train = pd.DataFrame(dataset_train, columns=['data_source', 'fen', 'vis_board', 'condition', 'ground_truth'])
    dataset_test = pd.DataFrame(dataset_test, columns=['data_source', 'fen', 'vis_board', 'condition', 'ground_truth'])
    
    return dataset_train, dataset_test


def create_datasets(args):
    
    dataset_best_train, dataset_best_test = get_chess_best_dataset(args)
    dataset_legal_train, dataset_legal_test = get_chess_legal_dataset(args)
    
    dataset_train = pd.concat([dataset_best_train, 
                               dataset_legal_train
                            ], axis=0).reset_index(drop=True).reset_index()
    dataset_test = pd.concat([dataset_best_test, 
                              dataset_legal_test
                            ], axis=0).reset_index(drop=True).reset_index()

    dataset_train = dataset_train.apply(lambda x: dataset_mapper(x, "train"), axis=1)
    dataset_test = dataset_test.apply(lambda x: dataset_mapper(x, "test"), axis=1)
    
    dataset_train = Dataset.from_list(dataset_train['inputs'].tolist())
    dataset_test = Dataset.from_list(dataset_test['inputs'].tolist())
    
    return dataset_train, dataset_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./input_data/chess_multi')
    parser.add_argument('--train_size', type=int, default=8196*8)
    parser.add_argument('--test_size', type=int, default=512)
    parser.add_argument('--puzzle_path', type=str, default='./source_data/lichess_db_puzzle.csv')
    args = parser.parse_args()
    
    dataset_train, dataset_test = create_datasets(args)

    dataset_train.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    dataset_test.to_parquet(os.path.join(args.local_dir, 'test.parquet'))