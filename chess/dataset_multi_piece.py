import re
import os
from datasets import Dataset
from tqdm import tqdm
import argparse
import pdb
import chess
import random
import json
from utils import mapper, fen_to_board, fen_to_pieces
import pandas as pd


# def get_prompt_best_wo(sample, vis):
    
#     fen = sample['fen']
#     vis_board = sample['vis_board']
    
#     if vis:
#         prefix = f"You are given a chess position in FEN: {fen}.\nThe visual representation of the chessboard is as follows:\n{vis_board}"
#     else:
#         prefix = f"You are given a chess position in FEN: {fen}.\n"
    
#     question = f"Find the best move for the current position. Denote the move in UCI format.\n"
#     instruction = f"Let's think step by step and output the final answer in <answer> </answer> tags.\n"
#     examples = f"Example final answers: <answer>e2e4</answer>, <answer>g2f1b</answer>, <answer>e8g8</answer>, etc.\n"

#     return prefix + question + instruction + examples


# def get_prompt_best_w(sample, vis):
    
#     fen = sample['fen']
#     vis_board = sample['vis_board']
#     condition = sample['condition']
    
#     if vis:
#         prefix = f"You are given a chess position in FEN: {fen}.\nThe visual representation of the chessboard is as follows:\n{vis_board}"
#     else:
#         prefix = f"You are given a chess position in FEN: {fen}.\n"
    
#     question = f"Find the best move for the current position. Denote the move in UCI format.\nThe legal moves are as follows:\n{condition}\n"
#     instruction = f"Let's think step by step and output the final answer in <answer> </answer> tags.\n"
#     examples = f"Example final answers: <answer>e2e4</answer>, <answer>g2f1b</answer>, <answer>e8g8</answer>, etc.\n"

#     return prefix + question + instruction + examples


def get_prompt_legal_any(sample):
    
    fen = sample['fen']
    pieces = sample['pieces']
    condition = sample['condition']
    
    prefix = f"You are given a chess position in FEN: {fen}.\nThe piece arrangement is as follows:\n{pieces}\n"
    
    question = f"Find a legal move for the piece on the square {condition}. Denote the move in UCI format.\n"
    instruction = f"Let's think step by step and output the final answer in <answer> </answer> tags.\n"
    examples = f"Example final answers: <answer>e2e4</answer>, <answer>g2f1b</answer>, <answer>e8g8</answer>, etc.\n"
    
    return prefix + question + instruction + examples


def get_prompt_legal_all(sample):
    
    fen = sample['fen']
    pieces = sample['pieces']
    condition = sample['condition']
    
    prefix = f"You are given a chess position in FEN: {fen}.\nThe piece arrangement is as follows:\n{pieces}\n"
    
    question = f"Find all the legal moves for the piece on the square {condition}. Denote the moves in UCI format, separated by commas.\n"
    instruction = f"Let's think step by step and output the final answer in <answer> </answer> tags.\n"
    examples = f"Example final answers: <answer>e2e4, e2e3</answer>, <answer>g2f1b, g2f1q, g2f1r</answer>, <answer>e8g8, e8f8</answer>, etc.\n"

    return prefix + question + instruction + examples


def get_prompt_legal_left(sample):
    
    fen = sample['fen']
    pieces = sample['pieces']
    condition = sample['condition'].split(' ')
    legal_moves = condition[1:]
    square = condition[0]
    
    prefix = f"You are given a chess position in FEN: {fen}.\nThe piece arrangement is as follows:\n{pieces}\n"
    
    question = f"Find one more legal move for the piece on the square {square} other than the following moves: {', '.join(legal_moves)}. Denote the move in UCI format.\n"
    instruction = f"Let's think step by step and output the final answer in <answer> </answer> tags.\n"
    examples = f"Example final answers: <answer>e2e4</answer>, <answer>g2f1b</answer>, <answer>e8g8</answer>, etc.\n"
    return prefix + question + instruction + examples
    


def get_prompt_piece(sample):
    
    fen = sample['fen']
    pieces = sample['pieces']
    condition = sample['condition']
    
    prefix = f"You are given a chess position in FEN: {fen}.\nThe piece arrangement is as follows:\n{pieces}\n"
    
    question = f"Find the piece on the square {condition}.\nRespond with the piece color and type or 'Empty' if there is no piece.\n"
    instruction = f"Let's think step by step and output the final answer in <answer> </answer> tags.\n"
    examples = f"Example final answers: <answer>White King</answer>, <answer>Black Queen</answer>, <answer>Empty</answer>, etc.\n"
    
    return prefix + question + instruction + examples


# def get_prompt_matein1_wo(sample, vis):
    
#     fen = sample['fen']
#     vis_board = sample['vis_board']
    
#     if vis:
#         prefix = f"You are given a chess position in FEN: {fen}.\nThe visual representation of the chessboard is as follows:\n{vis_board}"
#     else:
#         prefix = f"You are given a chess position in FEN: {fen}.\n"
    
#     question = f"Find the best move for the current position to checkmate the opponent in one move. Denote the move in UCI format.\n"
#     instruction = f"Let's think step by step and output the final answer in <answer> </answer> tags.\n"
#     examples = f"Example final answers: <answer>e2e4</answer>, <answer>g2f1</answer>, <answer>c8a6</answer>, etc.\n"

#     return prefix + question + instruction + examples


def get_prompt_matein1_w(sample):
    
    fen = sample['fen']
    pieces = sample['pieces']
    condition = sample['condition']
    
    prefix = f"You are given a chess position in FEN: {fen}.\nThe piece arrangement is as follows:\n{pieces}\n"
    
    question = f"Find the best move for the current position to checkmate the opponent in one move. Denote the move in UCI format.\nThe legal moves are as follows:\n{condition}\n"
    instruction = f"Let's think step by step and output the final answer in <answer> </answer> tags.\n"
    examples = f"Example final answers: <answer>e2e4</answer>, <answer>g2f1</answer>, <answer>c8a6</answer>, etc.\n"

    return prefix + question + instruction + examples


def dataset_mapper(sample, split):
    
    data_source = sample['data_source']

    if data_source in ["chess_best_wo_train", "chess_best_wo_test"]:
        prompt = get_prompt_best_wo(sample)
    elif data_source in ["chess_best_w_train", "chess_best_w_test"]:
        prompt = get_prompt_best_w(sample)
    elif data_source in ["chess_legal_any_train", "chess_legal_any_test"]:
        prompt = get_prompt_legal_any(sample)
    elif data_source in ["chess_legal_all_train", "chess_legal_all_test"]:
        prompt = get_prompt_legal_all(sample)
    elif data_source in ["chess_legal_left_train", "chess_legal_left_test"]:
        prompt = get_prompt_legal_left(sample)
    elif data_source in ["chess_piece_train", "chess_piece_test"]:
        prompt = get_prompt_piece(sample)
    elif data_source in ["chess_matein1_wo_train", "chess_matein1_wo_test"]:
        prompt = get_prompt_matein1_wo(sample)
    elif data_source in ["chess_matein1_w_train", "chess_matein1_w_test"]:
        prompt = get_prompt_matein1_w(sample)
    else:
        raise ValueError("Unknown data source")


    inputs = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": prompt,
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


# def get_chess_best_wo_dataset(args):

#     dataset_train = []
#     dataset_test = []
#     counter = 0
    
#     with open(args.puzzle_path, 'r') as f:
#         f.readline()
#         for line in tqdm(f):
#             line = line.strip().split(',')
#             fen = line[1]
#             board = chess.Board(fen)
#             moves = line[2].split(' ')
#             pre_move = moves[0]
#             board.push(chess.Move.from_uci(pre_move))
            
#             fen = board.fen()
#             vis_board = fen_to_board(fen)
#             best_move = moves[1]
            
#             if counter < args.train_size:
#                 dataset_train.append(['chess_best_wo_train', fen, vis_board, "placeholder", best_move])
#             else:
#                 dataset_test.append(['chess_best_wo_test', fen, vis_board, "placeholder", best_move])
#             counter += 1
            
#             if counter >= args.train_size + args.test_size:
#                 break
    
#     dataset_train = pd.DataFrame(dataset_train, columns=['data_source', 'fen', 'vis_board', 'condition', 'ground_truth'])
#     dataset_test = pd.DataFrame(dataset_test, columns=['data_source', 'fen', 'vis_board', 'condition', 'ground_truth'])

#     return dataset_train, dataset_test


# def get_chess_best_w_dataset(args):
    
#     dataset_train = []
#     dataset_test = []
#     counter = 0
    
#     with open(args.puzzle_path, 'r') as f:
#         f.readline()
#         for line in tqdm(f):
#             line = line.strip().split(',')
#             fen = line[1]
#             board = chess.Board(fen)
#             moves = line[2].split(' ')
#             pre_move = moves[0]
#             board.push(chess.Move.from_uci(pre_move))
            
#             fen = board.fen()
#             vis_board = fen_to_board(fen)
#             best_move = moves[1]
            
#             legal_moves = board.legal_moves
#             legal_moves_list = [move.uci() for move in legal_moves]
            
#             assert best_move in legal_moves_list, f"Best move {best_move} not in legal moves {legal_moves_list}!"
            
#             if counter < args.train_size:
#                 dataset_train.append(['chess_best_w_train', fen, vis_board, ' '.join(legal_moves_list), best_move])
#             else:
#                 dataset_test.append(['chess_best_w_test', fen, vis_board, ' '.join(legal_moves_list), best_move])
#             counter += 1
            
#             if counter >= args.train_size + args.test_size:
#                 break
    
#     dataset_train = pd.DataFrame(dataset_train, columns=['data_source', 'fen', 'vis_board', 'condition', 'ground_truth'])
#     dataset_test = pd.DataFrame(dataset_test, columns=['data_source', 'fen', 'vis_board', 'condition', 'ground_truth'])
    
#     return dataset_train, dataset_test


def get_chess_legal_any_dataset(args):
    
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
            
            pieces = fen_to_pieces(fen)
            random_square = random.choice(list(legal_moves_per_square.keys()))
            
            if counter < args.train_size:
                dataset_train.append(['chess_legal_any_train', fen, json.dumps(pieces), random_square, ' '.join(legal_moves_per_square[random_square])])
            else:
                dataset_test.append(['chess_legal_any_test', fen, json.dumps(pieces), random_square, ' '.join(legal_moves_per_square[random_square])])
            counter += 1
            
            if counter >= args.train_size + args.test_size:
                break

    dataset_train = pd.DataFrame(dataset_train, columns=['data_source', 'fen', 'pieces', 'condition', 'ground_truth'])
    dataset_test = pd.DataFrame(dataset_test, columns=['data_source', 'fen', 'pieces', 'condition', 'ground_truth'])
    
    return dataset_train, dataset_test


def get_chess_legal_all_dataset(args):
    
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
                
                if move.promotion is not None:
                    move_type = "promotion"
                elif board.is_en_passant(move):
                    move_type = "en_passant"
                elif board.is_castling(move):
                    move_type = "castling"
                else:
                    move_type = "regular"
                
                from_square = chess.square_name(move.from_square)
                if from_square not in legal_moves_per_square:
                    legal_moves_per_square[from_square] = []
                legal_moves_per_square[from_square].append({"move": move.uci(), "type": move_type})
            
            pieces = fen_to_pieces(fen)
            
            for square, moves in legal_moves_per_square.items():
                if len(moves) > 1:
                    for move in moves:
                        if move['type'] in ["promotion", "en_passant", "castling"]:
                            if counter < args.train_size:
                                dataset_train.append(['chess_legal_all_train', 
                                                      fen, 
                                                      json.dumps(pieces), 
                                                      square, 
                                                      ' '.join([mv['move'] for mv in moves])])
                                regular_square = random.choice(list(legal_moves_per_square.keys()))
                                dataset_train.append(['chess_legal_all_train',
                                                        fen, 
                                                        json.dumps(pieces), 
                                                        regular_square, 
                                                        ' '.join([mv['move'] for mv in legal_moves_per_square[regular_square]])])
                            else:
                                dataset_test.append(['chess_legal_all_test', 
                                                     fen, 
                                                     json.dumps(pieces), 
                                                     square, 
                                                     ' '.join([mv['move'] for mv in moves])])
                                regular_square = random.choice(list(legal_moves_per_square.keys()))
                                dataset_test.append(['chess_legal_all_test',
                                                        fen, 
                                                        json.dumps(pieces), 
                                                        regular_square, 
                                                        ' '.join([mv['move'] for mv in legal_moves_per_square[regular_square]])])
                            counter += 2
                            if counter >= args.train_size + args.test_size:
                                break
            if counter >= args.train_size + args.test_size:
                break
            
    dataset_train = pd.DataFrame(dataset_train, columns=['data_source', 'fen', 'pieces', 'condition', 'ground_truth'])
    dataset_test = pd.DataFrame(dataset_test, columns=['data_source', 'fen', 'pieces', 'condition', 'ground_truth'])
    
    return dataset_train, dataset_test


def get_chess_legal_left_dataset(args):
    
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
                
                if move.promotion is not None:
                    move_type = "promotion"
                elif board.is_en_passant(move):
                    move_type = "en_passant"
                elif board.is_castling(move):
                    move_type = "castling"
                else:
                    move_type = "regular"
                
                from_square = chess.square_name(move.from_square)
                if from_square not in legal_moves_per_square:
                    legal_moves_per_square[from_square] = []
                legal_moves_per_square[from_square].append({"move": move.uci(), "type": move_type})
            
            pieces = fen_to_pieces(fen)
            
            for square, moves in legal_moves_per_square.items():
                if len(moves) > 1:
                    for move in moves:
                        if move['type'] in ["promotion", "en_passant", "castling"]:
                            if counter < args.train_size:
                                dataset_train.append(['chess_legal_left_train', 
                                                      fen, 
                                                      json.dumps(pieces), 
                                                      ' '.join([square, ' '.join([mv['move'] for mv in moves if mv != move])]), 
                                                      move['move']])
                                regular_square = random.choice(list(legal_moves_per_square.keys()))
                                regular_move = random.choice(legal_moves_per_square[regular_square])
                                dataset_train.append(['chess_legal_left_train',
                                                        fen, 
                                                        json.dumps(pieces), 
                                                        ' '.join([regular_square, ' '.join([mv['move'] for mv in legal_moves_per_square[regular_square] if mv != regular_move])]), 
                                                        regular_move['move']])
                            else:
                                dataset_test.append(['chess_legal_left_test', 
                                                     fen, 
                                                     json.dumps(pieces), 
                                                     ' '.join([square, ' '.join([mv['move'] for mv in moves if mv != move])]), 
                                                     move['move']])
                                regular_square = random.choice(list(legal_moves_per_square.keys()))
                                regular_move = random.choice(legal_moves_per_square[regular_square])
                                dataset_test.append(['chess_legal_left_test',
                                                        fen, 
                                                        json.dumps(pieces), 
                                                        ' '.join([regular_square, ' '.join([mv['move'] for mv in legal_moves_per_square[regular_square] if mv != regular_move])]), 
                                                        regular_move['move']])
                            counter += 2
                            if counter >= args.train_size + args.test_size:
                                break
            if counter >= args.train_size + args.test_size:
                break
    
    dataset_train = pd.DataFrame(dataset_train, columns=['data_source', 'fen', 'pieces', 'condition', 'ground_truth'])
    dataset_test = pd.DataFrame(dataset_test, columns=['data_source', 'fen', 'pieces', 'condition', 'ground_truth'])
    
    return dataset_train, dataset_test


def get_chess_piece_dataset(args):
    
    dataset_train = []
    dataset_test = []
    counter = 0
    
    with open(args.puzzle_path, 'r') as f:
        f.readline()
        for line in tqdm(f):
            line = line.strip().split(',')
            fen = line[1]
            pieces = fen_to_pieces(fen)
            
            random_name = random.choice(list(pieces.keys()))
            random_square = random.choice(pieces[random_name])
            
            if counter < args.train_size:
                dataset_train.append(['chess_piece_train', fen, json.dumps(pieces), random_square, random_name.split(' (')[0]])
            else:
                dataset_test.append(['chess_piece_test', fen, json.dumps(pieces), random_square, random_name.split(' (')[0]])
            counter += 1
            
            if counter >= args.train_size + args.test_size:
                break
    
    dataset_train = pd.DataFrame(dataset_train, columns=['data_source', 'fen', 'pieces', 'condition', 'ground_truth'])
    dataset_test = pd.DataFrame(dataset_test, columns=['data_source', 'fen', 'pieces', 'condition', 'ground_truth'])
    
    return dataset_train, dataset_test


# def get_chess_matein1_wo_dataset(args):
    
#     dataset_train = []
#     dataset_test = []
#     counter = 0
    
#     with open(args.puzzle_path, 'r') as f:
#         f.readline()
#         for line in tqdm(f):
#             line = line.strip().split(',')
#             fen = line[1]
#             board = chess.Board(fen)
#             moves = line[2].split(' ')
#             pre_move = moves[0]
#             board.push(chess.Move.from_uci(pre_move))
            
#             fen = board.fen()
#             # vis_board = fen_to_board(fen)
#             pieces = fen_to_pieces(fen)
#             best_move = moves[1]
            
#             board.push(chess.Move.from_uci(best_move))
#             if board.is_checkmate():
#                 if counter < args.train_size:
#                     dataset_train.append(['chess_matein1_wo_train', fen, vis_board, "placeholder", best_move])
#                 else:
#                     dataset_test.append(['chess_matein1_wo_test', fen, vis_board, "placeholder", best_move])
#                 counter += 1
            
#             if counter >= args.train_size + args.test_size:
#                 break
    
#     dataset_train = pd.DataFrame(dataset_train, columns=['data_source', 'fen', 'vis_board', 'condition', 'ground_truth'])
#     dataset_test = pd.DataFrame(dataset_test, columns=['data_source', 'fen', 'vis_board', 'condition', 'ground_truth'])
    
#     return dataset_train, dataset_test


def get_chess_matein1_w_dataset(args):
    
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
            # vis_board = fen_to_board(fen)
            pieces = fen_to_pieces(fen)
            best_move = moves[1]
            
            legal_moves = board.legal_moves
            legal_moves_list = [move.uci() for move in legal_moves]
            
            assert best_move in legal_moves_list, f"Best move {best_move} not in legal moves {legal_moves_list}!"
            
            board.push(chess.Move.from_uci(best_move))
            if board.is_checkmate():
                if counter < args.train_size:
                    dataset_train.append(['chess_matein1_w_train', fen, json.dumps(pieces), ' '.join(legal_moves_list), best_move])
                else:
                    dataset_test.append(['chess_matein1_w_test', fen, json.dumps(pieces), ' '.join(legal_moves_list), best_move])
                counter += 1
            
            if counter >= args.train_size + args.test_size:
                break
    
    dataset_train = pd.DataFrame(dataset_train, columns=['data_source', 'fen', 'pieces', 'condition', 'ground_truth'])
    dataset_test = pd.DataFrame(dataset_test, columns=['data_source', 'fen', 'pieces', 'condition', 'ground_truth'])
    
    return dataset_train, dataset_test


def create_datasets(cfg):
    
    # dataset_best_wo_train, dataset_best_wo_test = get_chess_best_wo_dataset(cfg)
    # dataset_best_w_train, dataset_best_w_test = get_chess_best_w_dataset(cfg)
    dataset_legal_any_train, dataset_legal_any_test = get_chess_legal_any_dataset(cfg)
    dataset_legal_all_train, dataset_legal_all_test = get_chess_legal_all_dataset(cfg)
    dataset_legal_left_train, dataset_legal_left_test = get_chess_legal_left_dataset(cfg)
    dataset_piece_train, dataset_piece_test = get_chess_piece_dataset(cfg)
    # dataset_matein1_wo_train, dataset_matein1_wo_test = get_chess_matein1_wo_dataset(cfg)
    dataset_matein1_w_train, dataset_matein1_w_test = get_chess_matein1_w_dataset(cfg)
    
    dataset_train = pd.concat([
                            # dataset_best_wo_train,
                            # dataset_best_w_train,
                            dataset_legal_any_train,
                            dataset_legal_all_train,
                            dataset_legal_left_train,
                            dataset_piece_train,
                            # dataset_matein1_wo_train,
                            dataset_matein1_w_train
                            ], axis=0).reset_index(drop=True).reset_index()
    dataset_test = pd.concat([
                            # dataset_best_wo_test,
                            # dataset_best_w_test,
                            dataset_legal_any_test,
                            dataset_legal_all_test,
                            dataset_legal_left_test,
                            dataset_piece_test,
                            # dataset_matein1_wo_test,
                            dataset_matein1_w_test
                            ], axis=0).reset_index(drop=True).reset_index()

    dataset_train = dataset_train.apply(lambda x: dataset_mapper(x, "train"), axis=1)
    dataset_test = dataset_test.apply(lambda x: dataset_mapper(x, "test"), axis=1)
    
    dataset_train = Dataset.from_list(dataset_train['inputs'].tolist())
    dataset_test = Dataset.from_list(dataset_test['inputs'].tolist())
    
    dataset_train.to_parquet(os.path.join(cfg.local_dir, 'train.parquet'))
    dataset_test.to_parquet(os.path.join(cfg.local_dir, 'test.parquet'))


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./input_data/chess_multi')
    parser.add_argument('--train_size', type=int, default=1024*16)
    parser.add_argument('--test_size', type=int, default=512)
    parser.add_argument('--puzzle_path', type=str, default='./source_data/lichess_db_puzzle.csv')
    return parser.parse_args()


if __name__ == "__main__":

    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)
    
    create_datasets(cfg)
    
    