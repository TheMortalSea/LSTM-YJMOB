import os
import argparse
import logging
import random
import datetime
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler

from dataset import *
from model import *

path_arr = [
    './dataset/city_A_challengedata.csv.gz',
    './dataset/city_B_challengedata.csv.gz',
    './dataset/city_C_challengedata.csv.gz',
    './dataset/city_D_challengedata.csv.gz'
]

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    d = [item['d'] for item in batch]
    t = [item['t'] for item in batch]
    input_x = [item['input_x'] for item in batch]
    input_y = [item['input_y'] for item in batch]
    time_delta = [item['time_delta'] for item in batch]
    city = [item['city'] for item in batch]
    label_x = [item['label_x'] for item in batch]
    label_y = [item['label_y'] for item in batch]
    len_tensor = torch.tensor([item['len'] for item in batch])

    d_padded = pad_sequence(d, batch_first=True, padding_value=0)
    t_padded = pad_sequence(t, batch_first=True, padding_value=0)
    input_x_padded = pad_sequence(input_x, batch_first=True, padding_value=0)
    input_y_padded = pad_sequence(input_y, batch_first=True, padding_value=0)
    time_delta_padded = pad_sequence(time_delta, batch_first=True, padding_value=0)
    city_padded = pad_sequence(city, batch_first=True, padding_value=0)
    label_x_padded = pad_sequence(label_x, batch_first=True, padding_value=0)
    label_y_padded = pad_sequence(label_y, batch_first=True, padding_value=0)

    return {
        'd': d_padded,
        't': t_padded,
        'input_x': input_x_padded,
        'input_y': input_y_padded,
        'time_delta': time_delta_padded,
        'city': city_padded,
        'label_x': label_x_padded,
        'label_y': label_y_padded,
        'len': len_tensor
    }

def train(args):
    name = 'LSTM-postembedABCD_test'  # Updated for clarity

    dataset_train = TrainSet(path_arr[:])
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    device = torch.device(f'cuda:{args.cuda}')

    # Changed to LSTMLocationPredictor
    model = LSTMLocationPredictor(args.layers_num, args.embed_size, args.cityembed_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()

    best_loss = float('inf')  # Initialize best loss

    for epoch_id in range(args.epochs):
        total_epoch_loss = 0
        for batch_id, batch in enumerate(tqdm(dataloader_train)):
            batch['d'] = batch['d'].to(device)
            batch['t'] = batch['t'].to(device)
            batch['input_x'] = batch['input_x'].to(device)
            batch['input_y'] = batch['input_y'].to(device)
            batch['time_delta'] = batch['time_delta'].to(device)
            batch['city'] = batch['city'].to(device)
            batch['label_x'] = batch['label_x'].to(device)
            batch['label_y'] = batch['label_y'].to(device)
            batch['len'] = batch['len'].to(device)

            def check_range(name, tensor, max_allowed):
                if tensor.max() >= max_allowed or tensor.min() < 0:
                    print(f"[ERROR] {name} out of range!")
                    print(f"Min: {tensor.min().item()}, Max: {tensor.max().item()}, Allowed: 0–{max_allowed-1}")
                    raise ValueError(f"{name} values out of range.")

            check_range("day", batch['d'], model.embedding_layer.day_embedding.day_embedding.num_embeddings)
            check_range("time", batch['t'], model.embedding_layer.time_embedding.time_embedding.num_embeddings)
            check_range("location_x", batch['input_x'], model.embedding_layer.location_x_embedding.location_embedding.num_embeddings)
            check_range("location_y", batch['input_y'], model.embedding_layer.location_y_embedding.location_embedding.num_embeddings)
            check_range("timedelta", batch['time_delta'], model.embedding_layer.timedelta_embedding.timedelta_embedding.num_embeddings)
            check_range("city", batch['city'], model.city_embedding.city_embedding.num_embeddings)

            with autocast():
                output = model(batch['d'], batch['t'], batch['input_x'], batch['input_y'], batch['time_delta'], batch['len'], batch['city'])
                
                label = torch.stack((batch['label_x'], batch['label_y']), dim=-1)
                pred_mask = (batch['input_x'] == 201)
                pred_mask = torch.cat((pred_mask.unsqueeze(-1), pred_mask.unsqueeze(-1)), dim=-1)
                loss = criterion(output[pred_mask], label[pred_mask])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_epoch_loss += loss.detach().item()

        avg_epoch_loss = total_epoch_loss / len(dataloader_train)
        scheduler.step()

        current_time = datetime.datetime.now()
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"Epoch {epoch_id + 1}/{args.epochs}, Average Loss: {avg_epoch_loss:.4f} - NEW BEST! Model saved to {args.save_path}")
        else:
            print(f"Epoch {epoch_id + 1}/{args.epochs}, Average Loss: {avg_epoch_loss:.4f} - Best loss still: {best_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--cityembed_size', type=int, default=4)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive/best_lstm_model.pth', help='Path to save the best model (including filename)')
    args = parser.parse_args()

    set_random_seed(args.seed)
    train(args)