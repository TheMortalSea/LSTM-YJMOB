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

def finetune(args):
    # City names and paths for B, C, D only
    city_names = ['B', 'C', 'D']
    city_paths = path_arr[1:4]  # Exclude city A

    # Ensure save directory exists
    os.makedirs(args.save_path, exist_ok=True)

    for city_idx, city_path in enumerate(city_paths):
        city_name = city_names[city_idx]
        print(f"\nFine-tuning on City {city_name}...")

        # Load dataset for current city
        dataset_train = TrainSet(city_path)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, 
                                    collate_fn=collate_fn, num_workers=args.num_workers)

        # Initialize model and load pre-trained weights
        device = torch.device(f'cuda:{args.cuda}')
        model = LSTMLocationPredictor(args.layers_num, args.embed_size, args.cityembed_size).to(device)
        model.load_state_dict(torch.load(args.pretrained_model))

        # Freeze all parameters except FFN layer
        for name, param in model.named_parameters():
            if 'ffn_layer' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.train()

        # Set up optimizer, scheduler, and loss function
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        best_loss = float('inf')
        city_save_path = os.path.join(args.save_path, f'city_{city_name}_finetune_model.pth')

        # Fine-tune for 5 epochs
        for epoch_id in range(args.epochs):
            total_epoch_loss = 0
            for batch_id, batch in enumerate(tqdm(dataloader_train, desc=f"City {city_name} Epoch {epoch_id + 1}")):
                batch['d'] = batch['d'].to(device)
                batch['t'] = batch['t'].to(device)
                batch['input_x'] = batch['input_x'].to(device)
                batch['input_y'] = batch['input_y'].to(device)
                batch['time_delta'] = batch['time_delta'].to(device)
                batch['city'] = batch['city'].to(device)
                batch['label_x'] = batch['label_x'].to(device)
                batch['label_y'] = batch['label_y'].to(device)
                batch['len'] = batch['len'].to(device)

                with autocast():
                    output = model(batch['d'], batch['t'], batch['input_x'], batch['input_y'], 
                                 batch['time_delta'], batch['len'], batch['city'])
                    label = torch.stack((batch['label_x'], batch['label_y']), dim=-1)
                    pred_mask = (batch['input_x'] == 201)
                    pred_mask = torch.cat((pred_mask.unsqueeze(-1), pred_mask.unsqueeze(-1)), dim=-1)
                    loss = criterion(output[pred_mask], label[pred_mask])

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_epoch_loss += loss.detach().item()

            avg_epoch_loss = total_epoch_loss / len(dataloader_train)
            scheduler.step()

            print(f"City {city_name} Epoch {epoch_id + 1}/{args.epochs}, Average Loss: {avg_epoch_loss:.4f}")

            # Save model if loss improves
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), city_save_path)
                print(f"City {city_name} - NEW BEST! Model saved to {city_save_path}")

        print(f"Finished fine-tuning on City {city_name}. Best loss: {best_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='/content/drive/MyDrive/best_lstm_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--cityembed_size', type=int, default=4)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=3704)
    parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive', help='Directory to save fine-tuned models')
    args = parser.parse_args()

    set_random_seed(args.seed)
    finetune(args)