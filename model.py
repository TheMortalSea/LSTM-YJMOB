# This model is adapted from the champion work of HMPC2023, based on LP-BERT
# GitHub: https://github.com/caoji2001/LP-BERT/

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# embedding of day, timeslot, x, and y 

class DayEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    '''
    def __init__(self, embed_size):
        super(DayEmbeddingModel, self).__init__()
        self.day_embedding = nn.Embedding(
            num_embeddings=75+2,
            embedding_dim=embed_size,
        )
    def forward(self, day):
        embed = self.day_embedding(day)
        return embed

class TimeEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    '''
    def __init__(self, embed_size):
        super(TimeEmbeddingModel, self).__init__()

        self.time_embedding = nn.Embedding(
            num_embeddings=48+1,
            embedding_dim=embed_size,
        )
    def forward(self, time):
        embed = self.time_embedding(time)
        return embed

class LocationXEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    201: <MASK>
    '''
    def __init__(self, embed_size):
        super(LocationXEmbeddingModel, self).__init__()

        self.location_embedding = nn.Embedding(
            num_embeddings=200+2,
            embedding_dim=embed_size,
        )
    def forward(self, location):
        embed = self.location_embedding(location)
        return embed

class LocationYEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    201: <MASK>
    '''
    def __init__(self, embed_size):
        super(LocationYEmbeddingModel, self).__init__()

        self.location_embedding = nn.Embedding(
            num_embeddings=200+2,
            embedding_dim=embed_size,
        )
    def forward(self, location):
        embed = self.location_embedding(location)
        return embed

class TimedeltaEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    '''
    def __init__(self, embed_size):
        super(TimedeltaEmbeddingModel, self).__init__()

        self.timedelta_embedding = nn.Embedding(
            num_embeddings=48,
            embedding_dim=embed_size,
        )
    def forward(self, timedelta):
        embed = self.timedelta_embedding(timedelta)
        return embed

class CityEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(CityEmbedding, self).__init__()
        self.city_embedding = nn.Embedding(
            num_embeddings=4+1,
            embedding_dim=embed_size
        )
    def forward(self, city):
        embed = self.city_embedding(city)
        return embed
    
# sum up the embedding to form the embedding layer

class EmbeddingLayer(nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()

        self.day_embedding = DayEmbeddingModel(embed_size)
        self.time_embedding = TimeEmbeddingModel(embed_size)
        self.location_x_embedding = LocationXEmbeddingModel(embed_size)
        self.location_y_embedding = LocationYEmbeddingModel(embed_size)
        self.timedelta_embedding = TimedeltaEmbeddingModel(embed_size)

    def forward(self, day, time, location_x, location_y, timedelta):
        day_embed = self.day_embedding(day)
        time_embed = self.time_embedding(time)
        location_x_embed = self.location_x_embedding(location_x)
        location_y_embed = self.location_y_embedding(location_y)
        timedelta_embed = self.timedelta_embedding(timedelta)

        embed = day_embed + time_embed + location_x_embed + location_y_embed + timedelta_embed
        return embed

class TransformerEncoderModel(nn.Module):
    def __init__(self, layers_num, heads_num, embed_size):
        super(TransformerEncoderModel, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads_num)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=layers_num)
    def forward(self, input, src_key_padding_mask):
        out = self.transformer_encoder(input, src_key_padding_mask=src_key_padding_mask)
        return out

class FFNLayer(nn.Module):
    def __init__(self, embed_size):
        super(FFNLayer, self).__init__()

        self.ffn1 = nn.Sequential(
            nn.Linear(embed_size, 16),
            nn.ReLU(),
            nn.Linear(16, 200),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_size, 16),
            nn.ReLU(),
            nn.Linear(16, 200),
        )
    def forward(self, input):
        output_x = self.ffn1(input)
        output_y = self.ffn2(input)
        output = torch.stack([output_x, output_y], dim=-2)
        return output

class LPBERT(nn.Module):
    def __init__(self, layers_num, heads_num, embed_size, cityembed_size):
        super(LPBERT, self).__init__()

        self.embedding_layer = EmbeddingLayer(embed_size)
        self.city_embedding = CityEmbedding(cityembed_size)
        self.transformer_encoder = TransformerEncoderModel(layers_num, heads_num, embed_size)
        self.ffn_layer = FFNLayer(embed_size + cityembed_size)

    def forward(self, day, time, location_x, location_y, timedelta, len, city):
        embed = self.embedding_layer(day, time, location_x, location_y, timedelta)
        embed = embed.transpose(0, 1)

        max_len = day.shape[-1]
        indices = torch.arange(max_len, device=len.device).unsqueeze(0)
        src_key_padding_mask = ~(indices < len.unsqueeze(-1))

        transformer_encoder_output = self.transformer_encoder(embed, src_key_padding_mask)
        transformer_encoder_output = transformer_encoder_output.transpose(0, 1)

        city_embed = self.city_embedding(city)
        transformer_encoder_output = torch.cat([transformer_encoder_output, city_embed], dim=-1)

        output = self.ffn_layer(transformer_encoder_output)
        return output

class LSTMLocationPredictor(nn.Module):
    def __init__(self, layers_num, embed_size, cityembed_size, hidden_size=None, dropout=0.1):
        super(LSTMLocationPredictor, self).__init__()
        self.embedding_layer = EmbeddingLayer(embed_size)
        self.city_embedding = CityEmbedding(cityembed_size)
        
        # LSTM: input_size=embed_size, hidden_size (default to embed_size if not specified)
        hidden_size = hidden_size or embed_size
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=layers_num,
            batch_first=True,
            dropout=dropout if layers_num > 1 else 0,
            bidirectional=False
        )
        
        # FFN input matches LSTM output + city embed
        self.ffn_layer = FFNLayer(hidden_size + cityembed_size)

    def forward(self, day, time, location_x, location_y, timedelta, len_, city):
        # Embed: (batch, seq_len, embed_size)
        embed = self.embedding_layer(day, time, location_x, location_y, timedelta)
        
        # Handle padding: Pack sequences to ignore PAD in LSTM
        packed_embed = pack_padded_sequence(embed, lengths=len_.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward: packed_output, (h_n, c_n)
        packed_output, _ = self.lstm(packed_embed)
        
        # Unpack: (batch, seq_len, hidden_size)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # City embed: (batch, seq_len, cityembed_size)
        city_embed = self.city_embedding(city)
        
        # Concat LSTM output with city embed: (batch, seq_len, hidden_size + cityembed_size)
        combined = torch.cat([lstm_output, city_embed], dim=-1)
        
        # FFN: (batch, seq_len, 2, 200)
        output = self.ffn_layer(combined)
        return output