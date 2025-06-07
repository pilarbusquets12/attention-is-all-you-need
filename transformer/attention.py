# transformer implementation based on Aladdin Persson's tutorial
# https://www.youtube.com/watch?v=U0s0f995w14


import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        # the embedding is divided into heads number of parts
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # int

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(
            heads * self.head_dim, embed_size
        )  # fully connected layer

        # Node: heads* self.head_dim = embed_size

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # number of training examples sent at the same time
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads number of parts
        # Before: (N, value_len, embed_size)
        # After: (N, value_len, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Apply learnt linear transformations to the values, keys, and queries
        #    Q = input × W_q
        #    K = input × W_k
        #    V = input × W_v
        values = self.values(values)  # what info to extract from the values
        keys = self.keys(keys)  # what to attend to
        queries = self.queries(queries)  # what to look for

        # Batch matrix multiplication (energy = QK^T)
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)

        # Node: nqhd = (N, query_len, heads, head_dim)
        # Node: nkhd = (N, key_len, heads, head_dim)
        # Node: nhqk = (N, heads, query_len, key_len)

        # (for each word in the target sentence, how much should we pay attention to
        # each word in the source sentence)
        # query len = target source sentence
        # key len = source sentence

        # Masking
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            # the mask is a triangular matrix
            # -1e20 = - infinity
            # used mainly in decoder prevent looking ahead at future tokens.

        # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V,
        # where d_k is the dimension of the key vectors (embed_dim)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)
        # dim = 3 means normalize across the last dimension (keys)

        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values])
        # attention shape : (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out shape: (N, query_len, heads, heads_dim)

        # Conceptually: For each query position, compute a weighted average of all
        # value vectors

        # Concatenation of heads: flatten last two dimensions
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # Linear layer
        out = self.fc_out(out)  # maps embed_size to embed_size
        return out


class TransformerBlock(nn.Module):
    """
    Transformer Block Architecture:

    ┌─────────────────────────────────────┐
    │                                     │
    │  ┌─────────────────────────────┐    │
    │  │        Add & Norm           │    │
    │  └─────────────────────────────┘    │
    │                 ↑                   │
    │  ┌─────────────────────────────┐    │
    │  │      Feed Forward           │    │
    │  │   (Linear → ReLU → Linear)  │    │
    │  └─────────────────────────────┘    │
    │                 ↑                   │
    │  ┌─────────────────────────────┐    │
    │  │        Add & Norm           │    │
    │  └─────────────────────────────┘    │
    │                 ↑                   │
    │  ┌─────────────────────────────┐    │
    │  │    Multi-Head Attention     │    │
    │  │   ┌─────┐ ┌─────┐ ┌─────┐   │    │
    │  │   │  Q  │ │  K  │ │  V  │   │    │
    │  │   └─────┘ └─────┘ └─────┘   │    │
    │  └─────────────────────────────┘    │
    │                 ↑                   │
    └─────────────────┼───────────────────┘
                      │
                   Input

    Residual connections (→) bypass each sub-layer
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        # layerNorm normalizes the input across the feature dimension (last dimension)

        # 1. to have a mean of 0 and a standard deviation of 1
        # normalized_x = (x - mean) / std

        # 2. Applies learnable parameters to the input  γ (scale) and β (shift)
        # Final output: γ * normalized_x + β

        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(
                embed_size, forward_expansion * embed_size
            ),  # in the paper embed_size * 4
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )  # extra computation and mapping it back to embed_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # Note: attention + query is skip connection
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    """
    Embeddin model
    """

    def __init__(
        self,
        src_vocab_size,  # number of unique tokens in vocabulary
        embed_size,  # embedding dimension size
        num_layers,  # number of transformer blocks
        heads,  # number of attention heads
        forward_expansion,  # expansion factor for feed forward network
        dropout,  # dropout rate
        device,  # device to run the model on
        max_length,  # maximum length of the input sequence
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # nn.Embedding creates matrix of size (src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # N examples sent in, with sequence length
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        # [0 1 2 ... seq_length-1] for each example

        positions = positions.to(self.device)  # TODO: understand this
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        # for each token in the input sequence (x)
        # we generate a word embedding
        # we sum the word embedding with the positional embedding
        for layer in self.layers:
            out = layer(out, out, out, mask)
            # value = key = query
        return out


class DecoderBlock(nn.Module):
    """
    Decoder model
    """

    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # target_mask: essential!
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, target_vocab_size)  # Gray
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        create a mask to prevent the model from attending to the padding tokens
        src: (N, src_len)
        src_mask: (N, 1, 1, src_len)
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N , 1, 1, src_len)
        # N = batch size
        # 1 = we want to mask out the entire batch
        # 1 = we want to mask out the entire sequence
        # src_len = length of the source sequence
        # if its a source pad index, we want to mask it out -> set to 0
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )  # triangular matrix lower (tril)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
