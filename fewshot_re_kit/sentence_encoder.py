import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50, 
            pos_embedding_dim=50, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        tmp_mat = np.random.randn(word_vec_mat.shape[0]+4, word_vec_mat.shape[1])
        tmp_mat[:word_vec_mat.shape[0]] = word_vec_mat

        self.embedding = network.embedding.Embedding(tmp_mat, max_length,
                word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, 
                pos_embedding_dim, hidden_size)
        self.word2id = word2id
        origin_len = len(self.word2id)
        self.word2id['[e1]'] = origin_len
        self.word2id['[/e1]'] = origin_len + 1
        self.word2id['[e2]'] = origin_len + 2
        self.word2id['[/e2]'] = origin_len + 3

    def forward(self, inputs, pool=True):
        x = self.embedding(inputs)
        x = self.encoder(x, pool=pool)
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[e1]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[e2]')
                pos2_in_index = len(tokens)

            tokens.append(token)

            if cur_pos == pos_head[-1]:
                tokens.append('[/e1]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[/e2]')
            cur_pos += 1

        indexed_tokens = []
        for token in tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length
        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, pos1, pos2, mask

    # def tokenize(self, raw_tokens, pos_head, pos_tail):
    #     # token -> index
    #     indexed_tokens = []
    #     for token in raw_tokens:
    #         token = token.lower()
    #         if token in self.word2id:
    #             indexed_tokens.append(self.word2id[token])
    #         else:
    #             indexed_tokens.append(self.word2id['[UNK]'])
    #     valid_len = len(indexed_tokens)
    #     # padding
    #     while len(indexed_tokens) < self.max_length:
    #         indexed_tokens.append(self.word2id['[PAD]'])
    #     indexed_tokens = indexed_tokens[:self.max_length]
    #
    #     # pos
    #     pos1 = np.zeros((self.max_length), dtype=np.int32)
    #     pos2 = np.zeros((self.max_length), dtype=np.int32)
    #     pos1_in_index = min(self.max_length, pos_head[0])
    #     pos2_in_index = min(self.max_length, pos_tail[0])
    #     for i in range(self.max_length):
    #         pos1[i] = i - pos1_in_index + self.max_length
    #         pos2[i] = i - pos2_in_index + self.max_length
    #
    #     # mask
    #     mask = np.zeros((self.max_length), dtype=np.int32)
    #     mask[:valid_len] = 1
    #
    #     return indexed_tokens, pos1, pos2, mask


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        _, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, pos1, pos2, mask

class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens

class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.roberta = RobertaModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def forward(self, inputs):
        _, x = self.roberta(inputs['word'], attention_mask=inputs['mask'])
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        pE1 = 0
        pE2 = 0
        pE1_ = 0
        pE2_ = 0
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
            if ins[i][1] == E1b:
                pE1 = ins[i][0] + i
            elif ins[i][1] == E2b:
                pE2 = ins[i][0] + i
            elif ins[i][1] == E1e:
                pE1_ = ins[i][0] + i
            else:
                pE2_ = ins[i][0] + i
        pos1_in_index = pE1 + 1
        pos2_in_index = pE2 + 1
        sst = ['<s>'] + sst
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(1)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(sst)] = 1

        return indexed_tokens, pos1, pos2, mask


class RobertaPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.roberta = RobertaForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def forward(self, inputs):
        x = self.roberta(inputs['word'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)
        return indexed_tokens


class DummySentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0]+4, self.word_embedding_dim,
                                           padding_idx=word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data[:word_vec_mat.shape[0]].copy_(word_vec_mat)

        self.pos_embedding = nn.Embedding(max_length, pos_embedding_dim)
        self.rnn = nn.GRU(word_embedding_dim+pos_embedding_dim,
                          hidden_size//2,
                          bidirectional=True,
                          batch_first=True,
                          num_layers=1)
        self.word2id = word2id
        origin_len = len(self.word2id)
        self.word2id['[e1]'] = origin_len
        self.word2id['[/e1]'] = origin_len + 1
        self.word2id['[e2]'] = origin_len + 2
        self.word2id['[/e2]'] = origin_len + 3

    def forward(self, inputs, pool=True):
        word = inputs['word']
        lengths = inputs['mask'].sum(-1)
        seq_length = word.shape[1]
        device = word.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(word.shape)
        x = torch.cat([self.word_embedding(word),
                       self.pos_embedding(position_ids)], 2)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=self.max_length)[0]
        if pool:
            x = x.max(dim=1)[0]
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[e1]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[e2]')
                pos2_in_index = len(tokens)

            tokens.append(token)

            if cur_pos == pos_head[-1]:
                tokens.append('[/e1]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[/e2]')
            cur_pos += 1

        indexed_tokens = []
        for token in tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length
        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, pos1, pos2, mask


class ATTNSentenceEncoder(nn.Module):
    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim,
                                           padding_idx=word_vec_mat.shape[0] - 1)
        # self.word_embedding.weight.data[:word_vec_mat.shape[0]].copy_(word_vec_mat)

        self.pos_embedding = nn.Embedding(max_length, pos_embedding_dim)
        self.attn = nn.TransformerEncoderLayer(d_model=word_embedding_dim + pos_embedding_dim,  nhead=5, dim_feedforward=256, dropout=0.1)
        self.fc = nn.Linear(word_embedding_dim + pos_embedding_dim, hidden_size)
        # self.rnn = nn.GRU(word_embedding_dim + pos_embedding_dim,
        #                   hidden_size // 2,
        #                   bidirectional=True,
        #                   batch_first=True,
        #                   num_layers=1)
        self.word2id = word2id
        # origin_len = len(self.word2id)
        # self.word2id['[e1]'] = origin_len
        # self.word2id['[/e1]'] = origin_len + 1
        # self.word2id['[e2]'] = origin_len + 2
        # self.word2id['[/e2]'] = origin_len + 3

    def forward(self, inputs, pool=True):
        word = inputs['word']
        lengths = inputs['mask'].sum(-1)
        seq_length = word.shape[1]
        device = word.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(word.shape)
        x = torch.cat([self.word_embedding(word),
                       self.pos_embedding(position_ids)], 2)

        # x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.attn(src=x.transpose(0, 1),
                      src_key_padding_mask=~inputs['mask'].bool()).transpose(0, 1)
        x = self.fc(x)
        # x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=self.max_length)[0]
        if pool:
            x = x.max(dim=1)[0]
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])
        valid_len = len(indexed_tokens)
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:valid_len] = 1

        return indexed_tokens, pos1, pos2, mask