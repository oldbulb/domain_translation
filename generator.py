import torch
from torch import nn
import torch.nn.functional as F
import math

from decoding import TokenDecoder, CopyTokenDecoder
from transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask, Embedding
from data import ListsToTensor, BOS, EOS, _back_to_txt_for_check
from search import Hypothesis, Beam, search_by_batch
from module import MonoEncoder, MemEncoder

class Generator(nn.Module):
    def __init__(self, vocabs,
                embed_dim, ff_embed_dim, num_heads, dropout,
                enc_layers, dec_layers, label_smoothing):
        super(Generator, self).__init__()
        self.vocabs = vocabs

        self.encoder = MonoEncoder(vocabs['src'], enc_layers, embed_dim, ff_embed_dim, num_heads, dropout)

        self.tgt_embed = Embedding(vocabs['tgt'].size, embed_dim, vocabs['tgt'].padding_idx)
        self.tgt_pos_embed = SinusoidalPositionalEmbedding(embed_dim)
        self.decoder = Transformer(dec_layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
        
        self.embed_scale = math.sqrt(embed_dim)
        self.self_attn_mask = SelfAttentionMask()
        self.output = TokenDecoder(vocabs, self.tgt_embed, label_smoothing)
        self.dropout = dropout

    def encode_step(self, inp):
        src_repr, src_mask = self.encoder(inp['src_tokens'])
        return src_repr, src_mask

    def prepare_incremental_input(self, step_seq):
        token = torch.from_numpy(ListsToTensor(step_seq, self.vocabs['tgt']))
        return token

    def decode_step(self, step_token, state_dict, mem_dict, offset, topk): 
        src_repr = mem_dict['encoder_state']
        src_padding_mask = mem_dict['encoder_state_mask']
        _, bsz, _ = src_repr.size()

        new_state_dict = {}

        token_repr = self.embed_scale * self.tgt_embed(step_token) + self.tgt_pos_embed(step_token, offset)
        for idx, layer in enumerate(self.decoder.layers):
            name_i = 'decoder_state_at_layer_%d'%idx
            if name_i in state_dict:
                prev_token_repr = state_dict[name_i]
                new_token_repr = torch.cat([prev_token_repr, token_repr], 0)
            else:
                new_token_repr = token_repr

            new_state_dict[name_i] = new_token_repr
            token_repr, _, _ = layer(token_repr, kv=new_token_repr, external_memories=src_repr, external_padding_mask=src_padding_mask)
        name = 'decoder_state_at_last_layer'
        if name in state_dict:
            prev_token_state = state_dict[name]
            new_token_state = torch.cat([prev_token_state, token_repr], 0)
        else:
            new_token_state = token_repr
        new_state_dict[name] = new_token_state

        LL = self.output(token_repr, None, work=True)

        def idx2token(idx, local_vocab):
            if (local_vocab is not None) and (idx in local_vocab):
                return local_vocab[idx]
            return self.vocabs['tgt'].idx2token(idx)

        topk_scores, topk_token = torch.topk(LL.squeeze(0), topk, 1) # bsz x k

        results = []
        for s, t in zip(topk_scores.tolist(), topk_token.tolist()):
            res = []
            for score, token in zip(s, t):
                res.append((idx2token(token, None), score))
            results.append(res)

        return new_state_dict, results

    @torch.no_grad()
    def work(self, data, beam_size, max_time_step, min_time_step=1):
        src_repr, src_mask = self.encode_step(data)
        mem_dict = {'encoder_state':src_repr,
                    'encoder_state_mask':src_mask}
        init_hyp = Hypothesis({}, [BOS], 0.)
        bsz = src_repr.size(1)
        beams = [ Beam(beam_size, min_time_step, max_time_step, [init_hyp]) for i in range(bsz)]
        search_by_batch(self, beams, mem_dict)
        return beams

    def forward(self, data):
        src_repr, src_mask = self.encode_step(data)
        tgt_in_repr = self.embed_scale * self.tgt_embed(data['tgt_tokens_in']) + self.tgt_pos_embed(data['tgt_tokens_in'])
        tgt_in_repr = F.dropout(tgt_in_repr, p=self.dropout, training=self.training)
        tgt_in_mask = torch.eq(data['tgt_tokens_in'], self.vocabs['tgt'].padding_idx)
        attn_mask = self.self_attn_mask(data['tgt_tokens_in'])

        tgt_out = self.decoder(tgt_in_repr,
                                  self_padding_mask=tgt_in_mask, self_attn_mask=attn_mask,
                                  external_memories=src_repr, external_padding_mask=src_mask)
        
        return self.output(tgt_out, data)

class SrcMemGenerator(nn.Module):
    """Transformer model using Src Memory which is from the retrival sentences on Src dataset. by ZHAO Ming

    Args:
        config: similar with MemGenerator
    """
    def __init__(self, vocabs,
                embed_dim, ff_embed_dim, num_heads, dropout, mem_dropout,
                enc_layers, dec_layers, mem_enc_layers, label_smoothing, use_mem_score):
        super(SrcMemGenerator, self).__init__()
        self.vocabs = vocabs

        self.encoder = MemEncoder(vocabs['src'], enc_layers, embed_dim, ff_embed_dim, num_heads, dropout)

        self.tgt_embed = Embedding(vocabs['tgt'].size, embed_dim, vocabs['tgt'].padding_idx)
        self.tgt_pos_embed = SinusoidalPositionalEmbedding(embed_dim)
        self.decoder = Transformer(dec_layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
        
        self.mem_encoder = MonoEncoder(vocabs['src'], mem_enc_layers, embed_dim, ff_embed_dim, num_heads, mem_dropout)
        
        self.embed_scale = math.sqrt(embed_dim)
        self.self_attn_mask = SelfAttentionMask()
        self.output = TokenDecoder(vocabs, self.tgt_embed, label_smoothing)
        self.dropout = dropout

        if use_mem_score:
            self.mem_bias_scale = nn.Parameter(torch.ones(1))
            self.mem_bias_base = nn.Parameter(torch.zeros(1))
        self.use_mem_score = use_mem_score

    def encode_step(self, inp):

        mem_repr, mem_mask = self.mem_encoder(inp['all_mem_tokens'])
        # mem_repr -> seq_len x ( num_mem_sents * bsz) x dim
        # mem_mask -> seq_len x ( num_mem_sents * bsz)
        seq_len, _, dim = mem_repr.size()
        bsz = inp['src_tokens'].size(1)
        mem_repr = mem_repr.view(-1, bsz, dim)
        mem_mask = mem_mask.view(-1, bsz)

        if self.use_mem_score:
            attn_bias = inp['all_mem_scores'] * self.mem_bias_scale + self.mem_bias_base
            attn_bias = attn_bias.view(1, -1, bsz).expand(seq_len, -1, bsz).reshape(-1, bsz)
        else:
            attn_bias = None
        
        src_repr, src_mask = self.encoder(inp['src_tokens'],
                                    mem_repr=mem_repr,
                                    mem_mask=mem_mask,
                                    mem_attn_bias=attn_bias)
                                        
        return src_repr, src_mask

    def prepare_incremental_input(self, step_seq):
        token = torch.from_numpy(ListsToTensor(step_seq, self.vocabs['tgt']))
        return token

    def decode_step(self, step_token, state_dict, mem_dict, offset, topk): 
        src_repr = mem_dict['encoder_state']
        src_padding_mask = mem_dict['encoder_state_mask']

        _, bsz, _ = src_repr.size()

        new_state_dict = {}

        token_repr = self.embed_scale * self.tgt_embed(step_token) + self.tgt_pos_embed(step_token, offset)
        for idx, layer in enumerate(self.decoder.layers):
            name_i = 'decoder_state_at_layer_%d'%idx
            if name_i in state_dict:
                prev_token_repr = state_dict[name_i]
                new_token_repr = torch.cat([prev_token_repr, token_repr], 0)
            else:
                new_token_repr = token_repr

            new_state_dict[name_i] = new_token_repr
            token_repr, _, _ = layer(token_repr, kv=new_token_repr, external_memories=src_repr, external_padding_mask=src_padding_mask)
        name = 'decoder_state_at_last_layer'
        if name in state_dict:
            prev_token_state = state_dict[name]
            new_token_state = torch.cat([prev_token_state, token_repr], 0)
        else:
            new_token_state = token_repr
        new_state_dict[name] = new_token_state

        LL = self.output(token_repr, None, work=True)

        def idx2token(idx, local_vocab):
            if (local_vocab is not None) and (idx in local_vocab):
                return local_vocab[idx]
            return self.vocabs['tgt'].idx2token(idx)

        topk_scores, topk_token = torch.topk(LL.squeeze(0), topk, 1) # bsz x k

        results = []
        for s, t in zip(topk_scores.tolist(), topk_token.tolist()):
            res = []
            for score, token in zip(s, t):
                res.append((idx2token(token, None), score))
            results.append(res)

        return new_state_dict, results

    @torch.no_grad()
    def work(self, data, beam_size, max_time_step, min_time_step=1):
        src_repr, src_mask = self.encode_step(data)
        mem_dict = {'encoder_state':src_repr,
                    'encoder_state_mask':src_mask}
        init_hyp = Hypothesis({}, [BOS], 0.)
        bsz = src_repr.size(1)
        beams = [ Beam(beam_size, min_time_step, max_time_step, [init_hyp]) for i in range(bsz)]
        search_by_batch(self, beams, mem_dict)
        return beams

    def forward(self, data):
        src_repr, src_mask = self.encode_step(data)
        tgt_in_repr = self.embed_scale * self.tgt_embed(data['tgt_tokens_in']) + self.tgt_pos_embed(data['tgt_tokens_in'])
        tgt_in_repr = F.dropout(tgt_in_repr, p=self.dropout, training=self.training)
        tgt_in_mask = torch.eq(data['tgt_tokens_in'], self.vocabs['tgt'].padding_idx)
        attn_mask = self.self_attn_mask(data['tgt_tokens_in'])

        tgt_out = self.decoder(tgt_in_repr,
                                  self_padding_mask=tgt_in_mask, self_attn_mask=attn_mask,
                                  external_memories=src_repr, external_padding_mask=src_mask)
        
        return self.output(tgt_out, data)


