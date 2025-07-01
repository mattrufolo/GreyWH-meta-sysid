"""
Implementation of the Encoder-Decoder Transformer models for multi-step simulation of dynamical systems.

Partially based on:
* nanoGPT https://github.com/karpathy/nanoGPT/
* The Annotated Transformer http://nlp.seas.harvard.edu/annotated-transformer/
"""

import math
from dataclasses import dataclass
import torch.nn as nn
import torch
from torch.nn import functional as F
import metrics
import time


@dataclass
class Config:
    seq_len_ctx: int = 10_000
    seq_len_new: int = 128
    seq_len_patch: int = 400
    d_model_RNN: int = 128
    # d_model_patching: int = 128
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_u: int = 1
    n_y: int = 1
    n_x: int = 5
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device_name: str = "cpu"

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, causal=True, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         bias=bias, dropout=dropout, batch_first=True)
        self.causal = causal
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.causal:
            seq_len = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            x = self.mha(x, x, x, attn_mask=mask, is_causal=True)[0]
        else:
            x = self.mha(x, x, x, is_causal=False)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y


class CrossAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, causal=False, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         bias=bias, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)
        self.causal = causal

    def forward(self, x, mem):
        x = self.mha(x, mem, mem, is_causal=self.causal)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y


class MLP(nn.Module):

    def __init__(self, d_model, dropout=0.0, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
    

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.self_attn = SelfAttention(d_model, n_head, dropout=dropout, causal=False, bias=bias) # encoder is never causal
        
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.self_attn = SelfAttention(d_model, n_heads,
                                       dropout=dropout, causal=True, bias=bias)
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.cross_attn = CrossAttention(d_model, n_heads,
                                         dropout=dropout, causal=False, bias=bias)
        self.ln_3 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model)

    def forward(self, x, mem):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), mem)
        x = x + self.mlp(self.ln_3(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.0, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, dropout, bias) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(d_model, bias)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # final layer normalization
        return x


# import time
n_hidden = 29

def split_params(params):
    # params: (batch_size, 169)
    idx = 0
    # G1
    A1 = params[:, idx:idx+25].reshape(-1, 5, 5); idx += 25
    B1 = params[:, idx:idx+5].reshape(-1, 5, 1); idx += 5
    C1 = params[:, idx:idx+5].reshape(-1, 1, 5); idx += 5
    D1 = params[:, idx:idx+1].reshape(-1, 1, 1); idx += 1
    # F (MLP)
    w1 = params[:, idx:idx+n_hidden].reshape(-1, 1, n_hidden); idx += n_hidden
    b1 = params[:, idx:idx+n_hidden].reshape(-1, n_hidden); idx += n_hidden
    w2 = params[:, idx:idx+n_hidden].reshape(-1, n_hidden, 1); idx += n_hidden
    b2 = params[:, idx:idx+1].reshape(-1, 1); idx += 1
    # G2
    A2 = params[:, idx:idx+25].reshape(-1, 5, 5); idx += 25
    B2 = params[:, idx:idx+5].reshape(-1, 5, 1); idx += 5
    C2 = params[:, idx:idx+5].reshape(-1, 1, 5); idx += 5
    D2 = params[:, idx:idx+1].reshape(-1, 1, 1); idx += 1
    return (A1, B1, C1, D1), (w1, b1, w2, b2), (A2, B2, C2, D2)


class WHModel(nn.Module):
    def __init__(self, state_dim=5):
        super().__init__()
        self.state_dim = state_dim

    def forward(self, params_tensor, input_signal, initial_state=None):
        """
        Simulates the Wiener-Hammerstein model for a given input signal.

        Args:
            params_tensor (torch.Tensor): (batch_size, 169) tensor of WH model parameters.
            input_signal (torch.Tensor): (batch_size, seq_len, 1) input signal.
            initial_state (torch.Tensor, optional): (batch_size, state_dim, 1) initial state for G1.
                                                    Defaults to zeros.

        Returns:
            torch.Tensor: (batch_size, seq_len, 1) output signal of the WH model.
        """
        batch_size, seq_len, input_dim = input_signal.shape
        assert input_dim == 1, "Input signal to WH model must be 1-dimensional."
        assert params_tensor.shape[0] == batch_size, "Batch sizes must match."

        # Split parameters
        (A1, B1, C1, D1), (w1, b1, w2, b2), (A2, B2, C2, D2) = split_params(params_tensor)

        # Initialize state for G1
        if initial_state is None:
            x1 = torch.zeros(batch_size, self.state_dim, 1, device=input_signal.device, dtype=input_signal.dtype)
        else:
            assert initial_state.shape == (batch_size, self.state_dim, 1), "Initial state shape mismatch."
            x1 = initial_state

        output_seq_G1 = []
        output_seq_G2 = []

        # --- G1 simulation ---
        for t in range(seq_len):
            u_t = input_signal[:, t, :]  # (batch_size, 1)

            x1 = torch.bmm(A1, x1) + torch.bmm(B1, u_t.unsqueeze(-1))  # (batch_size, state_dim, 1)
            y1_t = torch.bmm(C1, x1) + torch.bmm(D1, u_t.unsqueeze(-1))  # (batch_size, 1, 1)
            output_seq_G1.append(y1_t.squeeze(-1))  # (batch_size, 1)

        output_seq_G1 = torch.stack(output_seq_G1, dim=1)  # (batch_size, seq_len, 1)
        

        # # --- Static nonlinearity (MLP) vectorized ---
        # y1_flat = output_seq_G1.reshape(batch_size * seq_len, 1)  # Flatten batch and seq

        # print(y1_flat.unsqueeze(-1).shape, w1.repeat(seq_len, 1, 1).shape)
        # output_l1 = torch.bmm(y1_flat.unsqueeze(-1), w1.repeat(seq_len, 1, 1)).squeeze(1) + b1.repeat(seq_len, 1)
        # output_l1 = torch.tanh(output_l1)

        # output_l2 = torch.bmm(output_l1.unsqueeze(1), w2.repeat(seq_len, 1, 1)).squeeze(1) + b2.repeat(seq_len, 1)

        # output_seq_F_processed = output_l2.reshape(batch_size, seq_len, 1)  # (batch_size, seq_len, 1)

        output_seq_F_processed = []
        for t in range(seq_len):
            input_mlp = output_seq_G1[:, t, :] # (batch_size, 1)

            # Layer 1
            # w1: (batch_size, 1, 32), input_mlp: (batch_size, 1)
            # output_l1: (batch_size, 32)
            output_l1 = torch.bmm(input_mlp.unsqueeze(-1), w1).squeeze(1) + b1

            output_l1 = F.tanh(output_l1) # Use ReLU as activation

            # Layer 2
            # w2: (batch_size, 32, 1), output_l1: (batch_size, 32)
            # output_l2: (batch_size, 1)
            output_l2 = torch.bmm(output_l1.unsqueeze(1), w2).squeeze(1) + b2
            output_seq_F_processed.append(output_l2)

        output_seq_F_processed = torch.stack(output_seq_F_processed, dim=1) # \

        # --- G2 simulation ---
        x2 = torch.zeros(batch_size, self.state_dim, 1, device=input_signal.device, dtype=input_signal.dtype)

        for t in range(seq_len):
            u_g2_t = output_seq_F_processed[:, t, :]  # (batch_size, 1)
            # print(u_g2_t.shape)
            x2 = torch.bmm(A2, x2) + torch.bmm(B2, u_g2_t.unsqueeze(-1))
            y2_t = torch.bmm(C2, x2) + torch.bmm(D2, u_g2_t.unsqueeze(-1))
            output_seq_G2.append(y2_t.squeeze(-1))

        final_output = torch.stack(output_seq_G2, dim=1)  # (batch_size, seq_len, 1)

        return final_output




class GreyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_len = config.seq_len_ctx//config.seq_len_patch
        print(patch_len)
        self.before_enc_wte = torch.nn.Linear(patch_len*(config.n_u+config.n_y), config.d_model_RNN)
        self.RNN = nn.RNN(config.n_u + config.n_y, config.d_model_RNN, num_layers=1, batch_first=True)#, bidirectional=True)
        self.encoder = TransformerEncoder(config.n_embd, config.n_head, config.n_layer,
                                          dropout=config.dropout, bias=config.bias)
        self.wh_model = WHModel(state_dim=config.n_x)
        self.param_projection = nn.Linear(config.n_embd, config.n_embd)

        self.encoder_wte = nn.Linear(config.d_model_RNN, config.n_embd)
        self.encoder_wte2 = nn.Linear(config.n_u+config.n_y, config.n_embd)
        self.encoder_wte_noRNN = nn.Linear(config.n_u+config.n_y, config.n_embd)
        self.encoder_wpe = PositionalEncoding(config.n_embd)
        self.decoder_wte1 = nn.Linear(config.n_u + config.n_y, config.n_embd) # it's alla about if it is commented or not!! so..seems really a seed 
        self.decoder_wte2 = nn.Linear(config.n_u, config.n_embd)#, useless
        self.decoder_wpe = PositionalEncoding(config.n_embd) # could also be the same as encoder_wpe

        self.lm_head_mean = nn.Linear(config.n_embd, config.n_y, bias=True)  # keep bias here maybe?
        self.lm_head_logvar = nn.Linear(config.n_embd, config.n_y, bias=True)  # keep bias here maybe?

    # def embed_ctx(self, y, u,u_new,y_new,n_in, n_init = 0):
    def embed_ctx(self, y, u, n_init = 0):
        max_len_enc = 400
        start_time1 = time.time()
        if u.shape[1]>max_len_enc:
            # print(u.shape)
            start = u.shape[1]%max_len_enc
            # print(start)
            u = u[:,start:,:]
            y = y[:,start:,:]
            [B,T,nu] = u.shape
            ny = y.shape[2]
            patch_len = (T-n_init)//max_len_enc
            
            yu_patch = torch.cat((y, u), dim=-1)
            # yu_patch = torch.cat((y[:,:-n_init,:], u[:,:-n_init,:]), dim=-1)
            #IF NO SKIP
            #---
            # yu_patch = yu_patch.view(B,max_len_enc, patch_len*(nu + ny))
            yu_patch = yu_patch.view(B*max_len_enc, patch_len, nu + ny)
            #yu = self.before_enc_wte(yu_patch)
            _, hn = self.RNN(yu_patch)
            # print(hn.shape)
            d_model = hn.shape[2]
            yu = hn[-1:].view(B, max_len_enc, d_model)
            tok_emb = self.encoder_wte(yu)
            #IF NO SKIP
            # yu_in = torch.cat((y_new[:,:n_in,:], u_new[:,:n_in,:]), dim=-1)
            # yu_in = torch.cat((y[:,-n_init:,:], u[:,-n_init:,:]), dim=-1)
            # tok_emb2 = self.encoder_wte2(yu_in)
            # tok_emb = torch.cat((tok_emb1, tok_emb2), dim=1)
            #---
            # print(tok_emb.shape,tok_emb2.shape, tok_emb1.shape)
        else:
            yu = torch.cat((y, u), dim=-1)
            tok_emb = self.encoder_wte_noRNN(yu)
            end_time1  = time.time()
        # print(f"outside_endcoder:{end_time1-start_time1}")
        src = self.encoder_wpe(tok_emb)
        return src

    def embed_new(self, u_new,y_new,n_in):
        ##UPDATE
        ## Need the size in order to cr,n_in)eate the vector of zeros of the length that I desire
        # the assert to not have problem with inital condition and length of the real vector, to be sure
        size_dec  = y_new.size()
        assert n_in<size_dec[1], "the number of initial condition has to be less then the elements that enter in the decoder"
        ## creation of zeros as desired from the y_new structue and initial condition
        # zeros = torch.zeros([size_dec[0],size_dec[1]-n_in,size_dec[2]])
        # torch.cat that I think is the most optimized to concatenate the CI y with zeros, and then concatenate with u_new
        # if cuda != 0:
        #     zeros = zeros.pin_memory().to(cuda, non_blocking=True)
        # y_concat = torch.cat((y_new[:,:n_in,:],zeros),dim = 31)
        # yu_new = torch.cat((u_new,y_concat), dim=-1)
        # change and use the matrix for linearity in the encoder
        yu_new = torch.cat((u_new[:,:n_in,:],y_new[:,:n_in,:]), dim=-1)
        tok_emb_new1 = self.decoder_wte1(yu_new)
        tok_emb_new2 = self.decoder_wte2(u_new[:,n_in:,:])
        tok_emb_new = torch.cat((tok_emb_new1,tok_emb_new2), dim=1)
        tgt = self.decoder_wpe(tok_emb_new)
        return tgt

    def forward(self, y, u, u_new):
        src = self.embed_ctx(y, u)  # perhaps dropout of this?
        mem = self.encoder(src)
        pooled_transformer_output = mem.mean(dim=1) # (batch_size, d_model), you don't want info for each context step
        wh_parameters = self.param_projection(pooled_transformer_output)
        output = self.wh_model(wh_parameters, u_new) #misses for now the initial state given from the initial conditions

        # u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-6)
        output = (output - output.mean(axis=0)) / (output.std(axis=0) + 1e-6)
        return output#, y_std,loss #, rmse_loss,nll
    

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = True #'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    

if __name__ == "__main__":

    B = 8
    T = 800
    n_ctx = 400
    n_new = 64
    n_u = 1
    n_y = 3
    n_layer = 12
    n_head = 4
    n_embd = 128
    d_model = 128
    device = "cuda:1"

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_y=n_y, n_u=n_u,
                      seq_len_ctx=n_ctx, seq_len_new=n_new,
                       bias=False, dropout=0.0)

    cfg = Config(**model_args)
    model = TSTransformer_noskip(cfg)
    model.to(device)
    batch_y = torch.randn((B, T+n_new, n_y))
    batch_u = torch.randn((B, T+n_new, n_u))

    batch_y_ctx = batch_y[:, :T, :]
    batch_u_ctx = batch_u[:, :T, :]

    batch_y_new = batch_y[:, T:, :]
    batch_u_new = batch_u[:, T:, :]

    batch_y_ctx = batch_y_ctx.pin_memory().to(device, non_blocking=True)
    batch_u_ctx = batch_u_ctx.pin_memory().to(device, non_blocking=True)
    batch_y_new = batch_y_new.pin_memory().to(device, non_blocking=True)
    batch_u_new = batch_u_new.pin_memory().to(device, non_blocking=True)

    # model.eval()
    batch_y_new_sim = model(batch_y_ctx, batch_u_ctx, batch_u_new,batch_y_new)