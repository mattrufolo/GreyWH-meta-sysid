import torch
import math
from dataclasses import dataclass
# from lti import dlsim
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


@dataclass
class Config:
    seq_len_ctx: int = 400
    seq_len_new: int = 100
    # seq_len_patch: int = 400
    # d_model_RNN: int = 128
    # d_model_patching: int = 128
    n_layers: int = 12
    n_head: int = 12
    d_model: int = 160
    n_u: int = 1
    n_y: int = 1
    n_x: int = 5
    n_hidden: int = 32
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device_name: str = "cpu"

def get_state_after_seq_single(A, B, u, x0=None):
    T, nu = u.shape
    nx = A.shape[0]
    if x0 is None:
        x = torch.zeros(nx, 1, dtype=u.dtype, device=u.device)
    else:
        x = x0
    for t in range(T):
        x = A @ x + B @ u[t].unsqueeze(-1)
    return x 

def get_state_after_seq_single_numpy(A, B, u, x0=None):
    T, nu = u.shape
    nx = A.shape[0]
    if x0 is None:
        x = np.zeros((nx,))#, dtype=u.dtype, device=u.device)
    else:
        x = x0
    for t in range(T):
        x = A @ x + B @ u[t, :]#.unsqueeze(-1)
    return x 

def dlsim2(A, B, C, D, u, x= None):
    """
    A: (nx, nx)
    B: (nx, nu)
    C: (ny, nx)
    D: (ny, nu)
    u: (seq_len, nu)
    Returns:
        y: (seq_len, ny)
    """
    seq_len, nu = u.shape
    nx = A.shape[0]
    ny = C.shape[0]
    if x == None:
        # print('iai')
        x = torch.zeros(nx, 1, dtype=u.dtype, device=u.device)  # (nx, 1)
    Bu_seq = torch.matmul(u, B.T).unsqueeze(2)  # (seq_len, nx, 1)

    A_seq = A.unsqueeze(0).expand(seq_len, nx, nx)  # (seq_len, nx, nx)
    states = [x]
    for i in range(seq_len):
        x = torch.matmul(A_seq[i], states[-1]) + Bu_seq[i]
        # print(x.dtype)

        states.append(x)
    X = torch.cat(states[:-1], dim=1).transpose(1, 0)  # (nx, seq_len)
    Y = (C @ X.T + D @ u.T).T  # (seq_len, ny)
    return Y

        

class WHModel(nn.Module):
    def __init__(self, state_dim=5, input_dim=1, output_dim=1, hidden_dim=32):
        super().__init__()
        # Example for G1
        self.A1 = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B1 = nn.Parameter(torch.randn(state_dim, input_dim))
        self.C1 = nn.Parameter(torch.randn(output_dim, state_dim))
        self.D1 = nn.Parameter(torch.randn(output_dim, input_dim))
        # Example for static nonlinearity
        self.w1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.randn(1,hidden_dim))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.b2 = nn.Parameter(torch.randn(1,output_dim))
        # # Example for G2
        self.A2 = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B2 = nn.Parameter(torch.randn(state_dim, input_dim))
        self.C2 = nn.Parameter(torch.randn(output_dim, state_dim))
        self.D2 = nn.Parameter(torch.randn(output_dim, input_dim))


    def forward(self, input_signal, initial_state1=None,initial_state2=None):
        seq_len, input_dim = input_signal.shape
        # print(self.A1)
        # A_np = self.A1.clone().numpy()
        # B_np = self.B1.detach().to('cpu').clone().numpy()
        # C_np = self.C1.detach().to('cpu').clone().numpy()
        # D_np = self.D1.detach().to('cpu').clone().numpy()
        # Split parameters
        # print(initial_state1)
        # --- G1 simulation (vectorized over batch) ---
        def g1_single(A, B, C, D, u,initial_state=None):
            y = dlsim2(A, B, C, D, u,initial_state)

            # Normalize along the sequence (dim=0)
            # y = (y - y.mean(dim=0)) / (y.std(dim=0) + 1e-6)
            return y
        # print(initial_state1.dtype,self.A1.device)
        if initial_state1!= None:
            # print('ua')
            output_seq_G1 = g1_single(self.A1, self.B1, self.C1, self.D1, input_signal.double(),initial_state1)  # (batch_size, seq_len, 1)
        else:
            output_seq_G1 = g1_single(self.A1, self.B1, self.C1, self.D1, input_signal.double())
        def mlp_single(x, w1, b1, w2, b2):
            # x: (seq_len, 1)
            out = torch.matmul(x, w1) + b1  # (seq_len, hidden_dim)
            out = F.relu(out)
            out = torch.matmul(out, w2) + b2  # (seq_len, 1)
            return out

        output_seq_F_processed = mlp_single(output_seq_G1, self.w1, self.b1, self.w2, self.b2)  # (batch_size, seq_len, 1)

        # # --- G2 simulation (vectorized over batch) ---
        def g2_single(A, B, C, D, u,initial_state=None):
            y = dlsim2(A, B, C, D, u,initial_state)
            # y = (y - y.mean(dim=0)) / (y.std(dim=0) + 1e-6)
            return y

        final_output = g2_single(self.A2, self.B2, self.C2, self.D2, output_seq_F_processed)  # (batch_size, seq_len, 1)
        # output_seq_G1 = (output_seq_G1 - output_seq_G1.mean(axis=0)[:, None]) / (output_seq_G1.std(axis=0)[:, None] + 1e-6)
        
        return final_output

def unflatten_like(flat_tensor_batch: torch.Tensor, model: torch.nn.Module):
    """
    Args:
        flat_tensor_batch: Tensor of shape (batch_size, total_num_params)
        model: a torch.nn.Module whose parameters define the shapes to unflatten to

    Returns:
        Dict mapping param names to tensors of shape (batch_size, *param_shape)
    """
    param_shapes = [p.shape for p in model.parameters()]
    param_numels = [p.numel() for p in model.parameters()]
    param_names = [name for name, _ in model.named_parameters()]
    batch_size = flat_tensor_batch.shape[0]

    # Compute split indices for flat parameter vector
    split_indices = [0] + list(torch.cumsum(torch.tensor(param_numels), dim=0).numpy())

    # Split the flat tensor for all batch elements at once
    split_tensors = [
        flat_tensor_batch[:, split_indices[j]:split_indices[j+1]].reshape(batch_size, *param_shapes[j])
        for j in range(len(param_numels))
    ]

    # Return as a dict suitable for functional_call
    params_dict = {name: tensor.double() for name, tensor in zip(param_names, split_tensors)}
    return params_dict

# def unflatten_like(flat_tensor: torch.Tensor, model: torch.nn.Module):
#     param_shapes = [p.shape for p in model.parameters()]
#     param_numels = [p.numel() for p in model.parameters()]
#     param_names = [name for name, _ in model.named_parameters()]
    
#     split_tensors = torch.split(flat_tensor, param_numels)
#     params = {name: t.double().view(shape) for t, shape, name in zip(split_tensors, param_shapes, param_names)}
#     return params


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
        self.self_attn = SelfAttention(d_model, n_head, dropout=dropout, causal=True, bias=bias) # encoder is never causal
        
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
def check_params_for_fused(optim_groups):
    incompatible_params = []
    for group_idx, group in enumerate(optim_groups):
        for p_idx, p in enumerate(group['params']):
            # Check if floating point
            is_float = p.dtype.is_floating_point
            # Check if on supported device
            is_on_supported_device = p.device.type in ['cuda', 'xpu', 'privateuseone']

            if not is_float or not is_on_supported_device:
                incompatible_params.append((
                    group_idx, p_idx, p.shape, p.dtype, p.device
                ))

    if incompatible_params:
        print("Found parameters incompatible with fused=True:")
        for g_i, p_i, shape, dtype, device in incompatible_params:
            print(f"  Group {g_i}, Param {p_i}, shape={shape}, dtype={dtype}, device={device}")
    else:
        print("All parameters are compatible with fused=True optimizer.")


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))  # learnable query

    def forward(self, x):  # x: (batch, seq_len, d_model)
        # Compute attention scores
        # (batch, seq_len, d_model) * (d_model,) -> (batch, seq_len)
        scores = torch.matmul(x, self.q)
        alpha = F.softmax(scores, dim=1)            # (batch, seq_len)
        pooled = torch.sum(x * alpha.unsqueeze(-1), dim=1)
        return pooled  # (batch, d_model)


class TransformerEncoder(nn.Module):
    def __init__(self, config):# d_model, n_heads, n_layers, dropout=0.0, bias=False):
        super().__init__()
        self.encoder_wte_noRNN = nn.Linear(config.n_u+config.n_y, config.d_model)
        self.encoder_wpe = PositionalEncoding(config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(config.d_model, config.n_head, config.dropout, config.bias) for _ in range(config.n_layers)]
        )
        self.ln_f = LayerNorm(config.d_model, config.bias)
        # self.attention_pool = AttentionPooling(config.d_model)
        total_params =2*(config.n_x*config.n_x + config.n_x*config.n_u + config.n_y*config.n_x + config.n_y*config.n_u) + 3*config.n_hidden + 1
        # self.param_projection = nn.Linear(config.d_model, total_params)
        

        self.mlp_param = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, total_params)
        )

        # self.mlp1 = nn.Sequential(
        #     nn.Linear(config.n_u+config.n_y, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, config.d_model)
        # )
        # self.margin_head = nn.Linear(d_model, 1)
        # self.margin_head2 = nn.Linear(d_model, 1)

    def forward(self, u,y,wh, par):
        # u = (u - u.mean(axis=1)[:, None]) / (u.std(axis=1)[:, None] + 1e-6)
        # y = (y - y.mean(axis=1)[:, None]) / (y.std(axis=1)[:, None] + 1e-6)
        inp = torch.cat((y,u.float()), dim = -1).float()
        
        # x0_1 = self.mlp1(inp).mean(dim = 1).unsqueeze(-1)
        # x0_2 = self.mlp2(inp).mean(dim = 1).unsqueeze(-1)
        # print(x0_1.shape)
        tok = self.encoder_wte_noRNN(inp)
        x = self.encoder_wpe(tok)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # final layer normalization
        # print(x.shape)
        pooled_transformer_output = x.mean(dim=1)#self.attention_pool(x)# # (d_model), you don't want info for each context step
        parameters = self.mlp_param(pooled_transformer_output)
        # print("Transformer pooled output:", pooled_transformer_output.mean().item(), pooled_transformer_output.std().item())
        # print("Projected param vector:", parameters.mean().item(), parameters.std().item())

        # margin_pred = self.margin_head(parameters).squeeze(-1)
        # margin_pred2 = self.margin_head2(parameters).squeeze(-1)
        # encoded = self.param_projection(parameters)
        # bounded = torch.tanh(parameters) * 0.5
        # print(bounded)
        wh_parameters = unflatten_like(parameters,wh)
        skip = False
        # min_margin = mag_range[0]
        # wh_parameters = par
        # eigvals_A1 = torch.linalg.eigvals(wh_parameters['A1'])
        # max_A1,_ = eigvals_A1.abs().max(dim=1)
        # print(max_A1)
        # penalty = 0# ((max_A1 - 0.9).relu() ** 2).mean()
        # print(wh_parameters['A1'].dtype)
        # eigenvalues, eigenvectors = torch.linalg.eig(wh_parameters['A1'])
        # tanh_eigenvalues = torch.tanh(eigenvalues)
        # Lambda_tanh = torch.diag_embed(tanh_eigenvalues)
        # wh_parameters['A1'] = (eigenvectors @ Lambda_tanh @ eigenvectors.transpose(-2, -1))#.real
        if torch.isnan(wh_parameters['A1']).any() or torch.isnan(wh_parameters['A2']).any():
            skip = True
            # continue
        else:
            eigenvalues, eigenvectors = torch.linalg.eig(wh_parameters['A1'])
            tanh_eigenvalues = torch.tanh(eigenvalues.abs())*eigenvalues/eigenvalues.abs()
            Lambda_tanh = torch.diag_embed(tanh_eigenvalues)
            wh_parameters['A1'] = (eigenvectors @ Lambda_tanh @ torch.linalg.inv(eigenvectors)).real


            eigenvalues, eigenvectors = torch.linalg.eig(wh_parameters['A2'])
            tanh_eigenvalues = torch.tanh(eigenvalues.abs())*eigenvalues/eigenvalues.abs()
            Lambda_tanh = torch.diag_embed(tanh_eigenvalues)
            wh_parameters['A2'] = (eigenvectors @ Lambda_tanh @ torch.linalg.inv(eigenvectors)).real
        # print(wh_parameters['A1'].dtype)

        
        # eigvals_A1 = torch.linalg.eigvals(wh_parameters['A1'])      # shape: (n,) or (batch, n)
        # For *magnitude* constraint: penalize any eigenvalue with modulus < 0.5
        # max_A1, _ = eigvals_A1.abs().max(dim=1)  # (batch_size,)
        # penalties1 = (0.5 - eigvals_A1.abs()).relu()
        # penalties2 = (eigvals_A1.abs()-1).relu()
        # Optionally, sum or mean across batch and eigenvalues:
        eigen_penalty = 0#penalties2.mean()

        # margin_pred should be of shape (batch_size,) and is a learnable output
        # alpha1 = min_margin + (mag_range[1] - min_margin) * torch.sigmoid(margin_pred)  # shape: (batch_size,)
        # alpha2 = min_margin + (mag_range[1] - min_margin) * torch.sigmoid(margin_pred2)  # shape: (batch_size,)

        # Now, for each batch element, scale A1 and A2 so their spectral radius is alpha
        # wh_parameters['A1'], wh_parameters['A2'] of shape (batch_size, n, n)
        # eigvals_A1 = torch.linalg.eigvals(wh_parameters['A1'])  # (batch_size, n)
        # # eigvals_A2 = torch.linalg.eigvals(wh_parameters['A2'])  # (batch_size, n)
        # max_A1, _ = eigvals_A1.abs().max(dim=1)  # (batch_size,)
        # # max_A2, _ = eigvals_A2.abs().max(dim=1)  # (batch_size,)
        # vec_09 = (torch.ones((max_A1.shape[0],))*0.9).to(device = wh_parameters['A1'].device)
        # # # Rescale so the largest eigenvalue modulus is alpha for each batch
        # wh_parameters['A1'] = (wh_parameters['A1'] / max_A1[:, None, None])*vec_09[:, None, None]# * alpha1[:, None, None]
        # for k, v in wh_parameters.items():
        #     print(f"{k}: shape {v.shape}, mean {v.mean().item():.4f}, std {v.std().item():.4f}")

        # wh_parameters['A2'] = wh_parameters['A2'] / max_A2[:, None, None] #* alpha2[:, None, None]

        # Optionally, print the new spectral radii for debugging
        # eigvals_A1_new = torch.linalg.eigvals(wh_parameters['A1'])
        # eigvals_A2_new = torch.linalg.eigvals(wh_parameters['A2'])
        # max_A1_new, _ = eigvals_A1_new.abs().max(dim=1)
        # max_A2_new, _ = eigvals_A2_new.abs().max(dim=1)
        # print("Spectral radius after scaling:", max_A1_new, max_A2_new)
        
        return wh_parameters, eigen_penalty, skip
    
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
        # print(optim_groups)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
