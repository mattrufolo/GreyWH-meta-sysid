from pathlib import Path
import time
import torch
import numpy as np
import math
from dataclasses import dataclass
from functools import partial
from dataset import seed_worker, WHDataset_Model
# from dataset_numba import CSTRDataset_numba
from torch.utils.data import DataLoader
from model import Config, WHModel, TransformerEncoder#, get_state_after_seq_single, dlsim2


# from grey_transformer_sim_2 import Config, GreyTransformer
from train_utils import warmup_cosine_lr
import torch.nn as nn
import tqdm
import argparse
import wandb
import copy
import sys
from torch.nn import functional as F
from torch.func import vmap, grad, functional_call

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta system identification with transformers')

    # Overall
    parser.add_argument('--model-dir', type=str, default="check_WH_over_WH", metavar='S',
                        help='Saved model folder')
    parser.add_argument('--out-file', type=str, default="test_linMLP_lr1e3_nonorm_3mil", metavar='S',
    # parser.add_argument('--out-file', type=str, default="ckpt_top32_err01", metavar='S',
                        help='Saved model name')
    parser.add_argument('--in-file', type=str, default="test_WH_more_embd", metavar='S',
                        help='Loaded model name (when resuming)')
    parser.add_argument('--init-from', type=str, default="scratch", metavar='S',
                        help='Init from (scratch|resume|pretrained)')
    parser.add_argument('--seed', type=int, default=44, metavar='N',
                        help='Seed for random number generation')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='Use wandb for data logging')

    # Dataset
    parser.add_argument('--nx', type=int, default=5, metavar='N',
                        help='Model order nx (default: 5)')
    parser.add_argument('--nu', type=int, default=1, metavar='N',
                        help='Number of inputs nu (default: 1)')
    parser.add_argument('--ny', type=int, default=1, metavar='N',
                        help='Number of outputs ny (default: 1)')
    parser.add_argument('--seq-len-skip', type=int, default=0, metavar='N',
                        help='sequence length of the separation between context and query (default: 0)')
    parser.add_argument('--seq-len-n-in', type=int, default=0, metavar='N',
                        help='sequence length of the initial condition previous to the query (default: 0)')
    parser.add_argument('--seq-len-ctx', type=int, default=400, metavar='N',
                        help='sequence length of the context sequence (default: 400)')
    parser.add_argument('--seq-len-new', type=int, default=100, metavar='N',
                        help='sequence length of the query one (default: 100)')
    parser.add_argument('--mag_range', type=tuple, default=(0.5, 0.6), metavar='N',
                        help='range of poles magnitude (default: (0.5, 0.97))')
    parser.add_argument('--phase_range', type=tuple, default=(0.0, math.pi/2), metavar='N',
                        help='range of poles phase (default: (0.0, math.pi/2))')
    parser.add_argument('--fixed-system', action='store_true', default=False,
                        help='If True, keep the same model all the times')

    # Model
    parser.add_argument('--n-layer', type=int, default=12, metavar='N',
                        help='Number of layers (default: 12)')
    parser.add_argument('--n-head', type=int, default=4, metavar='N',
                        help='Number heads (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='LR',
                        help='Dropout (default: 0.0)')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Use bias in model')

    # Training
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='Batch size (default:32)')
    parser.add_argument('--max-iters', type=int, default=3_000_100, metavar='N',
                        help='Number of iterations (default: 1000000)')
    parser.add_argument('--inner-steps', type=int, default=5, metavar='N',
                        help='Number of iterations inner optimization state (default: 30)')
    parser.add_argument('--warmup-iters', type=int, default=10_000, metavar='N',
                        help='Number of warmup iterations (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='Learning rate (default: 6e-4)')
    # parser.add_argument('--inner-lr', type=float, default=1e-2, metavar='LR',
    #                     help='Inner Learning rate (default: 6e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='D',
                        help='Optimizer weight decay (default: 0.0)')
    parser.add_argument('--n-embd', type=int, default=64, metavar='N',
                        help='Embedding size (default: 128)')
    parser.add_argument('--eval-interval', type=int, default=2000, metavar='N',
                        help='Frequency of performance evaluation (default:2000)')
    parser.add_argument('--eval-iters', type=int, default=100, metavar='N',
                        help='Number of batches used for performance evaluation')
    parser.add_argument('--fixed-lr', action='store_true', default=False,
                        help='Keep the learning rate constant, do not use cosine scheduling')

    # Compute
    parser.add_argument('--threads', type=int, default=10,
                        help='Number of CPU threads (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--cuda-device', type=str, default="cuda:1", metavar='S',
                        help='Cuda device (default: "cuda:2")')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Compile the model with torch.compile')

    try:
        
        cfg = parser.parse_args()

        # Other settings
        cfg.beta1 = 0.9
        cfg.beta2 = 0.95

        # Derived settings
        #cfg.block_size = cfg.seq_len
        cfg.lr_decay_iters = cfg.max_iters
        cfg.min_lr = cfg.lr/10.0  #
        cfg.decay_lr = not cfg.fixed_lr
        cfg.eval_batch_size = cfg.batch_size


        # HYPERPARAMETERS
        model_dir = Path(cfg.model_dir)
        model_dir.mkdir(exist_ok=True)
        
        if cfg.log_wandb:
            wandb.init(
                project="sysid-meta-WH-WH",
                name="test_linMLP_lr1e3_nonorm_3mil",
                #name="run1",
                # track hyperparameters and run metadata
                # config=vars(cfg)
            )

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed) # not needed? All randomness now handled with generators

        use_cuda = not cfg.no_cuda and torch.cuda.is_available()
        device_name = cfg.cuda_device if use_cuda else "cpu"
        device = torch.device(device_name)
        device_type = 'cuda' if 'cuda' in device_name else 'cpu'
        torch.set_float32_matmul_precision("high")

        lin_opts = dict(mag_range=cfg.mag_range, phase_range=cfg.phase_range, strictly_proper=True)
        train_ds = WHDataset_Model(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, 
                                            seq_len=cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in+cfg.seq_len_new,
                            system_seed=cfg.seed, input_seed=cfg.seed+1, noise_seed=cfg.seed+2,  
                            **lin_opts)

        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.threads, worker_init_fn=seed_worker)

        # if we work with a constant model we also validate with the same (thus same seed!)
        val_ds = WHDataset_Model(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, 
                                            seq_len=cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_skip+cfg.seq_len_n_in+cfg.seq_len_new,
                        system_seed=cfg.seed+5, input_seed=cfg.seed+6, noise_seed=cfg.seed+7, 
                        **lin_opts)

        val_dl = DataLoader(val_ds, batch_size=cfg.eval_batch_size, num_workers=cfg.threads,worker_init_fn=seed_worker)


        n_hidden = 32
            
        wh = WHModel(cfg.nx,hidden_dim=n_hidden).to(device)
        params_WH = dict(wh.named_parameters())
        flat_params_WH = nn.utils.parameters_to_vector(wh.parameters())
        n_params_WH = flat_params_WH.shape[0]

        model_args = dict(n_layers=cfg.n_layer, n_head=cfg.n_head, n_y=cfg.ny, n_u=cfg.nu, n_x = cfg.nx,
                            seq_len_ctx=cfg.seq_len_ctx, seq_len_new=cfg.seq_len_new,
                            bias=cfg.bias, dropout=cfg.dropout, device_name = cfg.cuda_device, d_model = cfg.n_embd, n_hidden = n_hidden)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if cfg.init_from == "scratch":
            gptconf = Config(**model_args)
                
        # Instantiate the hypernetwork
            hypernet = TransformerEncoder(gptconf).to(device)
            hypernet.apply(init_weights)
        elif cfg.init_from == "resume" or cfg.init_from == "pretrained":
            ckpt_path = model_dir / f"{cfg.in_file}.pt"
            checkpoint = torch.load(ckpt_path, map_location=device)
            gptconf = Config(**checkpoint["model_args"])
            hypernet = TransformerEncoder(gptconf)
            state_dict = checkpoint['model']
            hypernet.load_state_dict(state_dict)
            hypernet.to(device)
        # def hyperfan_init_linear(layer, d_out, d_in, input_var=1.0):
        #     """
        #     Initializes layer.weight using hyperfan-in from ICLR 2020.
        #     Assumes 'layer' is nn.Linear with .weight of shape (d_out, d_in).
        #     """
        #     # Standard deviation: sqrt(1 / (d_out * d_in * input_var))
        #     std = math.sqrt(1.0 / (d_out * d_in * input_var))
        #     nn.init.normal_(layer.weight, mean=0.0, std=std)
        #     if layer.bias is not None:
        #         nn.init.zeros_(layer.bias)

        # ...after creating the transformer encoder/hypernet
        # d_out = n_params_WH  # number of main WH model parameters
        # d_in = n_params_WH  # dimension of transformer output
        # hyperfan_init_linear(hypernet.param_projection, d_out, d_in)


        # def init_weights_hyperfan(m):
        #     if isinstance(m, nn.Linear):
        #         # Check if this is the param projection layer by name or shape, otherwise use Xavier
        #         if getattr(m, 'is_param_projection', False):
        #             hyperfan_init_linear(m, m.weight.shape[0], m.weight.shape[1], input_var=1.0)
        #         else:
        #             nn.init.xavier_uniform_(m.weight)
        #             if m.bias is not None:
        #                 nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     # ... other custom cases

        # # Tag the layer for clarity
        # hypernet.param_projection.is_param_projection = True
        # hypernet.apply(init_weights_hyperfan)

        
        params_hypernet = dict(hypernet.named_parameters())

        # Function that trains a single model
        # def loss_fn(p, x, y):
        #     y_hat = functional_call(wh, p, x)
        #     y = (y-y.mean(axis = 0))/(y.std(axis = 0)+1e-8)
        #     y_hat = (y_hat-y_hat.mean(axis = 0))/(y_hat.std(axis = 0)+1e-8)
        #     loss = torch.sqrt(torch.mean((y[-seq_len_new:,:] - y_hat[-seq_len_new:,:]) ** 2))
        #     return loss


        # def hypernet_loss(ph, x1, y1, x2, y2):

        #     # Generate the weights using the hypernetwork on the support set
        #     pm = functional_call(hypernet, ph, args=(x1, y1))
        #     # Compute the loss on the query set
        #     return loss_fn(pm, x2, y2)  # Loss for the second task

        # def batched_hypernet_loss(p, x1_b, y1_b, x2_b, y2_b):
        #     hypernet_loss_cfg = partial(hypernet_loss, p) # fix first argument
        #     hypernet_loss_batch = vmap(hypernet_loss_cfg) # vmap over the rest
        #     batch_losses = hypernet_loss_batch(x1_b, y1_b, x2_b, y2_b)
        #     return torch.mean(batch_losses)

        # def hypernet_loss(ph, u1, y1, u2, y2, x1, x2, wh):
        #     pm = functional_call(hypernet, ph, args=(u1, y1, wh))  # pm: (batch_size, ...)
        #     # vmap over loss_fn to handle batch of wh params and data
        #     # u = torch.cat((u1,u2), dim= 1)
        #     # y = torch.cat((y1,y2), dim= 1)
        #     batched_loss_fn = vmap(loss_fn)
        #     return torch.mean(batched_loss_fn(pm, u2, y2, x1, x2))


        def loss_fn(p, u, y, x1= None, x2= None):
            # A,B,C,D = p['A1'],p['B1'],p['C1'],p['D1']
            # # print(f'see{u.dtype}')
            # # y_true = dlsim(A, B, C, D, u.double())

            # # Cut input at step 200
            # cut = 350
            # x_cut = get_state_after_seq_single(A, B, u[:cut].double())
            # # Simulate remaining 300 steps with:
            # # (1) correct state
            # y_correct = dlsim2(A, B, C, D, u[cut:].double(), x=x_cut.double())
            # # print(y_correct.mean(axis = 0),y[200:,:].mean(axis = 0))
            # # print(y_correct.std(axis = 0),y[200:,:].std(axis = 0))
            # y_correct = (y_correct-y.mean(axis = 0))/(y.std(axis = 0)+1e-6)


            # # (2) zero state (what you're doing now)
            # y_wrong = dlsim2(A, B, C, D, u[cut:].double())
            # y_wrong = (y_wrong-y.mean(axis = 0))/(y.std(axis = 0)+1e-6)
            # y = (y-y.mean(axis = 0))/(y.std(axis = 0)+1e-6)


            # # Compare difference vs. ground truth
            # # gt_partial = y_true[cut:]

            # print("Relative L2 loss (correct x0):", torch.sqrt(torch.nn.functional.mse_loss(y_correct[-cfg.seq_len_new:,:], y[-cfg.seq_len_new:,:])))
            # print("Relative L2 loss (zero x0):", torch.sqrt(torch.nn.functional.mse_loss(y_wrong[-cfg.seq_len_new:,:],   y[-cfg.seq_len_new:,:])))
            # print(y_hat,y[-cfg.seq_len_new:,:].std())
            # print(y[-cfg.seq_len_new:,:].std(axis = 0),y_hat[-cfg.seq_len_new:,:].std(axis = 0))
            # y = (y-y.mean(axis = 0))/(y.std(axis = 0)+1e-8)
            # y_hat = (y_hat-y_hat.mean(axis = 0))/(y_hat.std(axis = 0)+1e-8)
            # loss1 = torch.sqrt(torch.mean((y[:cfg.seq_len_new,:] - y_hat[:cfg.seq_len_new,:]) ** 2))
            # loss2 = torch.sqrt(torch.mean((y[-cfg.seq_len_new:,:] - y_hat[-cfg.seq_len_new:,:]) ** 2))
            # loss3 = torch.sqrt(torch.mean((y[-cfg.seq_len_new:,:] - y_hat[-cfg.seq_len_new:,:]) ** 2))
            # loss4 = torch.sqrt(torch.mean((y[-cfg.seq_len_new:,:] - y_hat[-cfg.seq_len_new:,:]) ** 2))
            # print(loss1,loss4)
            # print(y.shape,y[:-cfg.seq_len_new,:].shape)
            y_hat = functional_call(wh, p, args = (u[-cfg.seq_len_new-50:,:]))
            # y_hat_fake = functional_call(wh, p, args = (u[:-cfg.seq_len_new,:]))
            # print(y_hat.mean(axis = 0),y_hat.std(axis = 0))
            # y_hat_fake = (y_hat_fake-y.mean(axis = 0))/(y.std(axis = 0)+1e-6)
            # y_hat = (y_hat-y_hat.mean(axis = 0))/(y_hat.std(axis = 0)+1e-6)
            # y = (y-y.mean(axis = 0))/(y.std(axis = 0)+1e-6)
            # print(y.std(axis = 0),y_hat.std(axis = 0))
            # print(y.mean(axis = 0),y_hat.mean(axis = 0))
            loss = torch.sqrt(torch.mean((y[-cfg.seq_len_new:,:] - y_hat[-cfg.seq_len_new:,:]) ** 2))
            # print(loss)
            # loss_fake = torch.sqrt(torch.mean((y[:-cfg.seq_len_new,:] - y_hat_fake) ** 2))
            return loss#,loss_fake


        
        def batched_loss_fn(pm, batch_u2,batch_y2):#,batch_x2= None
            batched_loss_fn = vmap(loss_fn)
            loss = batched_loss_fn(pm, batch_u2,batch_y2)
            # print(f'loss1 {torch.mean(loss1)}, loss2 {torch.mean(loss4)}')
            return torch.mean(loss)#,torch.mean(loss_fake)#,batch_x2



        # opt = torch.optim.Adam(params_hypernet.values(), lr=lr)
        # print(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)
        if cfg.compile:
            model = torch.compile(hypernet)  # requires PyTorch 2.0

        opt = hypernet.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)
        if cfg.init_from == "resume":
            opt.load_state_dict(checkpoint['optimizer'])

        losses = []

        @torch.no_grad()
        def estimate_loss(dl,params):
            hypernet.eval()
            loss = 0.0
            # rmse = 0.0
            for eval_iter, (batch_y, batch_u, par) in enumerate(dl):
                params_dict = {k: v.to(device) for k, v in par.items()}
                if device_type == "cuda":
                    batch_y = batch_y.pin_memory().to(device, non_blocking=True)
                    batch_u = batch_u.pin_memory().to(device, non_blocking=True)
                batch_y_enc = batch_y[:,:cfg.seq_len_ctx,:]
                batch_u_enc = batch_u[:,:cfg.seq_len_ctx,:]
                batch_y_dec = batch_y[:,cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in:,:]
                batch_u_dec = batch_u[:,cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in:,:]
                wh_params, _ , _= functional_call(hypernet, params_hypernet, args=(batch_u_enc, batch_y_enc, wh,params_dict))
                

                    # x0_2 = torch.zeros(cfg.batch_size, cfg.nx, 1, device=batch_x.device, requires_grad=True)
                    
                    # # 3. Inner optimizer just for initial states
                    # inner_opt = torch.optim.Adam([x0_1], lr=cfg.inner_lr)

                    # # 4. Inner loop: optimize x0_1 and x0_2 to fit outputs for this batch (few steps)
                    # # batched_loss_fn = vmap(loss_fn)
                    # for _ in range(cfg.inner_steps):
                    #     inner_opt.zero_grad()
                    #     val_loss = batched_loss_fn(wh_params, batch_x2, batch_y2, x0_1)
                    #     val_loss.backward()
                    #     inner_opt.step()
                val_loss = batched_loss_fn(wh_params, batch_u, batch_y)

                # else:
                #     loss_iter = hypernet_loss(params, batch_x1, batch_y1, batch_x2, batch_y2, wh)


                #     _, _, loss_iter, rmse_iter,_ = model(batch_y_enc, batch_u_enc, batch_u_dec, batch_y_dec,cfg.seq_len_n_in)
                # else:
                #     _, _, loss_iter, rmse_iter,_ = model(batch_y_enc, batch_u_enc, batch_u_dec, batch_y_dec,cfg.seq_len_n_in)


                loss += val_loss.item()
                # rmse += rmse_iter.item()
                if eval_iter == cfg.eval_iters:
                    break
            loss /= cfg.eval_iters
            #rmse/= cfg.eval_iters
            hypernet.train()
            return loss#, rmse
        get_lr = partial(warmup_cosine_lr, lr=cfg.lr, min_lr=cfg.min_lr,
                            warmup_iters=cfg.warmup_iters, lr_decay_iters=cfg.lr_decay_iters)
        LOSS_VAL = []
        SKIP_ITR = []
        LOSS_ITR = []
        loss_val = np.nan
        time_start = time.time()
       
        if cfg.init_from == "scratch" or cfg.init_from == "pretrained":
            iter_num = 0
            iter_num_new =0
            best_val_loss = np.inf
            best_val_rmse = np.inf
        elif cfg.init_from == "resume":
            iter_num = checkpoint["iter_num"]
            iter_num_new = 0
            best_val_rmse = np.inf
            best_val_loss = np.inf#checkpoint['best_val_loss']


        iter_val = 0
        # (batch_y, batch_x, par) = next(iter(train_dl))
        # if device_type == "cuda":
        #     batch_x = batch_x.pin_memory().to(device, non_blocking=True)
        #     batch_y = batch_y.pin_memory().to(device, non_blocking=True)
        for iter_num_new, (batch_y, batch_x, par) in tqdm.tqdm(enumerate(train_dl, start=iter_num_new)):
        # for iter_num_new in range(cfg.max_iters):
            params_dict = {k: v.to(device) for k, v in par.items()}
            # print(params_dict['D1'])
            if iter_num_new+iter_num==cfg.max_iters:
                break
            if ((iter_num+iter_num_new) % cfg.eval_interval == 0) and iter_num+iter_num_new > 0:# or iter_num == checkpoint["iter_num"]+1:
                print(f"\n best loss val {best_val_loss}")
                print('val_flag')
                loss_val = estimate_loss(val_dl,params_hypernet)#params_dict
                LOSS_VAL.append(loss_val)
                print(f"\n{iter_num+iter_num_new=} {loss_val=:.4f}\n")
                if loss_val < best_val_loss:
                    print("changed.pt")
                    iter_val += 1
                    best_val_loss = loss_val
                    checkpoint = {
                        'model': hypernet.state_dict(),
                        'optimizer': opt.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num+iter_num_new,
                        'train_time': time.time() - time_start,
                        'LOSS': LOSS_ITR,
                        'LOSS_VAL': LOSS_VAL,
                        'best_val_loss': best_val_loss,
                        'cfg': cfg,
                    }
                    if cfg.log_wandb:
                        wandb.log({"iter_save": iter_val})
                    torch.save(checkpoint, model_dir / f"{cfg.out_file}.pt")
            if cfg.decay_lr:
                lr_iter = get_lr(iter_num+iter_num_new)
            else:
                lr_iter = cfg.lr
            for param_group in opt.param_groups:
                    param_group['lr'] = lr_iter
            if device_type == "cuda":
                batch_x = batch_x.pin_memory().to(device, non_blocking=True)
                batch_y = batch_y.pin_memory().to(device, non_blocking=True)

            if torch.isnan(batch_y).any():
                print("Nan at iteration {iter_num}, skipping...") # wtf?
                SKIP_ITR.append(iter_num+iter_num_new)
                continue
            batch_x1 = batch_x[:, :cfg.seq_len_ctx,:]
            batch_y1 = batch_y[:, :cfg.seq_len_ctx,:]
            batch_x2 = batch_x[:, cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in-50:,:]
            batch_y2 = batch_y[:, cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in-50:,:]
            
            # 1. Forward through hypernet to get WH parameters for this batch
            wh_params, penalty,skip  = functional_call(hypernet, params_hypernet, args=(batch_x1, batch_y1, wh, params_dict))
            if skip == True:
                print("Nan at iteration {iter_num} in matrices, skipping...") # wtf?
                SKIP_ITR.append(iter_num+iter_num_new)
                continue
            # print(x0_1)
            # wh_params_detached = {k: v.detach() for k, v in wh_params.items()}
            # 2. Create new initial states for this batch (requires_grad for optimization)
            # x0_1 = torch.zeros(cfg.batch_size, cfg.nx, 1, requires_grad=True, device=device, dtype=torch.double)
            # x0_2 = torch.zeros(cfg.batch_size, cfg.nx, 1, device=batch_x.device, requires_grad=True)
            
            # 3. Inner optimizer just for initial states
            # inner_opt = torch.optim.Adam([x0_1], lr=cfg.inner_lr)
            
            # 4. Inner loop: optimize x0_1 and x0_2 to fit outputs for this batch (few steps)
            # batched_loss_fn = vmap(loss_fn)
            # for _ in range(cfg.inner_steps):
            #     # print(wh_params.grad_fn)
            #     # print(x0_1)
            #     inner_opt.zero_grad()
            #     inner_loss = batched_loss_fn(wh_params_detached, batch_x2, batch_y2, x0_1)
            #     # print(inner_loss)
            #     inner_loss.backward()
            #     inner_opt.step()
                # if _% 100 ==0:
                #     print(f"\n{_=} {inner_loss=:.4f} \n")

            

            loss = batched_loss_fn(wh_params, batch_x, batch_y)
            opt.zero_grad()
            param_loss = 0
            for i in wh_params.keys():
                # print(nn.functional.mse_loss(wh_params[i], params_dict[i]))
                param_loss += nn.functional.mse_loss(wh_params[i], params_dict[i])
            # batched_loss = vmap(loss_fn)
            # loss = torch.mean(batched_loss(par, batch_x2, batch_y2))
            
            # output = hypernet(batch_x1, batch_y1)
            # print("Model output:", output)
            real_loss = loss + 100*penalty#+param_loss
            real_loss.backward()

            params_before = {name: param.clone().detach() for name, param in hypernet.named_parameters()}
            opt.step()
            params_after = {name: param.clone().detach() for name, param in hypernet.named_parameters()}
            total_norm = 0
            for p in hypernet.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            for name, param in hypernet.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"{name} has NaN or Inf!")
            eigvals_A1 = torch.linalg.eigvals(wh_params['A1'])
            max_A1,_ = eigvals_A1.abs().max(dim=1)
            # print(f"{loss=:.4f} Gradient norm: {total_norm}")
            
            LOSS_ITR.append(loss.item())

            losses.append(loss.item())
            # print(loss)
            if (iter_num+iter_num_new) % 100 == 0:
                if cfg.log_wandb:
                    wandb.log({"param_loss":param_loss,"loss": loss, "loss_val": loss_val})
                # Compare and print which parameters changed
                if ((iter_num+iter_num_new) % cfg.eval_interval == 0) and iter_num+iter_num_new > 0:
                    for name in params_before:
                        changed = not torch.allclose(params_before[name], params_after[name], atol=1e-8)
                        diff_norm = (params_after[name] - params_before[name]).norm().item()
                        print(f"{name}: changed={changed}, L2 diff={diff_norm:.4e}")


                print(f"\n{iter_num+iter_num_new=} {max_A1=} {param_loss=:.4f} {penalty=:.4f} {loss=:.4f} {loss_val=:.4f}  {lr_iter=} Gradient norm: {total_norm}\n")
                
            #     pbar.set_postfix(loss=loss.item())

    except KeyboardInterrupt:
            print("closing gracefully")
            sys.exit()

    checkpoint = {
        'model': hypernet.state_dict(),
        'optimizer': opt.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num+iter_num_new,
        'train_time': time.time() - time_start,
        'LOSS': LOSS_ITR,
        'LOSS_VAL': LOSS_VAL,
        'best_val_loss': best_val_loss,
        'cfg': cfg,
    }
    
    torch.save(checkpoint, model_dir / f"{cfg.out_file}_last.pt")
    
    if cfg.log_wandb:
        wandb.finish()

