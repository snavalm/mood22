from functools import partial
import torch
import torch.nn as nn
import network.vit
wandb_config = None

patch_size = [160, 160, 160]
max_iterations = 35000
eval_frequency = 00


train_loader_config = {"batch_size":2}
val_loader_config = {"batch_size":2}

optimizer = partial(torch.optim.AdamW, lr=1e-4, weight_decay=1e-5)
scheduler = partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=1e-3, total_steps=max_iterations)


