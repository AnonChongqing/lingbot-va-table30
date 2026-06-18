# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_table30_lint_cfg import va_table30_lint_cfg
import os

va_table30_lint_train_cfg = EasyDict(__name__='Config: VA table30 lint train')
va_table30_lint_train_cfg.update(va_table30_lint_cfg)

va_table30_lint_train_cfg.dataset_path = '/root/autodl-tmp/datasets/lerobot/final/table30_lint_roller_remove_dirt_fi5'
va_table30_lint_train_cfg.empty_emb_path = os.path.join(va_table30_lint_train_cfg.dataset_path, 'empty_emb.pt')
va_table30_lint_train_cfg.enable_wandb = False
va_table30_lint_train_cfg.num_init_worker = 2
va_table30_lint_train_cfg.load_worker = 2
va_table30_lint_train_cfg.save_interval = 200
va_table30_lint_train_cfg.gc_interval = 50
va_table30_lint_train_cfg.cfg_prob = 0.1

va_table30_lint_train_cfg.learning_rate = 1e-5
va_table30_lint_train_cfg.beta1 = 0.9
va_table30_lint_train_cfg.beta2 = 0.95
va_table30_lint_train_cfg.weight_decay = 0.1
va_table30_lint_train_cfg.warmup_steps = 10
va_table30_lint_train_cfg.batch_size = 1
va_table30_lint_train_cfg.gradient_accumulation_steps = 8
va_table30_lint_train_cfg.num_steps = 1000
