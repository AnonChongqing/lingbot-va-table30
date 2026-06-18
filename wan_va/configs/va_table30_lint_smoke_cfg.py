# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_table30_lint_train_cfg import va_table30_lint_train_cfg

va_table30_lint_smoke_cfg = EasyDict(__name__='Config: VA table30 lint smoke')
va_table30_lint_smoke_cfg.update(va_table30_lint_train_cfg)

va_table30_lint_smoke_cfg.load_worker = 2
va_table30_lint_smoke_cfg.num_init_worker = 1
va_table30_lint_smoke_cfg.save_interval = 999999
va_table30_lint_smoke_cfg.gradient_accumulation_steps = 1
va_table30_lint_smoke_cfg.num_steps = 2
