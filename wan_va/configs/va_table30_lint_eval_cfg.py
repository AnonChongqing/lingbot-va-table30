# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_table30_lint_cfg import va_table30_lint_cfg

va_table30_lint_eval_cfg = EasyDict(__name__='Config: VA table30 lint eval')
va_table30_lint_eval_cfg.update(va_table30_lint_cfg)

va_table30_lint_eval_cfg.wan22_pretrained_model_name_or_path = (
    "/root/autodl-tmp/checkpoints/lingbot-va-table30-lint-step1000"
)
va_table30_lint_eval_cfg.infer_mode = 'server'
va_table30_lint_eval_cfg.host = '0.0.0.0'
va_table30_lint_eval_cfg.port = 29536
va_table30_lint_eval_cfg.save_root = '/root/autodl-tmp/eval_out/table30_lint_step1000'

# Keep evaluation lighter and closer to action-only scoring. Increase these if
# a later smoke test shows the policy is unstable.
va_table30_lint_eval_cfg.num_inference_steps = 8
va_table30_lint_eval_cfg.action_num_inference_steps = 20
va_table30_lint_eval_cfg.video_exec_step = 1
