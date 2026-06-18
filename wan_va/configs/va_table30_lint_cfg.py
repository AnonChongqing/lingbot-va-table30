# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_table30_lint_cfg = EasyDict(__name__='Config: VA table30 lint')
va_table30_lint_cfg.update(va_shared_cfg)

va_table30_lint_cfg.wan22_pretrained_model_name_or_path = "/root/autodl-tmp/checkpoints/lingbot-va-base"
va_table30_lint_cfg.attn_window = 72
va_table30_lint_cfg.frame_chunk_size = 2
va_table30_lint_cfg.env_type = 'table30'

va_table30_lint_cfg.height = 256
va_table30_lint_cfg.width = 320
va_table30_lint_cfg.action_dim = 30
va_table30_lint_cfg.action_per_frame = 16
va_table30_lint_cfg.obs_cam_keys = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
]
va_table30_lint_cfg.guidance_scale = 5
va_table30_lint_cfg.action_guidance_scale = 1

va_table30_lint_cfg.num_inference_steps = 25
va_table30_lint_cfg.video_exec_step = -1
va_table30_lint_cfg.action_num_inference_steps = 50

va_table30_lint_cfg.snr_shift = 5.0
va_table30_lint_cfg.action_snr_shift = 1.0

va_table30_lint_cfg.used_action_channel_ids = list(range(30))
va_table30_lint_cfg.inverse_used_action_channel_ids = list(range(30))

va_table30_lint_cfg.action_norm_method = 'quantiles'
va_table30_lint_cfg.norm_stat = {
    'q01': [0.054331161081790924, -0.18856248259544373, 0.09510599821805954, -0.7347129583358765, -0.9935190677642822, -0.1204204261302948, -0.39769864082336426, 0.054589398205280304, -0.07887384295463562, 0.07090424001216888, -0.7115306854248047, -0.9908267855644226, -0.20115943253040314, -0.5160776376724243, -0.4700285792350769, 0.00291314790956676, -1.7276886701583862, -1.0381426811218262, -0.07596862316131592, -0.1475420594215393, -1.0, -0.3341049253940582, 0.0010117520578205585, -1.8186228275299072, -0.4872618615627289, -0.9222021698951721, -1.5726637840270996, -1.0, -0.0010999999940395355, -0.0005000000237487257],
    'q99': [0.41694000363349915, 0.009918062016367912, 0.3687697649002075, 0.7157775163650513, 0.9310657978057861, 0.16525515913963318, 0.7461124658584595, 0.38873377442359924, 0.30741745233535767, 0.3243957459926605, 0.7134991884231567, 0.9903830885887146, 0.14515888690948486, 0.7431222200393677, 0.07028187811374664, 2.3061840534210205, 0.0005582079757004976, 0.07404977828264236, 1.1584516763687134, 1.8589667081832886, 1.0, 0.7666114568710327, 2.2890512943267822, 0.001273412024602294, 1.143523931503296, 1.215899109840393, 0.4785063564777374, 1.0, 0.10610000044107437, 0.10400000214576721],
}
