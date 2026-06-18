# LingBot-VA Table30 Lint Evaluation

Minimal code export for RoboChallenge Table30 v2 task `lint_roller_remove_dirt` (`task_id=f31b`).

## Links

- Checkpoint: https://huggingface.co/Yak9Ce3teeh/lingbot-va-table30/tree/main/lint/step1000
- Code: https://github.com/AnonChongqing/lingbot-va-table30

## Checkpoint Layout

The evaluation config expects the merged checkpoint at:

```text
/root/autodl-tmp/checkpoints/lingbot-va-table30-lint-step1000
```

with these subdirectories:

```text
assets
text_encoder
tokenizer
transformer
vae
```

## Start Inference Server

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/envs/lingbotva

cd /root/projects/lingbot-va-table30-export-clean

export TOKENIZERS_PARALLELISM=false
export TORCH_DISABLE_ADDR2LINE=1

CUDA_VISIBLE_DEVICES=0 \
NGPU=1 \
CONFIG_NAME=table30_lint_eval \
MASTER_PORT=29670 \
PORT=29536 \
LOG_RANK=0 \
bash script/run_launch_va_server_sync.sh \
  --save_root /root/autodl-tmp/eval_out/table30_lint_step1000
```

Health check:

```bash
curl http://127.0.0.1:29536/healthz
```

## Run RoboChallenge Worker

```bash
cd /root/projects/lingbot-va-table30-export-clean

python tools/run_robochallenge_table30_lint.py \
  --user-token "$USER_TOKEN" \
  --submission-id "$SUBMISSION_ID" \
  --policy-host 127.0.0.1 \
  --policy-port 29536
```

The worker converts RoboChallenge ALOHA observations to LingBot-VA observations and converts the model's 30-channel action output to the `(N, 14)` joint action format expected by RoboChallenge.

## RoboChallenge Submission

Use these fields when creating the model on RoboChallenge:

```text
Base Model Name: lingbot-va-table30-lint-step1000
Checkpoint Link: https://huggingface.co/Yak9Ce3teeh/lingbot-va-table30/tree/main/lint/step1000
Code Link: https://github.com/AnonChongqing/lingbot-va-table30
```
