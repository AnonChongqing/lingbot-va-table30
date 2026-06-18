import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wan_va.modules.utils import (
    load_text_encoder,
    load_tokenizer,
    load_vae,
)


CAM_MAP = {
    "observation.images.cam_high": "cam_high_rgb.mp4",
    "observation.images.cam_left_wrist": "cam_left_wrist_rgb.mp4",
    "observation.images.cam_right_wrist": "cam_right_wrist_rgb.mp4",
}


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def encode_text(text, tokenizer, text_encoder, dtype, device, max_sequence_length=226):
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    enc_device = next(text_encoder.parameters()).device
    with torch.no_grad():
        embeds = text_encoder(
            text_input_ids.to(enc_device),
            mask.to(enc_device),
        ).last_hidden_state
    embeds = embeds.to(dtype=dtype, device=device)
    embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
    embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in embeds
        ],
        dim=0,
    )
    return embeds[0].cpu()


def normalize_latents(latents, latents_mean, inv_latents_std):
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    inv_latents_std = inv_latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    return ((latents.float() - latents_mean) * inv_latents_std).to(latents)


def read_video_segment(video_path: Path, frame_ids, height: int, width: int):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        for sampled_frame_id in frame_ids:
            raw_frame_id = int(sampled_frame_id) * 5
            cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_id)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read {video_path} at raw frame {raw_frame_id}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()

    arr = np.stack(frames)
    tensor = torch.from_numpy(arr).float().permute(3, 0, 1, 2)
    tensor = F.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).unsqueeze(0)
    return tensor, arr.shape[1], arr.shape[2]


def encode_video_tensor(video, vae, dtype, device):
    video = video / 255.0 * 2.0 - 1.0
    with torch.no_grad():
        enc = vae.encode(video.to(device=device, dtype=dtype))
        mu = enc.latent_dist.mean
        latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device)
        latents_std = torch.tensor(vae.config.latents_std, device=mu.device)
        mu_norm = normalize_latents(mu, latents_mean, 1.0 / latents_std)
    latent = mu_norm[0]
    latent = rearrange(latent, "c f h w -> (f h w) c").to(torch.bfloat16).cpu()
    return latent, int(mu_norm.shape[2]), int(mu_norm.shape[3]), int(mu_norm.shape[4])


def make_frame_ids(start_frame: int, end_frame: int):
    # action_config ranges are already in sampled 6fps LeRobot frame indices.
    return list(range(start_frame, end_frame))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", required=True, type=Path)
    parser.add_argument("--raw-task-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/checkpoints/lingbot-va-base", type=Path)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("lerobot_root:", args.lerobot_root)
    print("raw_task_dir:", args.raw_task_dir)
    print("checkpoint:", args.checkpoint)
    print("device:", device)
    print("dtype:", dtype)

    vae = load_vae(args.checkpoint / "vae", torch_dtype=dtype, torch_device=device).eval()
    text_encoder = load_text_encoder(
        args.checkpoint / "text_encoder", torch_dtype=dtype, torch_device=device
    ).eval()
    tokenizer = load_tokenizer(args.checkpoint / "tokenizer")

    empty_emb_path = args.lerobot_root / "empty_emb.pt"
    if args.overwrite or not empty_emb_path.exists():
        empty_emb = encode_text("", tokenizer, text_encoder, dtype=dtype, device=device)
        torch.save(empty_emb, empty_emb_path)
        print("saved:", empty_emb_path)

    episodes = read_jsonl(args.lerobot_root / "meta" / "episodes.jsonl")
    if args.start_episode:
        episodes = episodes[args.start_episode :]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    for episode in tqdm(episodes, desc="episodes"):
        ep_idx = int(episode["episode_index"])
        chunk = ep_idx // 1000
        raw_ep = args.raw_task_dir / "data" / f"episode_{ep_idx:06d}"

        for acfg in episode["action_config"]:
            start_frame = int(acfg["start_frame"])
            end_frame = int(acfg["end_frame"])
            text = acfg["action_text"]
            frame_ids = make_frame_ids(start_frame, end_frame)
            if not frame_ids:
                continue
            text_emb = encode_text(text, tokenizer, text_encoder, dtype=dtype, device=device)

            for key, video_name in CAM_MAP.items():
                out_dir = args.lerobot_root / "latents" / f"chunk-{chunk:03d}" / key
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"episode_{ep_idx:06d}_{start_frame}_{end_frame}.pth"
                if out_path.exists() and not args.overwrite:
                    continue

                video, ori_h, ori_w = read_video_segment(
                    raw_ep / "videos" / video_name,
                    frame_ids,
                    height=args.height,
                    width=args.width,
                )
                latent, latent_f, latent_h, latent_w = encode_video_tensor(
                    video,
                    vae=vae,
                    dtype=dtype,
                    device=device,
                )
                torch.save(
                    {
                        "latent": latent,
                        "latent_num_frames": latent_f,
                        "latent_height": latent_h,
                        "latent_width": latent_w,
                        "video_num_frames": len(frame_ids),
                        "video_height": ori_h,
                        "video_width": ori_w,
                        "text_emb": text_emb,
                        "text": text,
                        "frame_ids": frame_ids,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "fps": 6,
                        "ori_fps": 30,
                    },
                    out_path,
                )


if __name__ == "__main__":
    main()
