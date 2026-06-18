import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset


LINT_GRADING_SEGMENTS = [
    (0.00, 0.15, "Hold the clothing steady with one gripper."),
    (0.15, 0.30, "Pick up the lint roller with the other gripper."),
    (0.30, 0.58, "Remove the debris from the clothing with the lint roller."),
    (0.58, 0.70, "Put down the lint roller."),
    (0.70, 1.00, "Retract the gripper."),
]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pad(values, n):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.zeros(n, dtype=np.float32)
    out[: min(n, arr.shape[0])] = arr[:n]
    return out


def state_to_30(left, right):
    out = np.zeros(30, dtype=np.float32)
    out[0:7] = pad(left.get("ee_positions", []), 7)
    out[7:14] = pad(right.get("ee_positions", []), 7)
    out[14:21] = pad(left.get("joint_positions", []), 7)
    out[21:28] = pad(right.get("joint_positions", []), 7)
    out[28] = float(left.get("gripper_width", 0.0))
    out[29] = float(right.get("gripper_width", 0.0))
    return out


def create_dataset(repo_name, robot_type, fps, overwrite):
    dst_dir = HF_LEROBOT_HOME / repo_name
    if overwrite and dst_dir.exists():
        print(f"removing existing dataset: {dst_dir}")
        shutil.rmtree(dst_dir)

    features = {
        "state": {
            "dtype": "float32",
            "shape": (30,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (30,),
            "names": ["action"],
        },
    }

    return LeRobotDataset.create(
        repo_id=repo_name,
        root=dst_dir,
        robot_type=robot_type,
        fps=fps,
        features=features,
    )


def process_episode(ep_dir: Path, dataset, frame_interval: int, prompt: str, max_frames: int | None):
    states_dir = ep_dir / "states"
    left_states = load_jsonl(states_dir / "left_states.jsonl")
    right_states = load_jsonl(states_dir / "right_states.jsonl")

    n = min(len(left_states), len(right_states))
    if max_frames is not None:
        n = min(n, max_frames)

    if n <= frame_interval:
        print(f"skip short episode {ep_dir.name}: n={n}")
        return 0

    written = 0
    for idx in range(frame_interval, n, frame_interval):
        prev_state = state_to_30(left_states[idx - frame_interval], right_states[idx - frame_interval])
        action = state_to_30(left_states[idx], right_states[idx])
        dataset.add_frame(
            {
                "state": prev_state,
                "action": action,
            },
            task=prompt,
        )
        written += 1

    dataset.save_episode()
    return written


def monotonic_bounds(length: int, ratios):
    raw = [int(round(length * r)) for r in ratios]
    raw[0] = 0
    raw[-1] = length
    bounds = [raw[0]]
    for i, value in enumerate(raw[1:], 1):
        min_value = bounds[-1] + 1 if i < len(raw) - 1 else bounds[-1]
        bounds.append(max(min_value, min(value, length)))
    bounds[-1] = length
    return bounds


def make_action_config(length: int, prompt: str, mode: str):
    if mode == "single":
        return [
            {
                "start_frame": 0,
                "end_frame": length,
                "action_text": prompt,
            }
        ]

    if mode != "lint_grading":
        raise ValueError(f"unknown action config mode: {mode}")

    ratios = [seg[0] for seg in LINT_GRADING_SEGMENTS] + [LINT_GRADING_SEGMENTS[-1][1]]
    bounds = monotonic_bounds(length, ratios)
    out = []
    for i, (_, _, text) in enumerate(LINT_GRADING_SEGMENTS):
        out.append(
            {
                "start_frame": bounds[i],
                "end_frame": bounds[i + 1],
                "action_text": text,
            }
        )
    return out


def add_action_config(repo_name: str, prompt: str, mode: str):
    meta_path = HF_LEROBOT_HOME / repo_name / "meta" / "episodes.jsonl"
    backup_path = meta_path.with_suffix(".jsonl.bak_no_action_config")
    if not backup_path.exists():
        backup_path.write_text(meta_path.read_text())

    lines = []
    for line in meta_path.read_text().splitlines():
        item = json.loads(line)
        length = int(item["length"])
        item["action_config"] = make_action_config(length, prompt, mode)
        lines.append(json.dumps(item, ensure_ascii=False))
    meta_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--raw-task-dir", required=True, type=Path)
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--action-config-mode", choices=["single", "lint_grading"], default="single")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_dir = args.raw_task_dir
    task_info = json.loads((task_dir / "meta" / "task_info.json").read_text())
    task_desc = task_info["task_desc"]
    robot_type = task_desc["task_tag"][-1]
    fps = float(task_info["video_info"]["fps"]) / args.frame_interval
    prompt = task_desc["prompt"]

    print("task:", task_dir.name)
    print("repo_name:", args.repo_name)
    print("robot_type:", robot_type)
    print("prompt:", prompt)
    print("fps:", fps)
    print("action_config:", args.action_config_mode)

    dataset = create_dataset(
        repo_name=args.repo_name,
        robot_type=robot_type,
        fps=fps,
        overwrite=args.overwrite,
    )

    episodes = sorted((task_dir / "data").glob("episode_*"))
    if args.start_episode:
        episodes = episodes[args.start_episode :]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    total_frames = 0
    for i, ep in enumerate(episodes, 1):
        episode_num = args.start_episode + i
        total = args.start_episode + len(episodes)
        print(f"[{episode_num}/{total}] {ep.name}")
        total_frames += process_episode(
            ep,
            dataset=dataset,
            frame_interval=args.frame_interval,
            prompt=prompt,
            max_frames=args.max_frames_per_episode,
        )

    add_action_config(args.repo_name, prompt, args.action_config_mode)
    print("episodes:", len(episodes))
    print("frames:", total_frames)
    print("done:", HF_LEROBOT_HOME / args.repo_name)


if __name__ == "__main__":
    main()
