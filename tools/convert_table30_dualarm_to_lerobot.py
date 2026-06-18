import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


CAM_MAP = {
    "observation.images.cam_high": "cam_high_rgb.mp4",
    "observation.images.cam_left_wrist": "cam_left_wrist_rgb.mp4",
    "observation.images.cam_right_wrist": "cam_right_wrist_rgb.mp4",
}


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


def read_frame(caps, video_names):
    frames = {}
    for key, cap in caps.items():
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read frame from {video_names[key]}")
        frames[key] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frames


def skip_frames(caps, num_frames):
    for cap in caps.values():
        for _ in range(num_frames):
            cap.grab()


def create_dataset(repo_name, robot_type, fps, height, width, overwrite, append_existing):
    dst_dir = HF_LEROBOT_HOME / repo_name
    if append_existing:
        if not dst_dir.exists():
            raise FileNotFoundError(f"cannot append; dataset does not exist: {dst_dir}")
        print(f"appending existing dataset: {dst_dir}")
        return LeRobotDataset(repo_id=repo_name, root=dst_dir)

    if overwrite and dst_dir.exists():
        print(f"removing existing dataset: {dst_dir}")
        shutil.rmtree(dst_dir)

    features = {
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
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
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )


def process_episode(ep_dir: Path, dataset, frame_interval: int, prompt: str, max_frames: int | None):
    states_dir = ep_dir / "states"
    videos_dir = ep_dir / "videos"

    left_states = load_jsonl(states_dir / "left_states.jsonl")
    right_states = load_jsonl(states_dir / "right_states.jsonl")

    caps = {
        key: cv2.VideoCapture(str(videos_dir / name))
        for key, name in CAM_MAP.items()
    }

    try:
        counts = {
            key: int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for key, cap in caps.items()
        }
        n = min(len(left_states), len(right_states), *counts.values())
        if max_frames is not None:
            n = min(n, max_frames)

        if n <= frame_interval:
            print(f"skip short episode {ep_dir.name}: n={n}")
            return

        for idx in range(frame_interval, n, frame_interval):
            frames = read_frame(caps, CAM_MAP)
            skip_frames(caps, frame_interval - 1)
            prev_state = state_to_30(left_states[idx - frame_interval], right_states[idx - frame_interval])
            action = state_to_30(left_states[idx], right_states[idx])

            dataset.add_frame(
                {
                    "observation.images.cam_high": frames["observation.images.cam_high"],
                    "observation.images.cam_left_wrist": frames["observation.images.cam_left_wrist"],
                    "observation.images.cam_right_wrist": frames["observation.images.cam_right_wrist"],
                    "state": prev_state,
                    "action": action,
                },
                task=prompt,
            )

        dataset.save_episode()
    finally:
        for cap in caps.values():
            cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--raw-task-dir", required=True, type=Path)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_dir = args.raw_task_dir
    task_info = json.loads((task_dir / "meta" / "task_info.json").read_text())
    task_desc = task_info["task_desc"]
    robot_type = task_desc["task_tag"][-1]
    fps = float(task_info["video_info"]["fps"]) / args.frame_interval
    prompt = task_desc["prompt"]

    first_ep = sorted((task_dir / "data").glob("episode_*"))[0]
    cap = cv2.VideoCapture(str(first_ep / "videos" / "cam_high_rgb.mp4"))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("task:", task_dir.name)
    print("repo_name:", args.repo_name)
    print("robot_type:", robot_type)
    print("prompt:", prompt)
    print("fps:", fps)
    print("image:", height, width)

    dataset = create_dataset(
        repo_name=args.repo_name,
        robot_type=robot_type,
        fps=fps,
        height=height,
        width=width,
        overwrite=args.overwrite,
        append_existing=args.append_existing,
    )

    episodes = sorted((task_dir / "data").glob("episode_*"))
    if args.start_episode:
        episodes = episodes[args.start_episode :]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    for i, ep in enumerate(episodes, 1):
        episode_num = args.start_episode + i
        total = args.start_episode + len(episodes)
        print(f"[{episode_num}/{total}] {ep.name}")
        process_episode(
            ep,
            dataset=dataset,
            frame_interval=args.frame_interval,
            prompt=prompt,
            max_frames=args.max_frames_per_episode,
        )

    if hasattr(dataset, 'consolidate'):
        dataset.consolidate(run_compute_stats=False)
    print("done:", HF_LEROBOT_HOME / args.repo_name)


if __name__ == "__main__":
    main()
