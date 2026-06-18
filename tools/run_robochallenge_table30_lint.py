import argparse
import io
import logging
import pickle
import time
import uuid
from typing import Any

import numpy as np
import requests
from PIL import Image

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import (
    WebsocketClientPolicy,
)


BASE_URL = "http://api.robochallenge.cn"
ACTIVE_STATES = {"assigned", "prepare", "ready", "running"}

PROMPT = "Use a lint roller to remove the debris from the clothing."


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


class InterfaceClient:
    def __init__(self, user_token: str):
        self.user_token = user_token
        self.session = requests.Session()
        self.robot_url = ""

    def _headers(self) -> dict[str, str]:
        return {"x-user-id": self.user_token}

    def get_json(self, url: str, **kwargs: Any) -> Any:
        response = self.session.get(url, headers=self._headers(), timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()

    def post_json(self, url: str, **kwargs: Any) -> requests.Response:
        response = self.session.post(url, headers=self._headers(), timeout=30, **kwargs)
        response.raise_for_status()
        return response

    def get_all_runs(self, submission_id: str) -> list[dict[str, Any]]:
        return self.get_json(f"{BASE_URL}/v2/job_collections/submission/{submission_id}/runs")

    def get_all_jobs(self, run_id: str) -> dict[str, Any]:
        return self.get_json(f"{BASE_URL}/job_collections/{run_id}")

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        return self.get_json(f"{BASE_URL}/jobs/{job_id}")

    def update_robot(self, robot_id: str) -> None:
        self.robot_url = f"{BASE_URL}/robots/{robot_id}/direct"

    def start_robot(self, job_id: str) -> requests.Response:
        return self.post_json(f"{BASE_URL}/jobs/update", json={"job_id": job_id, "action": "start"})

    def wait_for_running(self, job_id: str, timeout_s: int = 600) -> bool:
        start = time.time()
        while time.time() - start < timeout_s:
            status = self.get_job_status(job_id).get("status")
            logging.info("job %s status=%s", job_id, status)
            if status == "running":
                return True
            if status not in {"prepare", "ready"}:
                return False
            time.sleep(2)
        return False

    def get_state(self, image_size: tuple[int, int], image_type: list[str], action_type: str) -> dict[str, Any] | None:
        response = self.session.get(
            f"{self.robot_url}/state.pkl",
            params={
                "width": image_size[0],
                "height": image_size[1],
                "image_type": image_type,
                "action_type": action_type,
            },
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        data = pickle.loads(response.content)
        if isinstance(data, dict) and data.get("status") == "size_none":
            logging.warning("robot state not ready: %s", data)
            return None
        return data

    def post_actions(self, actions: list[list[float]], duration: float, action_type: str) -> None:
        req_hash = f"gpu-server-{uuid.uuid4()}"
        response = self.post_json(
            f"{self.robot_url}/action?hash={req_hash}",
            params={"action_type": action_type},
            json={"actions": actions, "duration": duration},
        )
        body = response.json()
        if body.get("result") != "success":
            raise RuntimeError(f"robot rejected actions: {body}")


def png_to_rgb(png_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))


def state_to_model_obs(state: dict[str, Any], prompt: str) -> dict[str, Any]:
    images = state["images"]
    obs = {
        "observation.images.cam_high": png_to_rgb(images["cam_high"]),
        "observation.images.cam_left_wrist": png_to_rgb(images["cam_left_wrist"]),
        "observation.images.cam_right_wrist": png_to_rgb(images["cam_right_wrist"]),
    }

    joint_state = np.asarray(state["action"], dtype=np.float32).reshape(-1)
    full_state = np.zeros((30, 1, 1), dtype=np.float32)
    if joint_state.shape[0] >= 14:
        full_state[14:20, 0, 0] = joint_state[0:6]
        full_state[28, 0, 0] = joint_state[6]
        full_state[21:27, 0, 0] = joint_state[7:13]
        full_state[29, 0, 0] = joint_state[13]

    return {
        "obs": [obs],
        "state": full_state,
        "prompt": prompt,
    }


def model_action_to_joint_actions(model_action: np.ndarray, max_steps: int) -> list[list[float]]:
    action = np.asarray(model_action, dtype=np.float32)
    if action.ndim != 3 or action.shape[0] < 30:
        raise ValueError(f"expected model action shape (30, F, H), got {action.shape}")

    points = action.transpose(1, 2, 0).reshape(-1, action.shape[0])
    points = points[:max_steps]
    joint = np.concatenate(
        [
            points[:, 14:20],
            points[:, 28:29],
            points[:, 21:27],
            points[:, 29:30],
        ],
        axis=1,
    )
    return joint.astype(float).tolist()


class LingBotPolicy:
    def __init__(self, host: str, port: int, prompt: str, max_action_steps: int):
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.prompt = prompt
        self.max_action_steps = max_action_steps
        self.reset(prompt)

    def reset(self, prompt: str) -> None:
        self.prompt = prompt or self.prompt
        self.client.infer({"reset": True, "prompt": self.prompt})

    def infer(self, state: dict[str, Any], prompt: str | None = None) -> list[list[float]]:
        if prompt and prompt != self.prompt:
            self.reset(prompt)
        obs = state_to_model_obs(state, self.prompt)
        out = self.client.infer(obs)
        return model_action_to_joint_actions(out["action"], self.max_action_steps)


def pick_active_run(runs: list[dict[str, Any]], current_run_id: str | None) -> dict[str, Any] | None:
    if current_run_id:
        for run in runs:
            if run.get("run_id") == current_run_id and run.get("status") in ACTIVE_STATES:
                return run
    for status in ["prepare", "ready", "running", "assigned"]:
        for run in runs:
            if run.get("status") == status:
                return run
    return None


def process_job(
    client: InterfaceClient,
    policy: LingBotPolicy,
    job: dict[str, Any],
    image_size: tuple[int, int],
    image_type: list[str],
    action_type: str,
    duration: float,
    prompt: str,
    max_job_seconds: int,
) -> None:
    job_id = job["job_id"]
    robot_id = (job.get("device") or {}).get("robot_id")
    if not robot_id:
        logging.warning("ready job %s has no robot_id yet", job_id)
        return

    client.update_robot(robot_id)
    response = client.start_robot(job_id)
    logging.info("started robot for job %s: %s", job_id, response.text)
    if not client.wait_for_running(job_id):
        logging.warning("job %s did not enter running state", job_id)
        return

    policy.reset(prompt)
    start = time.time()
    while time.time() - start < max_job_seconds:
        job_status = client.get_job_status(job_id).get("status")
        if job_status != "running":
            logging.info("job %s left running state: %s", job_id, job_status)
            break
        state = client.get_state(image_size, image_type, action_type)
        if not state or state.get("state") != "normal" or state.get("pending_actions") != 0:
            time.sleep(0.5)
            continue
        actions = policy.infer(state, prompt=prompt)
        logging.info("posting %d actions, dim=%d", len(actions), len(actions[0]) if actions else 0)
        client.post_actions(actions, duration, action_type)


def run(args: argparse.Namespace) -> None:
    client = InterfaceClient(args.user_token)
    policy = LingBotPolicy(args.policy_host, args.policy_port, args.prompt, args.max_action_steps)

    current_run_id = None
    idle_count = 0
    while True:
        runs = client.get_all_runs(args.submission_id)
        run_info = pick_active_run(runs, current_run_id)
        if run_info is None:
            logging.info("no active run for submission %s", args.submission_id)
            idle_count += 1
            if args.exit_when_idle and idle_count >= args.max_idle_polls:
                return
            time.sleep(args.poll_interval)
            continue

        idle_count = 0
        current_run_id = run_info["run_id"]
        prompt = run_info.get("prompt") or args.prompt
        logging.info(
            "active run_id=%s task=%s robot=%s status=%s prompt=%s",
            current_run_id,
            run_info.get("task_name"),
            run_info.get("robotTag"),
            run_info.get("status"),
            prompt,
        )

        job_collection = client.get_all_jobs(current_run_id)
        jobs = job_collection.get("jobs") or []
        active_jobs = [job for job in jobs if job.get("status") in ACTIVE_STATES]
        if not active_jobs:
            time.sleep(args.poll_interval)
            continue

        for job in active_jobs:
            if job.get("status") == "ready":
                process_job(
                    client,
                    policy,
                    job,
                    tuple(args.image_size),
                    args.image_type,
                    args.action_type,
                    args.duration,
                    prompt,
                    args.max_job_seconds,
                )
        time.sleep(args.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-token", required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=29536)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument(
        "--image-type",
        nargs="+",
        default=["cam_left_wrist", "cam_right_wrist", "cam_high"],
    )
    parser.add_argument("--action-type", default="joint")
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--max-action-steps", type=int, default=8)
    parser.add_argument("--max-job-seconds", type=int, default=600)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-idle-polls", type=int, default=10)
    parser.add_argument("--exit-when-idle", action="store_true")
    parser.add_argument("--log-file", default="/root/autodl-tmp/eval_out/table30_lint_worker.log")
    args = parser.parse_args()

    setup_logging(args.log_file)
    run(args)


if __name__ == "__main__":
    main()
