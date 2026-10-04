# -*- coding: utf-8 -*-

import os
HF_ENDPOINT = "https://hf-mirror.net"
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)

import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import huggingface_hub
from huggingface_hub import HfApi, CommitOperationAdd

LOCAL_FOLDER = Path(
    r"E:\pythonProject\air_groud\get_img"
    r"\dataset_uav_multimap_multicamera_mot_weather500"
)

REPO_ID = "yutiangu/dataset_uav_multimap_multicamera_mot_weather500"
REPO_TYPE = "dataset"



HF_TOKEN = ""

MAX_FILES_PER_BATCH = 300
MAX_BYTES_PER_BATCH = 512 * 1024 * 1024

BATCH_RETRIES_BEFORE_SPLIT = 5
SINGLE_FILE_MAX_RETRIES = 20
RETRY_BASE_SECONDS = 10
MAX_RETRY_WAIT_SECONDS = 300

MAX_COMMITS_PER_HOUR = 120
COMMIT_WINDOW_SECONDS = 3600
COMMIT_WINDOW_SAFETY_SECONDS = 5
RATE_LIMIT_FALLBACK_SECONDS = 3700
RATE_LIMIT_SAFETY_SECONDS = 30

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_KEY = "dataset_uav_multimap_multicamera_mot_weather500"

CHECKPOINT_FILE = SCRIPT_DIR / f"{DATASET_KEY}_completed_files.txt"
FAILED_FILE = SCRIPT_DIR / f"{DATASET_KEY}_failed_files.txt"
BATCH_HISTORY_FILE = SCRIPT_DIR / f"{DATASET_KEY}_batch_history.jsonl"
RUN_HISTORY_FILE = SCRIPT_DIR / f"{DATASET_KEY}_run_history.jsonl"
COMMIT_HISTORY_FILE = SCRIPT_DIR / "hf_commit_history.jsonl"

IGNORE_DIRS = {".git", ".cache", "__pycache__"}
IGNORE_FILES = {"Thumbs.db", "desktop.ini"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}

def format_size(size_bytes):
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def format_time(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def should_ignore(path):
    if path.name in IGNORE_FILES:
        return True
    if path.suffix.lower() in IGNORE_SUFFIXES:
        return True
    return any(part in IGNORE_DIRS for part in path.parts)

def get_status_code(error):
    response = getattr(error, "response", None)
    if response is not None:
        try:
            return response.status_code
        except Exception:
            pass
    text = str(error).lower()
    for code in (400, 401, 403, 404, 405, 409, 429, 500, 502, 503, 504):
        if f"{code} " in text or f"{code} client error" in text:
            return code
    return None

def is_rate_limit_error(error):
    if get_status_code(error) == 429:
        return True
    text = str(error).lower()
    return (
        "too many requests" in text
        or "rate limit" in text
        or "128 per hour" in text
    )

def is_fatal_error(error):
    if is_rate_limit_error(error):
        return False
    status = get_status_code(error)
    if status in {401, 403, 404, 405}:
        return True
    text = str(error).lower()
    return any(
        x in text
        for x in (
            "invalid token",
            "permission denied",
            "repository not found",
            "method not allowed",
        )
    )

def parse_retry_after_seconds(error):
    response = getattr(error, "response", None)
    if response is not None:
        try:
            retry_after = response.headers.get("Retry-After")
        except Exception:
            retry_after = None
        if retry_after:
            retry_after = retry_after.strip()
            try:
                return max(1, int(retry_after) + RATE_LIMIT_SAFETY_SECONDS)
            except (TypeError, ValueError):
                pass
            try:
                retry_dt = parsedate_to_datetime(retry_after)
                if retry_dt.tzinfo is None:
                    retry_dt = retry_dt.replace(tzinfo=timezone.utc)
                seconds = (
                    retry_dt.astimezone(timezone.utc)
                    - datetime.now(timezone.utc)
                ).total_seconds()
                return max(1, int(seconds) + RATE_LIMIT_SAFETY_SECONDS)
            except Exception:
                pass
    return RATE_LIMIT_FALLBACK_SECONDS

def sleep_with_countdown(seconds, reason):
    remaining = int(seconds)
    print()
    print(f"[WAIT] {reason}")
    print(f"[WAIT] 预计等待：{format_time(remaining)}")
    while remaining > 0:
        step = min(60, remaining)
        time.sleep(step)
        remaining -= step
        if remaining > 0:
            print(f"[WAIT] 剩余约：{format_time(remaining)}", flush=True)
    print("[WAIT] 等待结束，继续上传。", flush=True)

def load_commit_history():
    timestamps = []
    if not COMMIT_HISTORY_FILE.exists():
        return timestamps
    now = time.time()
    try:
        with open(COMMIT_HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ts = float(obj["timestamp"])
                except Exception:
                    continue
                if now - ts < COMMIT_WINDOW_SECONDS:
                    timestamps.append(ts)
    except Exception as error:
        print(f"[WARN] 读取 commit 历史失败：{error}")
    timestamps.sort()
    return timestamps

def rewrite_commit_history(timestamps):
    try:
        with open(COMMIT_HISTORY_FILE, "w", encoding="utf-8") as f:
            for ts in timestamps:
                obj = {
                    "timestamp": ts,
                    "time_utc": datetime.fromtimestamp(
                        ts, tz=timezone.utc
                    ).isoformat(),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as error:
        print(f"[WARN] 清理 commit 历史失败：{error}")

def record_successful_commit(timestamp, label, files, bytes_size, elapsed):
    append_jsonl(
        COMMIT_HISTORY_FILE,
        {
            "timestamp": timestamp,
            "time_utc": datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).isoformat(),
            "repo_id": REPO_ID,
            "label": label,
            "files": files,
            "bytes": bytes_size,
            "elapsed_seconds": elapsed,
        },
    )

def wait_if_commit_window_full():
    while True:
        history = load_commit_history()
        rewrite_commit_history(history)
        count = len(history)

        print(
            f"[RATE] 最近60分钟成功 commit："
            f"{count}/{MAX_COMMITS_PER_HOUR}"
        )

        if count < MAX_COMMITS_PER_HOUR:
            return

        oldest = history[0]
        wait_seconds = (
            COMMIT_WINDOW_SECONDS
            - (time.time() - oldest)
            + COMMIT_WINDOW_SAFETY_SECONDS
        )

        if wait_seconds <= 0:
            continue

        print()
        print("=" * 72)
        print("[RATE] 最近60分钟 commit 已达到本地安全阈值")
        print("[RATE] 不会每批固定等待，只等待真正需要的时间。")
        print("=" * 72)

        sleep_with_countdown(
            wait_seconds,
            "等待最早一次 commit 退出60分钟窗口",
        )

def load_completed_files():
    completed = set()

    if not CHECKPOINT_FILE.exists():
        print()
        print("[INFO] 当前没有本数据集的本地断点文件。")
        print("[INFO] 不会删除远端已经存在的文件。")
        return completed

    print()
    print("[INFO] 检测到本数据集断点：")
    print(CHECKPOINT_FILE)

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                completed.add(item)

    print(f"[INFO] 本地断点已记录：{len(completed):,} 个文件")
    return completed

def save_completed_files(relative_paths, completed):
    with open(
        CHECKPOINT_FILE,
        "a",
        encoding="utf-8",
        buffering=1,
    ) as f:
        for path in relative_paths:
            if path in completed:
                continue
            f.write(path + "\n")
            completed.add(path)
        f.flush()
        os.fsync(f.fileno())

def save_failed_file(relative_path, error):
    with open(
        FAILED_FILE,
        "a",
        encoding="utf-8",
        buffering=1,
    ) as f:
        message = str(error).replace("\n", " ")
        f.write(f"{relative_path}\t{message}\n")
        f.flush()
        os.fsync(f.fileno())

def scan_dataset(completed):
    print()
    print("=" * 72)
    print("扫描本地 Multicamera MOT 数据集")
    print("=" * 72)

    pending = []
    total_files = 0
    total_size = 0
    completed_files = 0
    completed_size = 0
    pending_files = 0
    pending_size = 0
    scanned = 0

    for root, dirs, files in os.walk(LOCAL_FOLDER):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)

        for filename in sorted(files):
            absolute_path = Path(root) / filename

            if should_ignore(absolute_path):
                continue

            try:
                file_size = absolute_path.stat().st_size
            except OSError as error:
                print(f"[WARN] 无法读取：{absolute_path}")
                print(error)
                continue

            relative_path = (
                absolute_path
                .relative_to(LOCAL_FOLDER)
                .as_posix()
            )

            total_files += 1
            total_size += file_size

            if relative_path in completed:
                completed_files += 1
                completed_size += file_size
            else:
                pending_files += 1
                pending_size += file_size
                pending.append(
                    (absolute_path, relative_path, file_size)
                )

            scanned += 1

            if scanned % 5000 == 0:
                print(
                    "\r"
                    f"[SCAN] 已扫描 {scanned:,} 个文件...",
                    end="",
                    flush=True,
                )

    print()
    print()
    print(f"[INFO] 总文件：{total_files:,}")
    print(f"[INFO] 总大小：{format_size(total_size)}")
    print()
    print(f"[INFO] 已完成文件：{completed_files:,}")
    print(f"[INFO] 已完成大小：{format_size(completed_size)}")
    print()
    print(f"[INFO] 待上传文件：{pending_files:,}")
    print(f"[INFO] 待上传大小：{format_size(pending_size)}")

    return {
        "pending": pending,
        "total_files": total_files,
        "total_size": total_size,
        "completed_files": completed_files,
        "completed_size": completed_size,
        "pending_files": pending_files,
        "pending_size": pending_size,
    }

def build_batches(files):
    batch = []
    batch_bytes = 0

    for item in files:
        file_size = item[2]

        if batch:
            too_many = len(batch) >= MAX_FILES_PER_BATCH
            too_large = (
                batch_bytes + file_size
                > MAX_BYTES_PER_BATCH
            )

            if too_many or too_large:
                yield batch, batch_bytes
                batch = []
                batch_bytes = 0

        batch.append(item)
        batch_bytes += file_size

    if batch:
        yield batch, batch_bytes

def count_batches(files):
    return sum(
        1
        for _batch, _size
        in build_batches(files)
    )

def split_batch(batch):
    if len(batch) <= 1:
        return batch, []

    total_size = sum(item[2] for item in batch)
    target = total_size / 2

    left = []
    right = []
    current = 0

    for item in batch:
        if current < target or not left:
            left.append(item)
            current += item[2]
        else:
            right.append(item)

    if not right:
        midpoint = len(batch) // 2
        left = batch[:midpoint]
        right = batch[midpoint:]

    return left, right

def create_operations(batch):
    return [
        CommitOperationAdd(
            path_in_repo=relative_path,
            path_or_fileobj=str(absolute_path),
        )
        for absolute_path, relative_path, _file_size
        in batch
    ]

def retry_wait(attempt):
    return min(
        RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
        MAX_RETRY_WAIT_SECONDS,
    )

def commit_once(api, batch, label):
    wait_if_commit_window_full()

    operations = create_operations(batch)
    batch_bytes = sum(item[2] for item in batch)

    start = time.time()

    result = api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message=(
            f"Upload {label} "
            f"({len(batch)} files, "
            f"{format_size(batch_bytes)})"
        ),
    )

    elapsed = time.time() - start

    record_successful_commit(
        timestamp=time.time(),
        label=label,
        files=len(batch),
        bytes_size=batch_bytes,
        elapsed=elapsed,
    )

    return result, elapsed

def show_progress(state):
    elapsed = time.time() - state["start_time"]

    done_files = (
        state["initial_completed_files"]
        + state["session_files"]
    )

    done_bytes = (
        state["initial_completed_bytes"]
        + state["session_bytes"]
    )

    total_files = state["total_files"]
    total_bytes = state["total_bytes"]

    file_pct = (
        done_files / total_files * 100
        if total_files
        else 0
    )

    byte_pct = (
        done_bytes / total_bytes * 100
        if total_bytes
        else 0
    )

    speed = (
        state["session_bytes"] / elapsed
        if elapsed > 0
        else 0
    )

    remaining = max(0, total_bytes - done_bytes)

    eta = (
        remaining / speed
        if speed > 0 and remaining > 0
        else None
    )

    recent_commits = len(load_commit_history())

    print()
    print("=" * 72)
    print(
        f"[PROGRESS] 文件："
        f"{done_files:,}/{total_files:,} "
        f"({file_pct:.2f}%)"
    )
    print(
        f"[PROGRESS] 数据："
        f"{format_size(done_bytes)} / "
        f"{format_size(total_bytes)} "
        f"({byte_pct:.2f}%)"
    )
    print(
        f"[PROGRESS] 本轮上传："
        f"{format_size(state['session_bytes'])}"
    )
    print(
        f"[PROGRESS] 平均速度："
        f"{format_size(speed)}/s"
    )
    print(
        f"[PROGRESS] 本轮运行："
        f"{format_time(elapsed)}"
    )

    if eta is not None:
        print(
            f"[PROGRESS] 预计剩余："
            f"{format_time(eta)}"
        )

    print(
        f"[PROGRESS] 最近60分钟成功 commit："
        f"{recent_commits}/{MAX_COMMITS_PER_HOUR}"
    )
    print("=" * 72)

def mark_batch_success(
    batch,
    label,
    elapsed,
    completed,
    state,
):
    relative_paths = [item[1] for item in batch]
    batch_bytes = sum(item[2] for item in batch)

    save_completed_files(
        relative_paths,
        completed,
    )

    state["session_files"] += len(batch)
    state["session_bytes"] += batch_bytes
    state["successful_commits"] += 1

    append_jsonl(
        BATCH_HISTORY_FILE,
        {
            "time_utc": now_iso(),
            "repo_id": REPO_ID,
            "label": label,
            "status": "success",
            "files": len(batch),
            "bytes": batch_bytes,
            "elapsed_seconds": elapsed,
        },
    )

    show_progress(state)

def upload_batch_adaptive(
    api,
    batch,
    label,
    completed,
    state,
    depth=0,
):
    if not batch:
        return True

    batch_bytes = sum(item[2] for item in batch)
    indent = "  " * depth

    max_normal_attempts = (
        SINGLE_FILE_MAX_RETRIES
        if len(batch) == 1
        else BATCH_RETRIES_BEFORE_SPLIT
    )

    normal_attempt = 0
    rate_limit_waits = 0
    last_error = None

    while True:
        print()
        print("-" * 72)
        print(f"{indent}[UPLOAD] {label}")
        print(f"{indent}[INFO] 文件数：{len(batch)}")
        print(
            f"{indent}[INFO] 数据量："
            f"{format_size(batch_bytes)}"
        )
        print(
            f"{indent}[INFO] 普通网络失败次数："
            f"{normal_attempt}/{max_normal_attempts}"
        )
        print(
            f"{indent}[INFO] 429等待次数："
            f"{rate_limit_waits}"
        )
        print("-" * 72)

        try:
            _result, elapsed = commit_once(
                api=api,
                batch=batch,
                label=label,
            )

            print()
            print(f"{indent}[OK] {label} 上传成功")
            print(
                f"{indent}[INFO] 本批耗时："
                f"{format_time(elapsed)}"
            )

            mark_batch_success(
                batch=batch,
                label=label,
                elapsed=elapsed,
                completed=completed,
                state=state,
            )

            return True

        except KeyboardInterrupt:
            raise

        except Exception as error:
            last_error = error

            print()
            print(
                f"{indent}[WARN] "
                f"{type(error).__name__}: {error}"
            )

            if is_rate_limit_error(error):
                rate_limit_waits += 1

                wait_seconds = parse_retry_after_seconds(
                    error
                )

                append_jsonl(
                    BATCH_HISTORY_FILE,
                    {
                        "time_utc": now_iso(),
                        "repo_id": REPO_ID,
                        "label": label,
                        "status": "rate_limited",
                        "files": len(batch),
                        "bytes": batch_bytes,
                        "wait_seconds": wait_seconds,
                        "error": str(error),
                    },
                )

                print()
                print("=" * 72)
                print("[RATE LIMIT] 服务端返回429")
                print("[RATE LIMIT] 当前批次不拆分。")
                print("[RATE LIMIT] 等待后自动继续同一批。")
                print("=" * 72)

                sleep_with_countdown(
                    wait_seconds,
                    "HTTP 429，等待 commit 配额恢复",
                )

                continue

            if is_fatal_error(error):
                print()
                print(
                    "[FATAL] 认证、权限、仓库或接口错误。"
                )
                raise

            normal_attempt += 1

            append_jsonl(
                BATCH_HISTORY_FILE,
                {
                    "time_utc": now_iso(),
                    "repo_id": REPO_ID,
                    "label": label,
                    "status": "network_error",
                    "files": len(batch),
                    "bytes": batch_bytes,
                    "attempt": normal_attempt,
                    "error": str(error),
                },
            )

            if normal_attempt < max_normal_attempts:
                wait = retry_wait(normal_attempt)
                print(
                    f"{indent}[RETRY] "
                    f"{wait}秒后重试..."
                )
                time.sleep(wait)
                continue

            break

    if len(batch) == 1:
        _, relative_path, file_size = batch[0]

        print()
        print("=" * 72)
        print("[FAILED] 单文件连续失败：")
        print(relative_path)
        print(
            f"[FAILED] 大小："
            f"{format_size(file_size)}"
        )
        print("=" * 72)

        save_failed_file(
            relative_path,
            last_error,
        )

        return False

    left, right = split_batch(batch)

    print()
    print("=" * 72)
    print(
        f"{indent}[SPLIT] {label} "
        f"连续网络失败，自动拆批"
    )
    print(
        f"{indent}[SPLIT] 原："
        f"{len(batch)}文件 / "
        f"{format_size(batch_bytes)}"
    )
    print(
        f"{indent}[SPLIT] 左："
        f"{len(left)}文件 / "
        f"{format_size(sum(x[2] for x in left))}"
    )
    print(
        f"{indent}[SPLIT] 右："
        f"{len(right)}文件 / "
        f"{format_size(sum(x[2] for x in right))}"
    )
    print("=" * 72)

    upload_batch_adaptive(
        api,
        left,
        f"{label}-A",
        completed,
        state,
        depth + 1,
    )

    upload_batch_adaptive(
        api,
        right,
        f"{label}-B",
        completed,
        state,
        depth + 1,
    )

    return True

def main():
    run_start = time.time()

    append_jsonl(
        RUN_HISTORY_FILE,
        {
            "time_utc": now_iso(),
            "event": "run_start",
            "timestamp": run_start,
            "repo_id": REPO_ID,
        },
    )

    print()
    print("=" * 72)
    print("Hugging Face Multicamera MOT Dataset 上传")
    print("HF-Mirror + 大批次 + 时间窗 + 断点续传")
    print("=" * 72)

    print()
    print(f"[INFO] Python：{sys.version.split()[0]}")
    print(
        f"[INFO] huggingface_hub："
        f"{huggingface_hub.__version__}"
    )
    print(f"[INFO] Endpoint：{HF_ENDPOINT}")
    print("[INFO] Xet：DISABLED")
    print("[INFO] 远端 tree/list 扫描：DISABLED")

    print()
    print(
        f"[INFO] 每批最多："
        f"{MAX_FILES_PER_BATCH}个文件"
    )
    print(
        f"[INFO] 每批最多："
        f"{format_size(MAX_BYTES_PER_BATCH)}"
    )
    print(
        f"[INFO] 60分钟 commit 安全阈值："
        f"{MAX_COMMITS_PER_HOUR}"
    )

    if (
        not HF_TOKEN
        or HF_TOKEN
        == "hf_把你的新WriteToken填在这里"
    ):
        print()
        print("[ERROR] 请填写新的 Write Token。")
        return

    if not LOCAL_FOLDER.exists():
        print()
        print("[ERROR] 数据集目录不存在：")
        print(LOCAL_FOLDER)
        return

    if not LOCAL_FOLDER.is_dir():
        print()
        print("[ERROR] LOCAL_FOLDER 不是目录。")
        return

    print()
    print("[INFO] 本地目录：")
    print(LOCAL_FOLDER)

    print()
    print("[INFO] 目标仓库：")
    print(REPO_ID)

    api = HfApi(
        endpoint=HF_ENDPOINT,
        token=HF_TOKEN,
    )

    print()
    print("=" * 72)
    print("STEP 1：测试 HF-Mirror 登录")
    print("=" * 72)

    try:
        user = api.whoami()
        username = (
            user.get("name")
            or user.get("fullname")
            or "unknown"
        )
        print(f"[OK] 登录成功：{username}")
    except Exception as error:
        print(f"[ERROR] 登录失败：{error}")
        return

    print()
    print("=" * 72)
    print("STEP 2：检查 Dataset Repository")
    print("=" * 72)

    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            exist_ok=True,
        )
        print("[OK] Repository 可访问")
    except Exception as error:
        print(
            f"[ERROR] Repository 检查失败："
            f"{error}"
        )
        return

    completed = load_completed_files()
    info = scan_dataset(completed)
    pending = info["pending"]

    if not pending:
        print()
        print("=" * 72)
        print("[DONE] 本地断点显示没有待上传文件。")
        print("=" * 72)
        return

    initial_batches = count_batches(pending)

    print()
    print(
        f"[INFO] 预计初始批次数："
        f"{initial_batches:,}"
    )
    print(
        "[INFO] 不会删除远端已经上传成功的内容。"
    )
    print(
        "[INFO] 本地断点记录过的文件不会再传。"
    )

    state = {
        "start_time": time.time(),
        "session_files": 0,
        "session_bytes": 0,
        "initial_completed_files": info["completed_files"],
        "initial_completed_bytes": info["completed_size"],
        "total_files": info["total_files"],
        "total_bytes": info["total_size"],
        "successful_commits": 0,
    }

    print()
    print("=" * 72)
    print("STEP 3：开始上传")
    print("=" * 72)
    print()
    print("[INFO] 正常批次之间没有固定等待。")
    print("[INFO] 只有达到时间窗阈值才等待。")
    print("[INFO] 429会等待后继续同一批。")
    print("[INFO] 普通网络错误才会重试/拆批。")

    try:
        for batch_index, (
            batch,
            _batch_bytes,
        ) in enumerate(
            build_batches(pending),
            start=1,
        ):
            label = (
                f"Multicamera-Batch-"
                f"{batch_index:05d}"
            )

            upload_batch_adaptive(
                api=api,
                batch=batch,
                label=label,
                completed=completed,
                state=state,
            )

    except KeyboardInterrupt:
        elapsed = time.time() - run_start

        append_jsonl(
            RUN_HISTORY_FILE,
            {
                "time_utc": now_iso(),
                "event": "run_stopped",
                "timestamp": time.time(),
                "repo_id": REPO_ID,
                "elapsed_seconds": elapsed,
                "session_files": state["session_files"],
                "session_bytes": state["session_bytes"],
            },
        )

        print()
        print("=" * 72)
        print("[STOP] 用户手动停止")
        print("[INFO] 已成功内容已经写入断点。")
        print("[INFO] 下次直接重新运行即可继续。")
        print("=" * 72)
        return

    except Exception as error:
        elapsed = time.time() - run_start

        append_jsonl(
            RUN_HISTORY_FILE,
            {
                "time_utc": now_iso(),
                "event": "run_fatal_error",
                "timestamp": time.time(),
                "repo_id": REPO_ID,
                "elapsed_seconds": elapsed,
                "error": str(error),
            },
        )

        print()
        print("=" * 72)
        print("[FATAL] 上传过程出现致命错误")
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print("=" * 72)
        traceback.print_exc()
        return

    elapsed = time.time() - run_start

    append_jsonl(
        RUN_HISTORY_FILE,
        {
            "time_utc": now_iso(),
            "event": "run_finished",
            "timestamp": time.time(),
            "repo_id": REPO_ID,
            "elapsed_seconds": elapsed,
            "session_files": state["session_files"],
            "session_bytes": state["session_bytes"],
            "successful_commits": state["successful_commits"],
        },
    )

    print()
    print("=" * 72)
    print("本轮上传结束")
    print("=" * 72)
    print(
        f"[DONE] 本轮成功文件："
        f"{state['session_files']:,}"
    )
    print(
        f"[DONE] 本轮成功数据："
        f"{format_size(state['session_bytes'])}"
    )
    print(
        f"[DONE] 本轮成功commit："
        f"{state['successful_commits']:,}"
    )
    print(
        f"[DONE] 本轮运行时间："
        f"{format_time(elapsed)}"
    )
    print()
    print("[INFO] 本数据集断点：")
    print(CHECKPOINT_FILE)

if __name__ == "__main__":
    main()
