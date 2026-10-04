# -*- coding: utf-8 -*-

"""
Hugging Face Dataset - HF-Mirror 大批次 + 时间窗限流 + 断点续传版

适合当前场景：
- 使用 https://hf-mirror.net
- 数据集文件很多、总体积大
- 之前已经上传了一部分，不希望删除
- repository commits 限制约 128 次 / 小时
- 网络可能中断

核心逻辑：
1. 不删除 Hugging Face 上任何已经上传成功的内容。
2. 继续沿用旧的 hf_upload_completed_files.txt。
3. 每批默认最多 300 个文件 / 512 MB，比之前每批更大。
4. 不再固定每批等待 30 秒。
5. 每次成功 commit 都记录时间到 hf_commit_history.jsonl。
6. 每次准备 commit 前，只检查“最近 60 分钟”成功 commit 数：
   - < 120：立即上传，不等待
   - >= 120：只等待到最早一次 commit 滑出 60 分钟窗口
7. 遇到 HTTP 429：
   - 不拆批
   - 不算普通网络失败
   - 优先读取 Retry-After
   - 没有 Retry-After 时默认等约 61 分 40 秒
   - 等待后继续同一批
8. 普通网络错误：
   - 指数退避重试
   - 多次失败后自动拆小批次
9. 每批成功后立刻保存断点。
10. Ctrl+C、断网、重启电脑后，重新运行即可继续。

重要：
- 为了继承旧进度，请把本脚本放到旧脚本相同目录。
- 不要删除 hf_upload_completed_files.txt。
- 请使用新的 Hugging Face Write Token，不要使用已经泄露的旧 Token。
"""

# ============================================================
# 0. 环境变量（必须在 import huggingface_hub 之前）
# ============================================================

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.net"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)


# ============================================================
# 1. Imports
# ============================================================

import sys
import time
import json
import traceback
from pathlib import Path
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import huggingface_hub
from huggingface_hub import HfApi, CommitOperationAdd


# ============================================================
# 2. 用户配置
# ============================================================

LOCAL_FOLDER = Path(
    r"E:\pythonProject\air_groud\get_img"
    r"\dataset_uav_multimap_single_object_vot_weather500"
)

REPO_ID = "yutiangu/dataset_uav_multimap_single_object_vot_weather500"
REPO_TYPE = "dataset"
HF_ENDPOINT = "https://hf-mirror.net"

# ------------------------------------------------------------
# Hugging Face Write Token
# ------------------------------------------------------------
# 示例：
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
# ------------------------------------------------------------

HF_TOKEN = ""


# ============================================================
# 3. 批次参数：比之前更大
# ============================================================

# 每批最多 300 个文件
MAX_FILES_PER_BATCH = 300

# 每批最多 512 MB
MAX_BYTES_PER_BATCH = 512 * 1024 * 1024

# 普通网络错误：批次失败多少次后自动拆小
BATCH_RETRIES_BEFORE_SPLIT = 5

# 单文件最终最多重试次数
SINGLE_FILE_MAX_RETRIES = 20

# 普通网络错误第一次等待 10 秒
RETRY_BASE_SECONDS = 10

# 普通网络错误最大等待 5 分钟
MAX_RETRY_WAIT_SECONDS = 300


# ============================================================
# 4. Commit 限流控制
# ============================================================

# 服务端提示约 128 commits / hour。
# 本地保守使用 120，留一点余量。
MAX_COMMITS_PER_HOUR = 120

# 60 分钟滑动窗口
COMMIT_WINDOW_SECONDS = 3600

# 到阈值时，额外多等 5 秒安全余量
COMMIT_WINDOW_SAFETY_SECONDS = 5

# 429 没有 Retry-After 时，默认等待 3700 秒
RATE_LIMIT_FALLBACK_SECONDS = 3700

# Retry-After 基础上额外加 30 秒
RATE_LIMIT_SAFETY_SECONDS = 30


# ============================================================
# 5. 本地状态文件
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# 继续沿用之前的成功断点
CHECKPOINT_FILE = SCRIPT_DIR / "hf_upload_completed_files.txt"

# 最终仍失败的文件
FAILED_FILE = SCRIPT_DIR / "hf_upload_failed_files.txt"

# 成功 commit 的时间记录
COMMIT_HISTORY_FILE = SCRIPT_DIR / "hf_commit_history.jsonl"

# 每批上传结果记录
BATCH_HISTORY_FILE = SCRIPT_DIR / "hf_batch_history.jsonl"

# 每次脚本运行记录
RUN_HISTORY_FILE = SCRIPT_DIR / "hf_run_history.jsonl"


# ============================================================
# 6. 忽略内容
# ============================================================

IGNORE_DIRS = {
    ".git",
    ".cache",
    "__pycache__",
}

IGNORE_FILES = {
    "Thumbs.db",
    "desktop.ini",
}

IGNORE_SUFFIXES = {
    ".pyc",
    ".pyo",
}


# ============================================================
# 7. 基础工具
# ============================================================

def format_size(size_bytes):
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def format_time(seconds):
    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, obj):
    with open(
        path,
        "a",
        encoding="utf-8",
        buffering=1,
    ) as f:
        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            ) + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def should_ignore(path):
    if path.name in IGNORE_FILES:
        return True

    if path.suffix.lower() in IGNORE_SUFFIXES:
        return True

    for part in path.parts:
        if part in IGNORE_DIRS:
            return True

    return False


# ============================================================
# 8. HTTP 错误判断
# ============================================================

def get_status_code(error):
    response = getattr(error, "response", None)

    if response is not None:
        try:
            return response.status_code
        except Exception:
            pass

    text = str(error).lower()

    for code in (
        400, 401, 403, 404, 405,
        409, 429, 500, 502, 503, 504,
    ):
        if (
            f"{code} " in text
            or f"{code} client error" in text
        ):
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
    # 429 专门处理，不算 fatal
    if is_rate_limit_error(error):
        return False

    status = get_status_code(error)

    if status in {401, 403, 404, 405}:
        return True

    text = str(error).lower()

    fatal_patterns = [
        "invalid token",
        "permission denied",
        "repository not found",
        "method not allowed",
    ]

    return any(
        pattern in text
        for pattern in fatal_patterns
    )


# ============================================================
# 9. 解析 429 Retry-After
# ============================================================

def parse_retry_after_seconds(error):
    response = getattr(error, "response", None)

    if response is not None:
        try:
            retry_after = response.headers.get("Retry-After")
        except Exception:
            retry_after = None

        if retry_after:
            retry_after = retry_after.strip()

            # 秒数格式
            try:
                seconds = int(retry_after)

                return max(
                    1,
                    seconds + RATE_LIMIT_SAFETY_SECONDS,
                )
            except (TypeError, ValueError):
                pass

            # HTTP-date 格式
            try:
                retry_dt = parsedate_to_datetime(retry_after)

                if retry_dt.tzinfo is None:
                    retry_dt = retry_dt.replace(
                        tzinfo=timezone.utc
                    )

                now = datetime.now(timezone.utc)

                seconds = (
                    retry_dt.astimezone(timezone.utc)
                    - now
                ).total_seconds()

                return max(
                    1,
                    int(seconds) + RATE_LIMIT_SAFETY_SECONDS,
                )
            except Exception:
                pass

    return RATE_LIMIT_FALLBACK_SECONDS


# ============================================================
# 10. 倒计时等待
# ============================================================

def sleep_with_countdown(seconds, reason):
    remaining = int(seconds)

    print()
    print(f"[WAIT] {reason}")
    print(
        f"[WAIT] 预计等待："
        f"{format_time(remaining)}"
    )

    while remaining > 0:
        step = min(60, remaining)

        time.sleep(step)
        remaining -= step

        if remaining > 0:
            print(
                f"[WAIT] 剩余约："
                f"{format_time(remaining)}",
                flush=True,
            )

    print(
        "[WAIT] 等待结束，继续上传。",
        flush=True,
    )


# ============================================================
# 11. Commit 时间历史
# ============================================================

def load_commit_history():
    """
    读取最近 60 分钟的成功 commit 时间。
    """
    timestamps = []

    if not COMMIT_HISTORY_FILE.exists():
        return timestamps

    now = time.time()

    try:
        with open(
            COMMIT_HISTORY_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    ts = float(obj["timestamp"])
                except Exception:
                    continue

                if (
                    now - ts
                    < COMMIT_WINDOW_SECONDS
                ):
                    timestamps.append(ts)

    except Exception as error:
        print()
        print("[WARN] 读取 commit 历史失败：")
        print(error)

    timestamps.sort()
    return timestamps


def rewrite_commit_history(timestamps):
    """
    清理 commit 历史文件，只保留最近 60 分钟。
    """
    try:
        with open(
            COMMIT_HISTORY_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            for ts in timestamps:
                obj = {
                    "timestamp": ts,
                    "time_utc": datetime.fromtimestamp(
                        ts,
                        tz=timezone.utc,
                    ).isoformat(),
                }

                f.write(
                    json.dumps(
                        obj,
                        ensure_ascii=False,
                    ) + "\n"
                )

    except Exception as error:
        print()
        print("[WARN] 清理 commit 历史失败：")
        print(error)


def record_successful_commit(
    timestamp,
    label,
    files,
    bytes_size,
    elapsed,
):
    append_jsonl(
        COMMIT_HISTORY_FILE,
        {
            "timestamp": timestamp,
            "time_utc": datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            ).isoformat(),
            "label": label,
            "files": files,
            "bytes": bytes_size,
            "elapsed_seconds": elapsed,
        },
    )


# ============================================================
# 12. 滑动时间窗控制
# ============================================================

def wait_if_commit_window_full():
    """
    不再每批固定等待。

    只有最近 60 分钟成功 commit >= 120 时，
    才等待到最早一条 commit 滑出 60 分钟窗口。
    """
    while True:
        history = load_commit_history()

        # 顺便压缩历史文件
        rewrite_commit_history(history)

        count = len(history)

        print(
            f"[RATE] 最近 60 分钟成功 commit："
            f"{count}/{MAX_COMMITS_PER_HOUR}"
        )

        if count < MAX_COMMITS_PER_HOUR:
            return

        oldest = history[0]
        now = time.time()

        wait_seconds = (
            COMMIT_WINDOW_SECONDS
            - (now - oldest)
            + COMMIT_WINDOW_SAFETY_SECONDS
        )

        if wait_seconds <= 0:
            continue

        print()
        print("=" * 72)
        print(
            "[RATE] 最近 60 分钟 commit "
            "已达到本地安全阈值"
        )
        print(
            f"[RATE] 当前："
            f"{count}/{MAX_COMMITS_PER_HOUR}"
        )
        print(
            "[RATE] 不会每批固定等半分钟。"
        )
        print(
            "[RATE] 只等待到最早一次 commit "
            "退出 60 分钟时间窗。"
        )
        print("=" * 72)

        sleep_with_countdown(
            wait_seconds,
            "等待 commit 滑动时间窗释放名额",
        )


# ============================================================
# 13. 断点：读取 / 保存
# ============================================================

def load_completed_files():
    completed = set()

    if not CHECKPOINT_FILE.exists():
        print()
        print("[INFO] 没有发现旧断点文件。")
        return completed

    print()
    print("[INFO] 检测到旧断点文件：")
    print(CHECKPOINT_FILE)

    with open(
        CHECKPOINT_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            path = line.strip()

            if path:
                completed.add(path)

    print(
        f"[INFO] 已记录成功文件："
        f"{len(completed):,}"
    )

    return completed


def save_completed_files(
    relative_paths,
    completed,
):
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


def save_failed_file(
    relative_path,
    error,
):
    with open(
        FAILED_FILE,
        "a",
        encoding="utf-8",
        buffering=1,
    ) as f:
        message = str(error).replace(
            "\n",
            " ",
        )

        f.write(
            f"{relative_path}\t"
            f"{message}\n"
        )

        f.flush()
        os.fsync(f.fileno())


# ============================================================
# 14. 扫描数据集
# ============================================================

def scan_dataset(completed):
    print()
    print("=" * 72)
    print("扫描本地数据集")
    print("=" * 72)

    pending = []

    total_files = 0
    total_size = 0

    completed_files = 0
    completed_size = 0

    pending_files = 0
    pending_size = 0

    scanned = 0

    for root, dirs, files in os.walk(
        LOCAL_FOLDER
    ):
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in IGNORE_DIRS
        )

        files = sorted(files)

        for filename in files:
            absolute_path = (
                Path(root)
                / filename
            )

            if should_ignore(
                absolute_path
            ):
                continue

            try:
                file_size = (
                    absolute_path
                    .stat()
                    .st_size
                )
            except OSError as error:
                print()
                print("[WARN] 无法读取文件：")
                print(absolute_path)
                print(error)
                continue

            relative_path = (
                absolute_path
                .relative_to(
                    LOCAL_FOLDER
                )
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
                    (
                        absolute_path,
                        relative_path,
                        file_size,
                    )
                )

            scanned += 1

            if scanned % 5000 == 0:
                print(
                    "\r"
                    f"[SCAN] 已扫描 "
                    f"{scanned:,} 个文件...",
                    end="",
                    flush=True,
                )

    print()
    print()

    print(
        f"[INFO] 总文件："
        f"{total_files:,}"
    )
    print(
        f"[INFO] 总大小："
        f"{format_size(total_size)}"
    )

    print()
    print(
        f"[INFO] 已完成文件："
        f"{completed_files:,}"
    )
    print(
        f"[INFO] 已完成大小："
        f"{format_size(completed_size)}"
    )

    print()
    print(
        f"[INFO] 待上传文件："
        f"{pending_files:,}"
    )
    print(
        f"[INFO] 待上传大小："
        f"{format_size(pending_size)}"
    )

    return {
        "pending": pending,
        "total_files": total_files,
        "total_size": total_size,
        "completed_files": completed_files,
        "completed_size": completed_size,
        "pending_files": pending_files,
        "pending_size": pending_size,
    }


# ============================================================
# 15. 批次构建
# ============================================================

def build_batches(files):
    batch = []
    batch_bytes = 0

    for item in files:
        _, _, file_size = item

        if batch:
            too_many = (
                len(batch)
                >= MAX_FILES_PER_BATCH
            )

            too_large = (
                batch_bytes + file_size
                > MAX_BYTES_PER_BATCH
            )

            if too_many or too_large:
                yield (
                    batch,
                    batch_bytes,
                )

                batch = []
                batch_bytes = 0

        batch.append(item)
        batch_bytes += file_size

    if batch:
        yield (
            batch,
            batch_bytes,
        )


def count_batches(files):
    return sum(
        1
        for _batch, _size
        in build_batches(files)
    )


# ============================================================
# 16. 普通网络错误时拆批
# ============================================================

def split_batch(batch):
    """
    只有普通网络错误连续失败才拆批。
    429 不拆。
    """
    if len(batch) <= 1:
        return batch, []

    total_size = sum(
        item[2]
        for item in batch
    )

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


# ============================================================
# 17. Commit operations
# ============================================================

def create_operations(batch):
    operations = []

    for (
        absolute_path,
        relative_path,
        _file_size,
    ) in batch:
        operations.append(
            CommitOperationAdd(
                path_in_repo=relative_path,
                path_or_fileobj=str(
                    absolute_path
                ),
            )
        )

    return operations


# ============================================================
# 18. 普通网络错误退避
# ============================================================

def retry_wait(attempt):
    wait = (
        RETRY_BASE_SECONDS
        * (2 ** (attempt - 1))
    )

    return min(
        wait,
        MAX_RETRY_WAIT_SECONDS,
    )


# ============================================================
# 19. 总进度
# ============================================================

def show_progress(state):
    elapsed = (
        time.time()
        - state["start_time"]
    )

    session_bytes = (
        state["session_bytes"]
    )

    session_files = (
        state["session_files"]
    )

    done_bytes = (
        state["initial_completed_bytes"]
        + session_bytes
    )

    done_files = (
        state["initial_completed_files"]
        + session_files
    )

    total_bytes = state["total_bytes"]
    total_files = state["total_files"]

    bytes_percent = (
        done_bytes
        / total_bytes
        * 100
        if total_bytes
        else 0
    )

    files_percent = (
        done_files
        / total_files
        * 100
        if total_files
        else 0
    )

    speed = (
        session_bytes
        / elapsed
        if elapsed > 0
        else 0
    )

    remaining = max(
        0,
        total_bytes - done_bytes,
    )

    eta = (
        remaining / speed
        if speed > 0
        and remaining > 0
        else None
    )

    recent_commits = len(
        load_commit_history()
    )

    print()
    print("=" * 72)

    print(
        f"[PROGRESS] 文件："
        f"{done_files:,}/"
        f"{total_files:,} "
        f"({files_percent:.2f}%)"
    )

    print(
        f"[PROGRESS] 数据："
        f"{format_size(done_bytes)} / "
        f"{format_size(total_bytes)} "
        f"({bytes_percent:.2f}%)"
    )

    print(
        f"[PROGRESS] 本轮上传："
        f"{format_size(session_bytes)}"
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
        f"[PROGRESS] 最近 60 分钟成功 commit："
        f"{recent_commits}/"
        f"{MAX_COMMITS_PER_HOUR}"
    )

    print("=" * 72)


# ============================================================
# 20. 执行一次 commit
# ============================================================

def commit_once(
    api,
    batch,
    label,
):
    # 只有真的接近 60 分钟 commit 阈值才等待
    wait_if_commit_window_full()

    operations = create_operations(
        batch
    )

    batch_bytes = sum(
        item[2]
        for item in batch
    )

    start_ts = time.time()

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

    elapsed = (
        time.time()
        - start_ts
    )

    # 成功后记录 commit 时间
    record_successful_commit(
        timestamp=time.time(),
        label=label,
        files=len(batch),
        bytes_size=batch_bytes,
        elapsed=elapsed,
    )

    return (
        result,
        elapsed,
    )


# ============================================================
# 21. 成功后保存断点
# ============================================================

def mark_batch_success(
    batch,
    label,
    elapsed,
    completed,
    state,
):
    relative_paths = [
        item[1]
        for item in batch
    ]

    batch_bytes = sum(
        item[2]
        for item in batch
    )

    # 立刻保存文件断点
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
            "label": label,
            "status": "success",
            "files": len(batch),
            "bytes": batch_bytes,
            "elapsed_seconds": elapsed,
        },
    )

    show_progress(state)


# ============================================================
# 22. 自适应上传核心
# ============================================================

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

    batch_bytes = sum(
        item[2]
        for item in batch
    )

    indent = "  " * depth

    if len(batch) == 1:
        max_normal_attempts = (
            SINGLE_FILE_MAX_RETRIES
        )
    else:
        max_normal_attempts = (
            BATCH_RETRIES_BEFORE_SPLIT
        )

    normal_attempt = 0
    rate_limit_waits = 0
    last_error = None

    while True:
        print()
        print("-" * 72)

        print(
            f"{indent}[UPLOAD] {label}"
        )

        print(
            f"{indent}[INFO] 文件数："
            f"{len(batch)}"
        )

        print(
            f"{indent}[INFO] 数据量："
            f"{format_size(batch_bytes)}"
        )

        print(
            f"{indent}[INFO] 普通网络失败次数："
            f"{normal_attempt}/"
            f"{max_normal_attempts}"
        )

        print(
            f"{indent}[INFO] 429 等待次数："
            f"{rate_limit_waits}"
        )

        print("-" * 72)

        try:
            _result, elapsed = (
                commit_once(
                    api=api,
                    batch=batch,
                    label=label,
                )
            )

            print()
            print(
                f"{indent}[OK] "
                f"{label} 上传成功"
            )

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
                f"{indent}[WARN] 上传失败"
            )

            print(
                f"{indent}[WARN] "
                f"{type(error).__name__}: "
                f"{error}"
            )

            # =================================================
            # A. 429：不拆批，不计普通网络失败
            # =================================================
            if is_rate_limit_error(error):
                rate_limit_waits += 1

                wait_seconds = (
                    parse_retry_after_seconds(
                        error
                    )
                )

                append_jsonl(
                    BATCH_HISTORY_FILE,
                    {
                        "time_utc": now_iso(),
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
                print(
                    "[RATE LIMIT] 服务端返回 429"
                )
                print(
                    "[RATE LIMIT] 当前批次不会拆分。"
                )
                print(
                    "[RATE LIMIT] 之前已经上传成功的内容"
                    "不会删除。"
                )
                print(
                    "[RATE LIMIT] 等待额度恢复后，"
                    "自动继续当前批次。"
                )
                print("=" * 72)

                sleep_with_countdown(
                    wait_seconds,
                    (
                        "HTTP 429，等待 repository "
                        "commit 限流恢复"
                    ),
                )

                continue

            # =================================================
            # B. 认证/权限/接口错误
            # =================================================
            if is_fatal_error(error):
                append_jsonl(
                    BATCH_HISTORY_FILE,
                    {
                        "time_utc": now_iso(),
                        "label": label,
                        "status": "fatal",
                        "files": len(batch),
                        "bytes": batch_bytes,
                        "error": str(error),
                    },
                )

                print()
                print(
                    "[FATAL] 认证、权限、仓库"
                    "或写接口错误。"
                )
                print(
                    "[FATAL] 这类错误不会通过"
                    "等待或拆批解决。"
                )

                raise

            # =================================================
            # C. 普通网络错误
            # =================================================
            normal_attempt += 1

            append_jsonl(
                BATCH_HISTORY_FILE,
                {
                    "time_utc": now_iso(),
                    "label": label,
                    "status": "network_error",
                    "files": len(batch),
                    "bytes": batch_bytes,
                    "attempt": normal_attempt,
                    "error": str(error),
                },
            )

            if (
                normal_attempt
                < max_normal_attempts
            ):
                wait = retry_wait(
                    normal_attempt
                )

                print(
                    f"{indent}[RETRY] "
                    f"{wait} 秒后重试..."
                )

                time.sleep(wait)
                continue

            break

    # ========================================================
    # 单文件连续失败
    # ========================================================

    if len(batch) == 1:
        (
            _absolute_path,
            relative_path,
            file_size,
        ) = batch[0]

        print()
        print("=" * 72)
        print("[FAILED] 单文件连续失败：")
        print(relative_path)
        print(
            f"[FAILED] 文件大小："
            f"{format_size(file_size)}"
        )
        print(
            "[FAILED] 已记录到："
        )
        print(FAILED_FILE)
        print("=" * 72)

        save_failed_file(
            relative_path,
            last_error,
        )

        return False

    # ========================================================
    # 普通网络错误：自动拆成更小批次
    # ========================================================

    left, right = split_batch(
        batch
    )

    print()
    print("=" * 72)

    print(
        f"{indent}[SPLIT] "
        f"{label} 连续网络失败，"
        "自动拆成更小批次"
    )

    print(
        f"{indent}[SPLIT] 原批次："
        f"{len(batch)} 文件 / "
        f"{format_size(batch_bytes)}"
    )

    print(
        f"{indent}[SPLIT] 左："
        f"{len(left)} 文件 / "
        f"{format_size(sum(x[2] for x in left))}"
    )

    print(
        f"{indent}[SPLIT] 右："
        f"{len(right)} 文件 / "
        f"{format_size(sum(x[2] for x in right))}"
    )

    print("=" * 72)

    upload_batch_adaptive(
        api=api,
        batch=left,
        label=f"{label}-A",
        completed=completed,
        state=state,
        depth=depth + 1,
    )

    upload_batch_adaptive(
        api=api,
        batch=right,
        label=f"{label}-B",
        completed=completed,
        state=state,
        depth=depth + 1,
    )

    return True


# ============================================================
# 23. 主程序
# ============================================================

def main():
    run_start = time.time()

    append_jsonl(
        RUN_HISTORY_FILE,
        {
            "time_utc": now_iso(),
            "event": "run_start",
            "timestamp": run_start,
        },
    )

    print()
    print("=" * 72)
    print("Hugging Face UAV Dataset")
    print(
        "HF-Mirror 大批次 + "
        "滑动时间窗 + 断点续传版"
    )
    print("=" * 72)

    print()
    print(
        f"[INFO] Python："
        f"{sys.version.split()[0]}"
    )

    print(
        f"[INFO] huggingface_hub："
        f"{huggingface_hub.__version__}"
    )

    print(
        f"[INFO] Endpoint："
        f"{HF_ENDPOINT}"
    )

    print("[INFO] Xet：DISABLED")

    print()
    print(
        f"[INFO] 每批最多："
        f"{MAX_FILES_PER_BATCH} 个文件"
    )

    print(
        f"[INFO] 每批最多："
        f"{format_size(MAX_BYTES_PER_BATCH)}"
    )

    print(
        f"[INFO] 最近 60 分钟最多："
        f"{MAX_COMMITS_PER_HOUR} 个成功 commit"
    )

    print(
        "[INFO] 不再每批固定等待约 30 秒。"
    )

    print(
        "[INFO] 只有最近 60 分钟 "
        "commit 真正接近阈值时才等待。"
    )

    # ========================================================
    # Token 检查
    # ========================================================

    if (
        not HF_TOKEN
        or HF_TOKEN
        == "hf_把你的新WriteToken填在这里"
    ):
        print()
        print(
            "[ERROR] 请填写新的 "
            "Hugging Face Write Token。"
        )
        print(
            'HF_TOKEN = "hf_xxxxx..."'
        )
        return

    # ========================================================
    # 本地目录
    # ========================================================

    if not LOCAL_FOLDER.exists():
        print()
        print(
            "[ERROR] 数据集目录不存在："
        )
        print(LOCAL_FOLDER)
        return

    if not LOCAL_FOLDER.is_dir():
        print()
        print(
            "[ERROR] LOCAL_FOLDER 不是文件夹。"
        )
        return

    print()
    print("[INFO] 本地数据集：")
    print(LOCAL_FOLDER)

    print()
    print(
        f"[INFO] Repository："
        f"{REPO_ID}"
    )

    # ========================================================
    # API
    # ========================================================

    api = HfApi(
        endpoint=HF_ENDPOINT,
        token=HF_TOKEN,
    )

    # ========================================================
    # STEP 1：登录
    # ========================================================

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

        print()
        print(
            f"[OK] 登录成功："
            f"{username}"
        )

    except Exception as error:
        print()
        print("[ERROR] 登录失败：")
        print(error)
        return

    # ========================================================
    # STEP 2：仓库
    # ========================================================

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

        print()
        print("[OK] Repository 可访问")

    except Exception as error:
        print()
        print(
            "[ERROR] Repository 检查失败："
        )
        print(error)
        return

    # ========================================================
    # STEP 3：读取旧断点
    # ========================================================

    completed = (
        load_completed_files()
    )

    # ========================================================
    # STEP 4：扫描
    # ========================================================

    info = scan_dataset(
        completed
    )

    pending = info["pending"]

    if not pending:
        print()
        print("=" * 72)
        print(
            "[DONE] 根据本地断点，"
            "所有文件已经完成"
        )
        print("=" * 72)
        return

    initial_batches = (
        count_batches(
            pending
        )
    )

    print()
    print(
        f"[INFO] 当前预计初始批次数："
        f"{initial_batches:,}"
    )

    print(
        "[INFO] 之前已经上传成功的内容不会删除。"
    )

    print(
        "[INFO] 已进入旧断点的文件不会再次上传。"
    )

    # ========================================================
    # 状态
    # ========================================================

    state = {
        "start_time": time.time(),
        "session_files": 0,
        "session_bytes": 0,

        "initial_completed_files":
            info["completed_files"],

        "initial_completed_bytes":
            info["completed_size"],

        "total_files":
            info["total_files"],

        "total_bytes":
            info["total_size"],

        "successful_commits": 0,
    }

    # ========================================================
    # STEP 5：开始上传
    # ========================================================

    print()
    print("=" * 72)
    print("STEP 3：开始上传")
    print("=" * 72)

    print()
    print(
        "[INFO] 正常情况下批次之间不固定等待。"
    )

    print(
        "[INFO] 每个成功 commit 的时间会保存到本地。"
    )

    print(
        "[INFO] 只有最近 60 分钟真正接近 "
        "commit 阈值时才暂停。"
    )

    print(
        "[INFO] 429 会自动等待后继续同一批。"
    )

    print(
        "[INFO] 普通网络错误连续失败才会拆批。"
    )

    try:
        for batch_index, (
            batch,
            _batch_bytes,
        ) in enumerate(
            build_batches(
                pending
            ),
            start=1,
        ):
            label = (
                f"Batch-"
                f"{batch_index:05d}"
            )

            upload_batch_adaptive(
                api=api,
                batch=batch,
                label=label,
                completed=completed,
                state=state,
            )

    # ========================================================
    # Ctrl+C
    # ========================================================

    except KeyboardInterrupt:
        elapsed = (
            time.time()
            - run_start
        )

        append_jsonl(
            RUN_HISTORY_FILE,
            {
                "time_utc": now_iso(),
                "event": "run_stopped",
                "timestamp": time.time(),
                "elapsed_seconds": elapsed,
                "session_files":
                    state["session_files"],
                "session_bytes":
                    state["session_bytes"],
            },
        )

        print()
        print()
        print("=" * 72)
        print("[STOP] 用户手动停止")
        print("=" * 72)

        print()
        print(
            "[INFO] 已成功上传的内容"
            "已经写入原断点文件。"
        )

        print(
            "[INFO] 不需要删除 Hugging Face "
            "上任何已经上传好的内容。"
        )

        print(
            "[INFO] 下次重新运行即可继续。"
        )

        return

    # ========================================================
    # 致命错误
    # ========================================================

    except Exception as error:
        elapsed = (
            time.time()
            - run_start
        )

        append_jsonl(
            RUN_HISTORY_FILE,
            {
                "time_utc": now_iso(),
                "event": "run_fatal_error",
                "timestamp": time.time(),
                "elapsed_seconds": elapsed,
                "error": str(error),
            },
        )

        print()
        print()
        print("=" * 72)
        print(
            "[FATAL] 上传过程出现致命错误"
        )
        print("=" * 72)

        print()
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print()
        traceback.print_exc()

        print()
        print(
            "[INFO] 已完成的内容不会删除。"
        )

        print(
            "[INFO] 修复问题后重新运行即可继续。"
        )

        return

    # ========================================================
    # 本轮结束
    # ========================================================

    elapsed = (
        time.time()
        - run_start
    )

    append_jsonl(
        RUN_HISTORY_FILE,
        {
            "time_utc": now_iso(),
            "event": "run_finished",
            "timestamp": time.time(),
            "elapsed_seconds": elapsed,
            "session_files":
                state["session_files"],
            "session_bytes":
                state["session_bytes"],
            "successful_commits":
                state["successful_commits"],
        },
    )

    print()
    print()
    print("=" * 72)
    print("本轮上传结束")
    print("=" * 72)

    print()
    print(
        f"[DONE] 本轮成功文件："
        f"{state['session_files']:,}"
    )

    print(
        f"[DONE] 本轮成功数据："
        f"{format_size(state['session_bytes'])}"
    )

    print(
        f"[DONE] 本轮成功 commit："
        f"{state['successful_commits']:,}"
    )

    print(
        f"[DONE] 本轮运行时间："
        f"{format_time(elapsed)}"
    )

    print()
    print("[INFO] 成功断点文件：")
    print(CHECKPOINT_FILE)

    print()
    print("[INFO] Commit 时间历史：")
    print(COMMIT_HISTORY_FILE)

    print()
    print("[INFO] 批次历史：")
    print(BATCH_HISTORY_FILE)

    print()
    print("[INFO] 运行历史：")
    print(RUN_HISTORY_FILE)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
