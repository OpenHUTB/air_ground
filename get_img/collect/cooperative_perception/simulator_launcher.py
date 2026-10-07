"""Start the simulator package selected for the requested map when needed."""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_STANDARD_EXECUTABLE = Path(
    r"E:\OpenHUTB\中电软件园\hutb_windows_v2.10.0\CarlaUE4.exe"
)
DEFAULT_CCSP_EXECUTABLE = Path(
    r"E:\OpenHUTB\中电软件园\WindowsNoEditor\WindowsNoEditor\CarlaUE4.exe"
)
DEFAULT_CCSP_MARKERS = (
    "ccsp",
    "zhongdian",
    "software_park",
    "softwarepark",
    "中电园",
    "中电软件园",
)


def simulator_profile(map_name: Optional[str], config: Dict[str, Any]) -> str:
    requested = str(config.get("profile", "auto")).strip().lower()
    if requested in {"standard", "ccsp"}:
        return requested
    if requested != "auto":
        raise ValueError("simulator.profile must be auto, standard, or ccsp")
    normalized = str(map_name or "").lower()
    markers = tuple(config.get("ccsp_map_markers", DEFAULT_CCSP_MARKERS))
    return "ccsp" if any(str(marker).lower() in normalized for marker in markers) else "standard"


def simulator_executable(
    map_name: Optional[str],
    config: Dict[str, Any],
) -> Tuple[str, Path]:
    profile = simulator_profile(map_name, config)
    key = "ccsp_executable" if profile == "ccsp" else "standard_executable"
    value = config.get(
        key,
        DEFAULT_CCSP_EXECUTABLE if profile == "ccsp" else DEFAULT_STANDARD_EXECUTABLE,
    )
    return profile, Path(str(value))


def _probe(carla_module: Any, host: str, port: int, timeout: float) -> Tuple[str, str]:
    client = carla_module.Client(str(host), int(port))
    client.set_timeout(float(timeout))
    world = client.get_world()
    return str(client.get_server_version()), str(world.get_map().name)


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((str(host), int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def _windows_listener_pid(port: int) -> Optional[int]:
    if os.name != "nt":
        return None
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    for line in result.stdout.splitlines():
        parts = line.split()
        if (
            len(parts) >= 5
            and parts[0].upper() == "TCP"
            and parts[1].rsplit(":", 1)[-1] == str(int(port))
            and parts[-2].upper() == "LISTENING"
            and parts[-1].isdigit()
        ):
            return int(parts[-1])
    return None


def close_simulator(session: Optional[Dict[str, Any]]) -> None:
    """Stop only the exact simulator process tree launched by this collector."""
    if not session or not bool(session.get("managed")) or bool(session.get("closed")):
        return
    process = session.get("_process")
    pids = {
        int(value)
        for value in session.get("_pids", [])
        if int(value) > 0
    }
    if process is not None:
        pids.add(int(process.pid))
    if not pids:
        return
    print(
        "[SIM] Closing simulator process tree pid=%s ..."
        % ",".join(str(value) for value in sorted(pids)),
        flush=True,
    )
    try:
        if os.name == "nt":
            for pid in sorted(pids, reverse=True):
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        elif process is not None:
            process.terminate()
            try:
                process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                process.kill()
    finally:
        session["closed"] = True
    host = str(session.get("host", "127.0.0.1"))
    port = int(session.get("port", 2000))
    deadline = time.time() + float(session.get("shutdown_timeout_seconds", 30.0))
    while time.time() < deadline and _port_is_open(host, port):
        time.sleep(0.5)
    if _port_is_open(host, port):
        print(
            "[SIM][WARN] Process tree was closed but RPC port %s:%d is still occupied."
            % (host, port),
            flush=True,
        )
    else:
        print("[SIM] Simulator closed; RPC port can be reused.", flush=True)


def ensure_simulator(
    carla_module: Any,
    host: str,
    port: int,
    map_name: Optional[str],
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reuse a ready server or launch the map-appropriate local simulator."""
    settings = dict(config or {})
    if not bool(settings.get("auto_start", True)):
        return {"started": False, "reason": "auto_start_disabled"}

    probe_timeout = float(settings.get("probe_timeout_seconds", 3.0))
    try:
        server_version, current_map = _probe(
            carla_module, host, port, probe_timeout
        )
        print(
            "[SIM] Reusing ready simulator at %s:%d (server=%s, map=%s)"
            % (host, int(port), server_version, current_map),
            flush=True,
        )
        session = {
            "started": False,
            "reason": "already_ready",
            "server_version": server_version,
            "current_map": current_map,
            "managed": False,
        }
        if bool(settings.get("stop_after_collection", True)):
            listener_pid = _windows_listener_pid(int(port))
            if listener_pid is not None:
                session.update(
                    {
                        "managed": True,
                        "_pids": [listener_pid],
                        "host": str(host),
                        "port": int(port),
                        "shutdown_timeout_seconds": float(
                            settings.get("shutdown_timeout_seconds", 30.0)
                        ),
                        "closed": False,
                    }
                )
                atexit.register(close_simulator, session)
        return session
    except Exception as initial_error:
        if str(host).lower() not in LOCAL_HOSTS:
            raise RuntimeError(
                "Cannot auto-start a simulator for remote host %s; initial probe: %s"
                % (host, initial_error)
            ) from initial_error

    profile, executable = simulator_executable(map_name, settings)
    if not executable.is_file():
        raise FileNotFoundError("Simulator executable not found: %s" % executable)

    command = [
        str(executable),
        "-quality-level=%s" % str(settings.get("quality_level", "Epic")),
        "-carla-rpc-port=%d" % int(port),
    ]
    visible = bool(settings.get("visible", True))
    if not visible:
        command.append("-RenderOffScreen")
    command.extend(str(value) for value in settings.get("extra_args", []))

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    if not visible:
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
    print(
        "[SIM] Auto-starting %s simulator for map %s: %s"
        % (profile, map_name, executable),
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=str(executable.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    startup_timeout = float(settings.get("startup_timeout_seconds", 360.0))
    poll_interval = max(0.5, float(settings.get("poll_interval_seconds", 2.0)))
    deadline = time.time() + startup_timeout
    last_error: Optional[BaseException] = None
    while time.time() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "Simulator exited before RPC became ready (exit=%s, executable=%s)"
                % (return_code, executable)
            )
        try:
            server_version, current_map = _probe(
                carla_module, host, port, probe_timeout
            )
            session = {
                "started": True,
                "managed": True,
                "profile": profile,
                "executable": str(executable),
                "pid": int(process.pid),
                "server_version": server_version,
                "current_map": current_map,
                "stop_after_collection": bool(
                    settings.get("stop_after_collection", True)
                ),
                "closed": False,
                "_process": process,
                "_pids": [
                    value
                    for value in (
                        int(process.pid),
                        _windows_listener_pid(int(port)),
                    )
                    if value is not None
                ],
                "host": str(host),
                "port": int(port),
                "shutdown_timeout_seconds": float(
                    settings.get("shutdown_timeout_seconds", 30.0)
                ),
            }
            if bool(session["stop_after_collection"]):
                atexit.register(close_simulator, session)
            print(
                "[SIM] Ready at %s:%d (server=%s, map=%s); stop_after_collection=%s"
                % (
                    host,
                    int(port),
                    server_version,
                    current_map,
                    session["stop_after_collection"],
                ),
                flush=True,
            )
            return session
        except Exception as error:
            last_error = error
            time.sleep(poll_interval)

    close_simulator(
        {
            "started": True,
            "managed": True,
            "closed": False,
            "_process": process,
            "_pids": [int(process.pid)],
            "host": str(host),
            "port": int(port),
            "shutdown_timeout_seconds": float(
                settings.get("shutdown_timeout_seconds", 30.0)
            ),
        }
    )
    raise TimeoutError(
        "Simulator did not become ready within %.1f seconds; last probe: %s"
        % (startup_timeout, last_error)
    )
