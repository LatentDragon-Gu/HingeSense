# -*- coding: utf-8 -*-
"""子进程并行调度与机器信息。

根目录脚本把每个（方法 × 折）的工作进程交给 run_jobs 按 --jobs / --gpus 运行：GPU 作业轮流分配到各 GPU
（CUDA_VISIBLE_DEVICES），CPU 作业屏蔽 GPU；每个作业的标准输出与错误写入其日志文件；任一作业失败即不再启动新作业，
最后汇报失败作业的日志路径。machine_info / write_provenance 记录产生结果的机器与软件版本。

Parallel job runner and machine information. The root scripts hand every (method × dataset) worker process to run_jobs,
which runs them according to --jobs / --gpus: GPU jobs rotate over the GPUs (CUDA_VISIBLE_DEVICES), CPU jobs hide the GPUs;
stdout and stderr of each job go to its log file; after a failure no new job is started and the failed logs are listed.
machine_info / write_provenance record the machine and software versions that produced the results.
"""
import datetime
import os
import platform
import subprocess
import sys
import time

from . import config as C


def machine_info():
    """操作系统、CPU、内存、GPU、Python / PyTorch / CUDA 版本。 / OS, CPU, RAM, GPU, Python / PyTorch / CUDA versions."""
    info = dict(os=platform.platform(), python=platform.python_version(), cpu="", n_cpu=os.cpu_count(), ram_gb=None,
                torch="", cuda="", gpus=[], driver="")
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / 2 ** 20, 1)
                break
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or ""
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            info["driver"] = out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    if not info["driver"]:
        try:
            first = open("/proc/driver/nvidia/version").readline().split()     # 内核模块版本（nvidia-smi 不可用时的后备）/ kernel-module version, fallback when nvidia-smi is unavailable
            info["driver"] = next((tok for tok in first if tok[0].isdigit() and "." in tok), "")
        except Exception:
            pass
    return info


def format_machine_info(info):
    return ["os: %s" % info["os"],
            "cpu: %s x%s, ram %s GB" % (info["cpu"], info["n_cpu"], info["ram_gb"]),
            "gpu: %s%s" % (", ".join(info["gpus"]) or "none", (" (driver %s)" % info["driver"]) if info["driver"] else ""),
            "python %s, torch %s, cuda %s" % (info["python"], info["torch"], info["cuda"])]


def parse_gpus(spec):
    """--gpus 参数 → GPU 编号列表；空串表示自动（有 CUDA 则全部可见 GPU，否则无）；"none" 表示只用 CPU。
    --gpus value → list of GPU ids; empty = automatic (all visible GPUs when CUDA is available); "none" = CPU only."""
    if spec is None or spec == "":
        try:
            import torch
            return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        except Exception:
            return []
    if spec.lower() in ("none", "cpu"):
        return []
    return [int(t) for t in spec.split(",") if t.strip()]


def run_jobs(jobs, n_jobs=1, gpus=(), dry_run=False, poll_s=2.0):
    """运行作业列表。job = dict(name, cmd(list), log(path), gpu(bool))。返回 (成功数, 失败作业列表)。
    Run a list of jobs, job = dict(name, cmd(list), log(path), gpu(bool)); returns (number succeeded, failed jobs)."""
    gpus = list(gpus)
    if dry_run:
        for j in jobs:
            print("[dry-run] %s\n    %s\n    log: %s" % (j["name"], " ".join(j["cmd"]), j["log"]))
        return 0, []
    pending = list(jobs)
    running = []
    failures = []
    n_ok = 0
    slot = 0
    t_all = time.time()
    while pending or running:
        while pending and len(running) < max(1, n_jobs) and not failures:
            j = pending.pop(0)
            env = dict(os.environ)
            if j.get("gpu") and gpus:
                env["CUDA_VISIBLE_DEVICES"] = str(gpus[slot % len(gpus)])
                slot += 1
            elif not j.get("gpu"):
                env["CUDA_VISIBLE_DEVICES"] = ""
            os.makedirs(os.path.dirname(os.path.abspath(j["log"])), exist_ok=True)
            lf = open(j["log"], "w", encoding="utf-8")
            p = subprocess.Popen(j["cmd"], stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=C.ROOT_DIR)
            running.append((p, j, lf, time.time()))
            print("[start %s] %s → %s" % (datetime.datetime.now().strftime("%H:%M:%S"), j["name"], os.path.relpath(j["log"], C.ROOT_DIR)), flush=True)
        time.sleep(poll_s)
        still = []
        for p, j, lf, t0 in running:
            rc = p.poll()
            if rc is None:
                still.append((p, j, lf, t0))
                continue
            lf.close()
            if rc == 0:
                n_ok += 1
                print("[done  %s] %s (%.0f s)" % (datetime.datetime.now().strftime("%H:%M:%S"), j["name"], time.time() - t0), flush=True)
            else:
                failures.append(dict(job=j, returncode=rc))
                print("[FAILED %s] %s exit=%d  see %s" % (datetime.datetime.now().strftime("%H:%M:%S"), j["name"], rc, j["log"]), flush=True)
        running = still
        if failures and pending:
            print("stopping: %d job(s) not started because of the failure above" % len(pending), flush=True)
            pending = []
    print("finished %d job(s), %d failed, %.0f s total" % (n_ok, len(failures), time.time() - t_all), flush=True)
    return n_ok, failures


def worker_cmd(module, *args):
    """构造 python -m hingesense.<module> ... 命令（与当前解释器相同）。 / Build the python -m hingesense.<module> ... command with the current interpreter."""
    return [sys.executable, "-m", "hingesense." + module] + [str(a) for a in args]


def write_provenance(path, command, started, inputs=(), notes=()):
    """写 provenance.txt：命令、时间、耗时、机器与软件版本、输入清单。 / Write provenance.txt: command, time, duration, machine and software versions, inputs."""
    info = machine_info()
    lines = ["command: " + command,
             "started: " + started.strftime("%Y-%m-%d %H:%M:%S"),
             "elapsed_s: %.1f" % (datetime.datetime.now() - started).total_seconds()] + format_machine_info(info)
    if inputs:
        lines.append("inputs:")
        lines += ["  " + x for x in inputs]
    lines += list(notes)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
