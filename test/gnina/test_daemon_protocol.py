#!/usr/bin/env python3

"""Smoke test for GNINA daemon protocol wire compatibility."""

import subprocess
import sys


gnina = sys.argv[1]  # path to gnina executable


def read_nonempty_line(proc: subprocess.Popen[bytes]) -> str:
    assert proc.stdout is not None
    while True:
        raw = proc.stdout.readline()
        if not raw:
            raise RuntimeError("gnina daemon terminated unexpectedly")
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            return line


cmd = [
    gnina,
    "--receptor",
    "data/C.xyz",
    "--score_only",
    "--cnn_scoring",
    "none",
    "--no_gpu",
    "--daemon_protocol",
    "--quiet",
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)

try:
    ready = read_nonempty_line(proc)
    assert ready.startswith("DAEMON\tREADY")

    payload = open("data/C8bent.sdf", "rb").read()
    req = f"JOBDATA t1 sdf {len(payload)}\n".encode("utf-8") + payload + b"\n"
    assert proc.stdin is not None
    proc.stdin.write(req)
    proc.stdin.flush()

    done = read_nonempty_line(proc)
    assert done.startswith("DAEMON\tDONE\tt1\t")
    parts = done.split("\t")
    assert len(parts) >= 7
    float(parts[3])  # enqueued
    float(parts[4])  # affinity
    float(parts[5])  # cnnscore
    float(parts[6])  # cnnaffinity

    proc.stdin.write(b"JOBDATA bad sdf X\n")
    proc.stdin.flush()
    err = read_nonempty_line(proc)
    assert err.startswith("DAEMON\tERROR\t-\t")

    proc.stdin.write(b"QUIT\n")
    proc.stdin.flush()
    bye = read_nonempty_line(proc)
    assert bye.startswith("DAEMON\tBYE")
finally:
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
