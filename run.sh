#!/bin/sh
# Runs the node, restarting it if it exits. On a phone it also holds a wake lock, if Termux:API is there.
cd "$(dirname "$0")"
export MAGPIE_SUPERVISED=1   # this loop starts the node again when it exits, so a portal may ask it to restart
command -v termux-wake-lock >/dev/null && termux-wake-lock
while true; do .venv/bin/python magpie.py; echo "[run] node exited, restarting in 5 s"; sleep 5; done
