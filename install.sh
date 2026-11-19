#!/bin/sh
# The one install: Magpie's own small environment, then `magpie.py setup`, which finds out what this machine is and installs what fits.
#   ./install.sh                       the modules marked auto (text generation), if they fit
#   ./install.sh --for text,speech     what this node is for
#   ./install.sh --plan                report only; install later from here, or from a portal
#   ./install.sh --allow-system        may install system packages (build tools, drivers); on Termux and in containers it may already
#   ./install.sh --autostart            start the node with the machine from now on (systemd, or Termux:Boot); magpie.py autostart off undoes it
set -e
cd "$(dirname "$0")"
# Run with sudo, it goes on as the user who asked: the node runs as that user, so everything it later changes (environments, models, state) has to be theirs. sudo is used only for the system packages it asks for.
if [ "$(id -u)" = 0 ] && [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != root ]; then exec sudo -u "$SUDO_USER" -H sh "$0" "$@"; fi
# Steps that install system packages cannot ask for a password later, so it is asked once here.
case " $* " in *" --allow-system "*) [ "$(id -u)" = 0 ] || sudo -v ;; esac
command -v python3 >/dev/null || { [ -n "$PREFIX" ] && pkg install -y python; } || { echo "python3 is needed"; exit 1; }
# A venv that failed halfway leaves a folder with no pip, so pip is what counts as having one.
venv() { rm -rf .venv; python3 -m venv .venv >/dev/null 2>&1; }
if [ ! -x .venv/bin/pip ] && ! venv; then
  case " $* " in *" --allow-system "*) command -v apt-get >/dev/null && { [ "$(id -u)" = 0 ] && apt-get install -y python3-venv || sudo apt-get install -y python3-venv; } ;; esac
  venv || { rm -rf .venv; echo "Python's venv module is needed: sudo apt-get install python3-venv (or run ./install.sh --allow-system)"; exit 1; }
fi
.venv/bin/pip install --quiet --no-cache-dir -r requirements.txt
.venv/bin/python magpie.py setup "$@"
case " $* " in *" --autostart "*) echo "Key: data/state/first_admin.key, or MAGPIE_ADMIN_KEY in magpie.env" ;; *) echo "Run it: ./run.sh (or start it with the machine: .venv/bin/python magpie.py autostart on)    Key: data/state/first_admin.key, or set MAGPIE_ADMIN_KEY" ;; esac