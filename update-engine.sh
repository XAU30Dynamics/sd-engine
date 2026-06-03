#!/usr/bin/env bash
#
# StrategyDynamics — Backtest Engine updater
# Run on the VPS with ONE command:
#   curl -fsSL https://raw.githubusercontent.com/XAU30Dynamics/sd-engine/main/update-engine.sh | bash
#
# It downloads the latest engine, clears any stray process off the port,
# restarts the service, and confirms the NEW version is actually live.
#
set -euo pipefail

RAW="https://raw.githubusercontent.com/XAU30Dynamics/sd-engine/main"
TARGET="/root/vps-server/main.py"
PORT=8000
SVC=algotrader

echo "==> Downloading latest engine..."
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
curl -fSL "$RAW/main.py" -o "$tmp"                     # -f: fail on HTTP error, -L: follow redirects

# Sanity: must be the real engine, not an HTML error page or empty file
if ! grep -q 'FastAPI' "$tmp"; then
  echo "ERROR: download does not look like the engine (got an error page?). Aborting; nothing changed." >&2
  exit 1
fi
NEW="$(grep -oP 'version="\K[^"]+' "$tmp" | head -1 || echo unknown)"
echo "    Downloaded v$NEW"

echo "==> Stopping service and freeing port $PORT..."
systemctl stop "$SVC" 2>/dev/null || true
fuser -k "${PORT}/tcp" 2>/dev/null || true            # kill any orphan squatting on the port
sleep 1

echo "==> Installing new engine..."
cp "$tmp" "$TARGET"

echo "==> Starting service..."
systemctl start "$SVC"

echo "==> Verifying (waiting for engine to come up)..."
LIVE=""
for i in $(seq 1 30); do
  LIVE="$(curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -oP '"version":"\K[^"]+' || true)"
  [ -n "$LIVE" ] && break
  sleep 1
done

if [ "$LIVE" = "$NEW" ]; then
  echo "✅ Engine updated and live: v$LIVE"
elif [ -n "$LIVE" ]; then
  echo "⚠️  Live version (v$LIVE) does not match downloaded (v$NEW)." >&2
  echo "    Check: systemctl status $SVC   and   journalctl -u $SVC -n 50" >&2
  exit 1
else
  echo "⚠️  Engine did not respond on port $PORT within 30s." >&2
  echo "    Check: systemctl status $SVC   and   journalctl -u $SVC -n 50" >&2
  exit 1
fi
