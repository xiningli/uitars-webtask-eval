#!/usr/bin/with-contenv bash
# linuxserver/chromium binds CDP to 127.0.0.1 even with --remote-debugging-address=0.0.0.0,
# so we expose it externally via a socat proxy on 0.0.0.0:9223.
until curl -sS http://127.0.0.1:9222/json/version >/dev/null 2>&1; do
  sleep 1
done
echo "[cdp-proxy] chromium CDP up; forwarding 0.0.0.0:9223 -> 127.0.0.1:9222"
exec socat TCP-LISTEN:9223,reuseaddr,fork TCP:127.0.0.1:9222
