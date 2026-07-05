FROM lscr.io/linuxserver/chromium:latest

# Expose Chrome DevTools Protocol so the host-side VLA agent can drive the browser
# over CDP. linuxserver/chromium passes CHROME_CLI through to the chromium command.
# Note: chromium binds CDP to 127.0.0.1 regardless of --remote-debugging-address,
# and rejects cross-origin connections without --remote-allow-origins=*.
ENV CHROME_CLI="--remote-debugging-port=9222 --remote-allow-origins=* --no-first-run --no-default-browser-check --disable-features=TranslateUI"

# socat is used to bridge 0.0.0.0:9223 -> 127.0.0.1:9222 from inside the container.
RUN apt-get update \
 && apt-get install -y --no-install-recommends socat curl \
 && rm -rf /var/lib/apt/lists/*

# linuxserver custom-services run as long-lived s6 services (see lscr docs).
COPY cdp-proxy.sh /custom-services.d/cdp-proxy
RUN chmod +x /custom-services.d/cdp-proxy

# 3000 = web UI (KasmVNC); 9223 = CDP (proxied)
EXPOSE 3000 9223
