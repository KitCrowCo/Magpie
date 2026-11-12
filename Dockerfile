# Optional: the same one install, in a container. The base holds only what most drivers build with; drivers install into the mounted data folder, so they survive a rebuild.
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN python3 -m venv .venv && .venv/bin/pip install --no-cache-dir -r requirements.txt
ENV MAGPIE_DATA=/data
EXPOSE 8790
CMD ["sh", "-c", ".venv/bin/python magpie.py setup ${MAGPIE_SETUP:-} && exec .venv/bin/python magpie.py"]
