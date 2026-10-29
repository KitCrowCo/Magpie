# Magpie - stateful llama.cpp and colibri supervisor (Ollama wire-compatible) for AMD APUs.
# Backend is switchable at build time: LLAMA_BACKEND=hip (default, ROCm) or vulkan (Mesa/RADV).
#   hip    - kernels target gfx1100; gfx1103 (760M/780M) runs them via HSA_OVERRIDE_GFX_VERSION=11.0.0 (the planner's rules set it).
#   vulkan - one FOSS driver path (RADV), no HSA override, no managed-memory env; the planner's hip-only rules skip themselves when it sees the vulkan backend.
# Build HIP (unchanged from before):     docker build -t magpie .
# Build Vulkan:                          docker build -t magpie --build-arg LLAMA_BACKEND=vulkan --build-arg COLIBRI_VK=1 .
# Stage 1: build llama-server.
FROM rocm/dev-ubuntu-22.04:6.1.2 AS builder

ARG AMDGPU_TARGETS=gfx1100
ARG LLAMA_BACKEND=hip
# Pin a llama.cpp commit/tag here for reproducible builds (e.g. --build-arg LLAMA_CPP_REF=b6500); "master" = whatever is newest at build time.
ARG LLAMA_CPP_REF=master

# Base build deps always; the Vulkan build also needs a shader compiler (glslc) and the Vulkan headers/loader.
# glslc is not a standalone apt package on jammy - it comes from the LunarG Vulkan SDK, which is what provides a build-usable glslc here.
RUN apt-get update && apt-get install -y git cmake build-essential libssl-dev libcurl4-openssl-dev hipblas-dev hipblaslt-dev && \
    if [ "$LLAMA_BACKEND" = "vulkan" ]; then apt-get install -y wget gnupg && wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc >/dev/null && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list && apt-get update && apt-get install -y vulkan-sdk; fi && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp && git checkout ${LLAMA_CPP_REF} && git log -1 --format="llama.cpp build ref: %H %cd" | tee /build/LLAMA_CPP_REF.txt

WORKDIR /build/llama.cpp

# Aggregate-initialization syntax fix for ROCm 6.1's clang 17 in the CUDA/HIP sources (no-op for the Vulkan build, which does not compile them).
RUN find ggml/src/ -type f \( -name "*.cu" -o -name "*.cuh" \) | xargs -r sed -i 's/{{0.0f, 0.0f}}/{}/g'

# One source tree, backend chosen by LLAMA_BACKEND. HIP: kernels for AMDGPU_TARGETS, BF16 template generation off (ROCm 6.1 compile failures). Vulkan: SPIR-V shaders, no arch to pick.
# -j1 keeps peak build RAM low on 16 GB nodes - raise if the build node has room.
RUN BK="-DGGML_HIP=ON -DGGML_HIP_NO_VMM=ON -DGGML_CUDA_ENABLE_BF16=OFF -DAMDGPU_TARGETS=${AMDGPU_TARGETS}"; [ "$LLAMA_BACKEND" = "vulkan" ] && BK="-DGGML_VULKAN=ON"; mkdir build && cd build && CXX=/opt/rocm/llvm/bin/clang++ CC=/opt/rocm/llvm/bin/clang cmake .. $BK -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release --target llama-server -j1
# The kernel targets this build carries; the planner's startup check uses it only on the hip backend (it compares them with the GPU and HSA_OVERRIDE_GFX_VERSION), and skips the check on vulkan.
RUN echo "${AMDGPU_TARGETS}" > /build/AMDGPU_TARGETS.txt

# Stage 2: build the LoRA trainer for the draft slots (aux_servers.py). Upstream llama.cpp only fine-tunes whole models in F32;
# this fork adds llama-finetune-lora, which trains a LoRA adapter on a frozen quantized model. Its adapters load with llama-server --lora.
# Built separately from llama-server and installed in its own folder, so its ggml libraries never replace the ones llama-server uses.
# Independent of LLAMA_BACKEND: TRAIN_BACKEND=cpu (default) is the backend the fork reports working; hip is untested there - try it with --build-arg TRAIN_BACKEND=hip.
# LORA_TRAIN_REF: a branch, tag or commit of LORA_TRAIN_REPO; if the branch has been deleted since its merge, use the fork's default branch or a commit hash.
FROM rocm/dev-ubuntu-22.04:6.1.2 AS trainer

ARG AMDGPU_TARGETS=gfx1100
ARG LORA_TRAIN_REPO=https://github.com/CambrianTech/llama.cpp.git
ARG LORA_TRAIN_REF=feat/props-weight-residency
ARG TRAIN_BACKEND=cpu

RUN apt-get update && apt-get install -y git cmake build-essential libssl-dev libcurl4-openssl-dev hipblas-dev hipblaslt-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone ${LORA_TRAIN_REPO} llama.cpp && cd llama.cpp && git checkout ${LORA_TRAIN_REF} && git log -1 --format="lora trainer ref: %H %cd" | tee /build/LORA_TRAIN_REF.txt

WORKDIR /build/llama.cpp

RUN find ggml/src/ -type f \( -name "*.cu" -o -name "*.cuh" \) | xargs -r sed -i 's/{{0.0f, 0.0f}}/{}/g'

RUN HIP=""; if [ "$TRAIN_BACKEND" = "hip" ]; then HIP="-DGGML_HIP=ON -DGGML_HIP_NO_VMM=ON -DGGML_CUDA_ENABLE_BF16=OFF -DAMDGPU_TARGETS=${AMDGPU_TARGETS}"; fi && mkdir build && cd build && CXX=/opt/rocm/llvm/bin/clang++ CC=/opt/rocm/llvm/bin/clang cmake .. $HIP -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release --target llama-finetune-lora -j1

# Stage 3: colibri - the engine for models llama.cpp does not run (engines.py starts it on demand).
# COLIBRI_VK=0 (default): CPU build, about 10 s, no GPU libraries. COLIBRI_VK=1: build each engine with VK=1 (Vulkan, any 1.2 driver e.g. Mesa/RADV) - a VK=1 binary still runs on CPU and uses the GPU only when COLI_VULKAN=1 is in its environment, so it is safe to build with VK=1 and decide per run.
# COLIBRI_REF: a release tag or commit. COLIBRI_ARCH: x86-64-v3 runs on any AVX2 CPU; x86-64-v4 adds AVX-512 (Zen 4 / Zen 5); native only when the image is built on the node that runs it.
# Built from source: colibri's own setup script (which downloads from GitHub releases and Hugging Face) is never run.
FROM ubuntu:22.04 AS colibri

ARG COLIBRI_REF=v2.0.0
ARG COLIBRI_ARCH=x86-64-v3
ARG COLIBRI_VK=1
# Engine binaries to build; coli serve picks the right one from a model's config.json. deepseek_v41 = DeepSeek V4/V4.1 Flash; qwen36 = Qwen3.6, Qwen3-Coder, Qwen3.8-27B. Add qwen38 for Qwen3.8-Flash-Next, glm53 for GLM-5.3-Flash, etc.
ARG COLIBRI_ENGINES="deepseek_v41 qwen36"

# Vulkan build needs glslc (LunarG SDK, as above) to compile colibri's SPIR-V shaders.
RUN apt-get update && apt-get install -y git build-essential && \
    if [ "$COLIBRI_VK" = "1" ]; then apt-get install -y wget gnupg && wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc >/dev/null && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list && apt-get update && apt-get install -y vulkan-sdk; fi && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/JustVugg/colibri /opt/colibri && cd /opt/colibri && git checkout ${COLIBRI_REF} && git log -1 --format="colibri ref: %H %cd" | tee /opt/colibri/COLIBRI_REF.txt
RUN for e in ${COLIBRI_ENGINES}; do make -C /opt/colibri/c $e ARCH=${COLIBRI_ARCH} $( [ "$COLIBRI_VK" = "1" ] && echo VK=1 ); done

# Stage 4: runtime
FROM rocm/dev-ubuntu-22.04:6.1.2

ARG LLAMA_BACKEND=hip
ARG COLIBRI_VK=1
RUN groupadd -g 992 gpu 2>/dev/null || true

# dmidecode: optional RAM-speed probe for the startup report (reads SMBIOS; runs as root in this container). libgomp1: OpenMP runtime for colibri.
# hipblas/rocblas are the ROCm runtime; harmless (unused) on a vulkan build, kept so one deps line serves both backends.
# When either engine uses Vulkan, install the RADV runtime driver. Jammy's stock Mesa (22.x) predates gfx1103 (Phoenix) Vulkan support, so pull a current Mesa from the kisak PPA - without this RADV does not see the 780M/760M and silently falls back to CPU (llvmpipe).
RUN apt-get update && apt-get install -y python3 python3-pip hipblas rocblas dmidecode libgomp1 && \
    if [ "$LLAMA_BACKEND" = "vulkan" ] || [ "$COLIBRI_VK" = "1" ]; then apt-get install -y software-properties-common && add-apt-repository -y ppa:kisak/kisak-mesa && apt-get update && apt-get install -y mesa-vulkan-drivers libvulkan1 vulkan-tools; fi && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# The whole build folder: main.py, optimizer.py, optimizer_rules.json, passive_worker.py, aux_servers.py, aux_servers.json, requirements.txt
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY --from=builder /build/llama.cpp/build/bin/ /app/
COPY --from=builder /build/LLAMA_CPP_REF.txt /build/AMDGPU_TARGETS.txt /app/
# Trainer and its own libraries; aux_servers.json train.env puts /app/trainer first on LD_LIBRARY_PATH for the trainer process only
COPY --from=trainer /build/llama.cpp/build/bin/ /app/trainer/
COPY --from=trainer /build/LORA_TRAIN_REF.txt /app/trainer/
# colibri (engines.json runs /opt/colibri/c/coli); its build record is /opt/colibri/COLIBRI_REF.txt
COPY --from=colibri /opt/colibri /opt/colibri

# LLAMA_BACKEND reaches the planner (optimizer.py info()): on vulkan it reports the vulkan backend, so the hip-only rules (HSA override, managed memory, forced flash-attn) skip and the gfx1103_vulkan rule applies instead.
ENV LLAMA_BACKEND=${LLAMA_BACKEND}
ENV LD_LIBRARY_PATH=/app:/opt/rocm/lib:${LD_LIBRARY_PATH}
EXPOSE 8790
CMD ["python3", "main.py"]
