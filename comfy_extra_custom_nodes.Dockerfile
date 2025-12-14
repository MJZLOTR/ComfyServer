# syntax=docker/dockerfile:1.10

ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim-bookworm
ARG VENV_PATH=/prod_venv
ARG COMFY_PATH=/opt/ComfyUI
ARG COMFY_VERSION_TAG=v0.3.56
ARG COMFY_GIT_URL=https://github.com/comfyanonymous/ComfyUI.git

FROM ${BASE_IMAGE} AS builder

ARG VENV_PATH
ARG COMFY_PATH
ARG COMFY_VERSION_TAG
ARG COMFY_GIT_URL

# Create venv and ensure PATH points to it for subsequent RUN and pip
RUN python -m venv ${VENV_PATH}
ENV VIRTUAL_ENV=${VENV_PATH}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev curl build-essential git \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# ========== Install Comfy ==========
RUN git clone --depth 1 --branch ${COMFY_VERSION_TAG} ${COMFY_GIT_URL} ${COMFY_PATH} \
  && rm -rf ${COMFY_PATH}/.git

RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r ${COMFY_PATH}/requirements.txt


# ========== Install kserve dependencies ==========
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =================== Final stage ===================
FROM ${BASE_IMAGE} AS prod

ARG VENV_PATH
ARG COMFY_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app

RUN useradd kserve -m -u 1000 -d /home/kserve

COPY --from=builder --chown=kserve:kserve ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=builder --chown=kserve:kserve ${COMFY_PATH} ${COMFY_PATH}


# ========== Install custome nodes ==========
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev curl build-essential python3-opencv git \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git ${COMFY_PATH}/custom_nodes/ComfyUI-Easy-Use && \
    git clone --depth 1 https://github.com/theUpsider/ComfyUI-Logic.git ${COMFY_PATH}/custom_nodes/ComfyUI-Logic && \
    git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor.git ${COMFY_PATH}/custom_nodes/ComfyUI-ReActor && \
    git clone --depth 1 https://github.com/WASasquatch/was-node-suite-comfyui.git ${COMFY_PATH}/custom_nodes/was-node-suite-comfyui && \
    rm -rf ${COMFY_PATH}/custom_nodes/*/.git


RUN pip install -r ${COMFY_PATH}/custom_nodes/ComfyUI-Easy-Use/requirements.txt
RUN pip install -r ${COMFY_PATH}/custom_nodes/ComfyUI-ReActor/requirements.txt
RUN pip install -r ${COMFY_PATH}/custom_nodes/was-node-suite-comfyui/requirements.txt

RUN chown -R kserve:kserve ${COMFY_PATH}
# ==============================================


COPY --chown=kserve:kserve ./comfyserver ./comfyserver
USER 1000
ENV PYTHONPATH="${COMFY_PATH}:/app"
ENV COMFY_PATH="${COMFY_PATH}"
ENTRYPOINT ["python", "-m", "comfyserver"]
