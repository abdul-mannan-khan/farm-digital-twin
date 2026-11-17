# Farm digital twin: NVIDIA Isaac Sim 6.2 with the scene code and its CC0 assets baked in.
#
#   docker build -f docker/Dockerfile -t farm-twin .
#   docker run --rm --gpus all -v "$PWD/out:/out" farm-twin preview     # one still per shot  -> out/preview/
#   docker run --rm --gpus all -v "$PWD/out:/out" farm-twin render      # 60 s film, mp4 + gif -> out/
#
# Needs an NVIDIA RTX GPU, a recent NVIDIA driver and the NVIDIA Container Toolkit on the host. Pulling the base
# image needs an NGC account (nvcr.io) and accepting the Isaac Sim licence (the ENV lines below accept it).
FROM nvcr.io/nvidia/isaac-sim:6.2.0

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    NVIDIA_DRIVER_CAPABILITIES=all \
    FARM_ASSETS=/farm/assets \
    FARM_OUT=/out

USER root
# ffmpeg encodes the film; the Isaac Sim base image does not ship it
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg wget curl     && rm -rf /var/lib/apt/lists/*
WORKDIR /farm
COPY twin/ /farm/twin/
COPY tests/ /farm/tests/
COPY docker/entrypoint.sh /farm/entrypoint.sh

# Poly Haven textures and HDRI (CC0, ~250 MB). Build with --build-arg FETCH_ASSETS=0 to download them at run time.
ARG FETCH_ASSETS=1
RUN if [ "$FETCH_ASSETS" = "1" ]; then bash /farm/twin/get_assets.sh "$FARM_ASSETS"; fi

VOLUME ["/out"]
ENTRYPOINT ["bash", "/farm/entrypoint.sh"]
CMD ["preview"]
