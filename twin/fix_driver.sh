#!/bin/bash
# Add the NVIDIA graphics/Vulkan/OptiX/NGX user-space libraries the container host did not mount
# (version matched to the host kernel driver). Safe to re-run.
set -e
V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
echo "driver $V"
L=/usr/lib/x86_64-linux-gnu
if [ -e $L/libGLX_nvidia.so.$V ] && [ -e $L/libnvidia-ngx.so.$V ] && [ -e $L/libnvoptix.so.$V ]; then echo "libs present"; exit 0; fi
cd /root
[ -f NVIDIA-Linux-x86_64-$V.run ] || wget -q https://download.nvidia.com/XFree86/Linux-x86_64/$V/NVIDIA-Linux-x86_64-$V.run
rm -rf drv && sh NVIDIA-Linux-x86_64-$V.run --extract-only --target drv >/dev/null
for lib in libGLX_nvidia libnvidia-glcore libnvidia-glvkspirv libnvidia-gpucomp libnvidia-rtcore libnvidia-tls \
           libnvidia-glsi libnvidia-eglcore libnvidia-allocator libEGL_nvidia libnvoptix libnvidia-ngx; do
  f=drv/$lib.so.$V
  [ -f "$f" ] || { echo "missing $f"; continue; }
  [ -e $L/$lib.so.$V ] || cp "$f" $L/
  ln -sf $lib.so.$V $L/$lib.so.1
  ln -sf $lib.so.$V $L/$lib.so.0
done
mkdir -p /usr/share/nvidia && cp -n drv/nvoptix.bin /usr/share/nvidia/ 2>/dev/null || true
if [ ! -e /etc/vulkan/icd.d/nvidia_icd.json ] && [ ! -e /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  mkdir -p /etc/vulkan/icd.d && cp drv/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null || true
fi
ldconfig
echo "driver libs installed"
