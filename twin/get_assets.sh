#!/bin/bash
# CC0 textures and HDRI from Poly Haven (polyhaven.com), downloaded into $1 (default /root/assets)
D=${1:-/root/assets}; mkdir -p $D; cd $D
T=https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k
for t in brown_mud_02 brown_mud_03 brown_mud_dry sparse_grass leafy_grass aerial_grass_rock aerial_mud_1; do
  for k in diff nor_gl rough; do [ -s ${t}_${k}_2k.jpg ] || wget -q $T/$t/${t}_${k}_2k.jpg; done
done
for h in farm_field countrytrax_midday; do
  [ -s ${h}_hdri_4k.exr ] || wget -q -O ${h}_hdri_4k.exr https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/4k/${h}_4k.exr
done
ls | wc -l
