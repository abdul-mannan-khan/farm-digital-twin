#!/bin/bash
# Commands: preview | render | frames <n,n,...> | test | shell
set -e
A=${FARM_ASSETS:-/farm/assets}
O=${FARM_OUT:-/out}
PY=/isaac-sim/python.sh
mkdir -p "$O"
[ -s "$A/farm_field_hdri_4k.exr" ] || bash /farm/twin/get_assets.sh "$A"
# some cloud hosts mount the NVIDIA compute libraries but not the graphics/RTX ones; add them if missing
if [ -n "$FARM_FIX_DRIVER" ]; then bash /farm/twin/fix_driver.sh; fi

case "${1:-preview}" in
  preview)
    $PY /farm/twin/farm.py --assets "$A" --out "$O/preview" --preview ;;
  frames)
    $PY /farm/twin/farm.py --assets "$A" --out "$O/frames" --frames "$2" ;;
  render)
    rm -rf "$O/frames"
    $PY /farm/twin/farm.py --assets "$A" --out "$O/frames" ${RES:+--res $RES} ${SUBFRAMES:+--subframes $SUBFRAMES}
    ffmpeg -y -loglevel error -framerate 30 -i "$O/frames/%04d.png" -c:v libx264 -preset slow -crf 18 \
           -pix_fmt yuv420p -movflags +faststart "$O/farm_twin.mp4"
    ffmpeg -y -loglevel error -i "$O/farm_twin.mp4" -c:v libx264 -preset slow -crf 26 -pix_fmt yuv420p \
           -movflags +faststart "$O/farm_twin_web.mp4"
    ffmpeg -y -loglevel error -i "$O/farm_twin.mp4" -vf "setpts=PTS/2,fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
           "$O/farm_twin.gif"
    ls -la "$O"/farm_twin* ;;
  test)
    $PY /farm/tests/run_tests.py ;;
  shell)
    exec bash ;;
  *)
    echo "usage: preview | render | frames <n,n,...> | test | shell"; exit 2 ;;
esac
