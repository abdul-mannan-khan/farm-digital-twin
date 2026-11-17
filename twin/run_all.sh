#!/bin/bash
# full render -> encode; each step logs an end marker; DONE_ALL arms the self-destroy watchdog
# expects this repository's twin/ folder at /root/twin
cd /root
command -v ffmpeg >/dev/null || (apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg)
rm -rf frames && mkdir frames
/isaac-sim/python.sh twin/farm.py --assets /root/assets --out frames > render.log 2>&1; echo "EXIT render $?" >> render.log
ls frames/*.png | wc -l >> render.log
ffmpeg -y -loglevel error -framerate 30 -i frames/%04d.png -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart farm_twin.mp4 >> render.log 2>&1; echo "EXIT mp4 $?" >> render.log
ffmpeg -y -loglevel error -i farm_twin.mp4 -c:v libx264 -preset slow -crf 26 -pix_fmt yuv420p -movflags +faststart farm_twin_web.mp4 >> render.log 2>&1; echo "EXIT web $?" >> render.log
ffmpeg -y -loglevel error -i farm_twin.mp4 -vf "setpts=PTS/2,fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" farm_twin.gif >> render.log 2>&1; echo "EXIT gif $?" >> render.log
md5sum farm_twin.mp4 farm_twin_web.mp4 farm_twin.gif twin/farm.py twin/plants.py twin/timeline.py twin/arm.py > md5.txt
echo "EXIT ALL" >> render.log
touch /root/DONE_ALL
