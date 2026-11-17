#!/bin/bash
# Self-destroy for a rented Vast.ai render box: 45 min after the job writes /root/DONE_ALL (time to download the
# results), and in any case after MAX_H hours. Uses the instance's own scoped API key that Vast injects.
source <(sed "s/^/export /" /root/.vastenv)
MAX_H=${1:-6}
destroy() { curl -s -X DELETE -H "Authorization: Bearer $CONTAINER_API_KEY" \
              "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/"; }
end=$(( $(date +%s) + MAX_H * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  if [ -f /root/DONE_ALL ]; then sleep 2700; echo "job done: destroying $(date)"; destroy; exit 0; fi
  sleep 60
done
echo "time limit: destroying $(date)"; destroy
