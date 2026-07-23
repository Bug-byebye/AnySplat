#!/bin/bash
# Run the full V3 demo
cd /home-ldap/sunchang/3dProjects/GuassDiff
export PYTHONPATH=/home-ldap/sunchang/3dProjects/GuassDiff/post:$PYTHONPATH
/home-ldap/sunchang/miniconda3/envs/wan_restorer/bin/python \
  /home-ldap/sunchang/3dProjects/GuassDiff/post/wan_restorer/demo_inference.py \
  --num_views 2 --num_dit_layers 4 \
  > /tmp/wan_demo.log 2>&1
echo "EXIT CODE: $?" >> /tmp/wan_demo.log
