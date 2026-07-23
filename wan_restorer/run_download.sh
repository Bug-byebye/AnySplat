#!/bin/bash
# Download and verify Wan2.2 model
source /home-ldap/sunchang/miniconda3/etc/profile.d/conda.sh
conda activate wan_restorer
python /home-ldap/sunchang/3dProjects/GuassDiff/post/wan_restorer/download_model.py
echo "---DOWNLOAD COMPLETE---"
python /home-ldap/sunchang/3dProjects/GuassDiff/post/wan_restorer/test_cache.py
echo "---VERIFICATION COMPLETE---"
