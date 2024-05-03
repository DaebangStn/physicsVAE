#!/bin/bash


CFG_TRAIN="configs/train/skillAlgo_sword.yaml"
CFG_ENV="configs/env/keypointMaxObsTask_no_sword.yaml"

# Initialize variables
MEMO=""
OPT_FOUND=0

# Parse command-line options using getopts
while getopts ":m:" opt; do
  case ${opt} in
    m )
      MEMO=$OPTARG
      OPT_FOUND=1
      ;;
    \? )
      echo "Usage: $0 [-m <memo>]"
      exit 1
      ;;
  esac
done


if [ $OPT_FOUND -eq 1 ]; then
  echo "python run.py --cfg_train $CFG_TRAIN --cfg_env $CFG_ENV --memo $MEMO"
  python run.py --cfg_train $CFG_TRAIN --cfg_env $CFG_ENV --memo "$MEMO"
else
  echo "python run.py --cfg_train $CFG_TRAIN --cfg_env $CFG_ENV"
  python run.py --cfg_train $CFG_TRAIN --cfg_env $CFG_ENV
fi
