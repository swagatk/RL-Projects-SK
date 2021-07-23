#!/bin/bash
# repeat the execution of a command for multiple times

echo "PPO: RaceCar"
for i in {1..3};
do
  echo "Run $i ...."; 
  echo "Run $i ...." >> outlog.txt;
  #python ./common/data_plot.py >> run_log.txt 2>>error.txt;
  python ./main.py >> rcppo_log.txt 2>> error.txt;
  #python ./main_mujoco.py >> freach_ppo_log.txt 2>> error.txt;
  #python ./main_mujoco.py >> outlog.txt 
  echo "###############" >> outlog.txt;
  # sleep 5
done

