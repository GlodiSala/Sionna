#!/bin/sh                                                                                      
#SBATCH --nodes=1                                                                                
# number of GPUs                                                                                 
#SBATCH --gres=gpu:1                                                                             
# number of CPUs for each GPU                                                                    
#SBATCH --cpus-per-gpu=4                                                                         
# amount of main memory in MB                                                                    
#SBATCH --mem=4096                                                                               
#SBATCH --job-name=gpu_test                                                                    
# maximum run time HH:MM:SS                                                                      
#SBATCH --time=00:01:00  echo "Job started on $(date)"
