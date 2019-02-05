#!/usr/bin/env bash
nc run -C GPU9 python /data/projects/swat/users/haih/GNAS/main.py --config_file /data/projects/swat/users/haih/GNAS/configs/study_lr/config_lr_1.json
nc run -C GPU9 python /data/projects/swat/users/haih/GNAS/main.py --config_file /data/projects/swat/users/haih/GNAS/configs/study_lr/config_lr_05.json
nc run -C GPU9 python /data/projects/swat/users/haih/GNAS/main.py --config_file /data/projects/swat/users/haih/GNAS/configs/study_lr/config_lr_15.json
