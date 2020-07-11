#!/usr/bin/env python3
import shutil
import os
import sys

dr = sys.argv[1]
target_name, target_ext = os.path.splitext(sys.argv[2])

for root, dirs, files in os.walk(dr):
    for file in files:
        if file == target_name + target_ext:
            spl = root.split("/"); newname = spl[-1]; sup = ("/").join(spl[:-1])
            shutil.move(root+"/"+file, sup+"/"+newname+target_ext);