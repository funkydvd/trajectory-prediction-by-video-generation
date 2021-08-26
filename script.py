import cv2
import os
import sys
from shutil import copyfile

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def main(argv):
    dir1 = argv[0]
    dir2 = argv[1]
    i =0
    for root, dirs, files in os.walk(dir1):
            files = sorted_alphanumeric(files)
            i = 0
            
            for f in files:
                if "png" in f :
                    i=i+1
                    print(i)
                    print(f)
                    
                    aux = root.split("/")
                    f3 = aux[1] + "_" + aux[2] + "_" + f
                    f2 = os.path.join(dir2, f3)
                    fsrc = os.path.join(root,f)
                    copyfile(fsrc, f2)
                    print(f2)
                    
                    #dir11 = os.path.join(dir2,root[21:-nrchs])
                    #if not os.path.exists(dir11):
                    #	os.makedirs(dir11)
                    #f4 = os.path.join(dir2, root[21:-nrchs], nume2)
                    #cv2.imwrite(f4, resized)
                   # print(f4)

if __name__ == "__main__":
   main(sys.argv[1:])
