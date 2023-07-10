import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--input_dir", "-i", type=str, help="Directory containing the root files to merge", required=True)
parser.add_argument("--output_file", "-o", type=str, help="Output root file name", required=True)


args = parser.parse_args()
import os
id = args.input_dir

files = [os.path.join(id, x) for x in os.listdir(id)]

cmd = "$ROOTSYS/bin/hadd"+ args.output_file + " " + " ".join(files)



import subprocess
subprocess.run(["/bin/hadd", args.output_file]+files)

