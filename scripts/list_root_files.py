import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--input_dir", "-i", type=str, help="Directory containing the root files to merge", required=True)
parser.add_argument("--output_file", "-o", type=str, help="Output file list name", required=True)


args = parser.parse_args()


import os

flist = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir)]

out = args.output_file

f = open(out, "w")

for fl in flist:
	f.write(fl + "\n")

f.close()
print("DONE")

