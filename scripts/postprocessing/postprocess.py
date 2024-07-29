import os
import json
import util
import argparse

parser = argparse.ArgumentParser(description='Process logfiles.')
parser.add_argument('datadir', type=str, help='Directory containing log files', default=".")
parser.add_argument('outdir', type=str, help='Directory to save JSON files', default=".")

args = parser.parse_args()
datadir = args.datadir
outdir = args.outdir

os.makedirs(outdir, exist_ok=True)

files=[x for x in os.listdir(datadir) if x.endswith(".txt") and x.startswith("run_log_fcs")]
files.sort()
files=[os.path.join(datadir,file) for file in files]

for file in files:
    
    data = util.parse_run_log(file)
    
    base = os.path.basename(file)
    tag, _ = os.path.splitext(base)
    json_filename = tag + ".json"
    json_filepath = os.path.join(outdir, json_filename)

    with open(json_filepath, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True, default=util.json_serial)