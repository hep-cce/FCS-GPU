import re
import csv
import io
import os
import json
import pandas as pd
from datetime import datetime

def parse_description(return_dic, description):
    description_parts = description.split("-")
    for i in range(len(description_parts)):
        if description_parts[i][0].isdigit():
            return_dic['image_type'] = "-".join(description_parts[:i])
            return_dic['root_version'] = description_parts[i]
            return_dic['image_tag'] = "-".join(description_parts[i+1:])
            break

def parse_timestamp(return_dic, timestamp):
    timestamp_match = re.search(r'\d{12}', timestamp)
    if timestamp_match:
        timestamp_str = timestamp_match.group()
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        return_dic['timestamp'] = timestamp

def parse_logfile_name(file):
    info_dic = {}
    tag, _ = os.path.splitext(os.path.basename(file))
    _, _, description, timestamp = tag.split("_")
    parse_description(info_dic, description)
    parse_timestamp(info_dic, timestamp)
    return info_dic

def parse_run_cmd(file):
    command_match = re.search(r'runTFCSSimulation\s.*', file)

    if command_match:
        return command_match.group(0)
    else:
        return None
    
def parse_datapath(file):
    datapath_match = re.search(r'DATAPATH:\s*(.*)', file)
    if datapath_match:
        return datapath_match.group(1)
    else:
        return None
    
def remove_trailing_commas(json_string):
    # Remove trailing commas before closing braces and brackets
    json_string = re.sub(r',\s*([\]}])', r'\1', json_string)
    return json_string

def parse_system_info(info_dic, file):
    json_string = str()
    hits = 0
    for line in file.splitlines():
        if "{" in line:
            # print("hit")
            hits += 1

        if hits >= 1:
            # print(json_string)
            json_string += line

        if "}" in line:
            # print("miss")
            hits -= 1
    
    if json_string:
        json_string = remove_trailing_commas(json_string)
        return json.loads(json_string)
    else:
        print(f"No system information found for {info_dic['image_type']} with tag {info_dic['image_tag']}")
        return None

def parse_cpu_info(file):
    cpu_info = {}
    patterns = {
        'Architecture': r'Architecture:\s+(.+)',
        'CPU op-mode(s)': r'CPU op-mode\(s\):\s+(.+)',
        'Address sizes': r'Address sizes:\s+(.+)',
        'Byte Order': r'Byte Order:\s+(.+)',
        'CPU(s)': r'CPU\(s\):\s+(\d+)',
        'On-line CPU(s) list': r'On-line CPU\(s\) list:\s+(.+)',
        'Vendor ID': r'Vendor ID:\s+(.+)',
        'Model name': r'Model name:\s+(.+)',
        'CPU family': r'CPU family:\s+(\d+)',
        'Model': r'Model:\s+(\d+)',
        'Thread(s) per core': r'Thread\(s\) per core:\s+(\d+)',
        'Core(s) per socket': r'Core\(s\) per socket:\s+(\d+)',
        'Socket(s)': r'Socket\(s\):\s+(\d+)',
        'Stepping': r'Stepping:\s+(\d+)',
        'Frequency boost': r'Frequency boost:\s+(.+)',
        'CPU max MHz': r'CPU max MHz:\s+([\d.]+)',
        'CPU min MHz': r'CPU min MHz:\s+([\d.]+)',
        'BogoMIPS': r'BogoMIPS:\s+([\d.]+)',
        'Flags': r'Flags:\s+(.+)',
        'Virtualization': r'Virtualization:\s+(.+)',
        'L1d cache': r'L1d cache:\s+(.+)',
        'L1i cache': r'L1i cache:\s+(.+)',
        'L2 cache': r'L2 cache:\s+(.+)',
        'L3 cache': r'L3 cache:\s+(.+)',
        'NUMA node(s)': r'NUMA node\(s\):\s+(\d+)',
        'NUMA node0 CPU(s)': r'NUMA node0 CPU\(s\):\s+(.+)',
        'NUMA node1 CPU(s)': r'NUMA node1 CPU\(s\):\s+(.+)',
        'Vulnerability Gather data sampling': r'Vulnerability Gather data sampling:\s+(.+)',
        'Vulnerability Itlb multihit': r'Vulnerability Itlb multihit:\s+(.+)',
        'Vulnerability L1tf': r'Vulnerability L1tf:\s+(.+)',
        'Vulnerability Mds': r'Vulnerability Mds:\s+(.+)',
        'Vulnerability Meltdown': r'Vulnerability Meltdown:\s+(.+)',
        'Vulnerability Mmio stale data': r'Vulnerability Mmio stale data:\s+(.+)',
        'Vulnerability Retbleed': r'Vulnerability Retbleed:\s+(.+)',
        'Vulnerability Spec rstack overflow': r'Vulnerability Spec rstack overflow:\s+(.+)',
        'Vulnerability Spec store bypass': r'Vulnerability Spec store bypass:\s+(.+)',
        'Vulnerability Spectre v1': r'Vulnerability Spectre v1:\s+(.+)',
        'Vulnerability Spectre v2': r'Vulnerability Spectre v2:\s+(.+)',
        'Vulnerability Srbds': r'Vulnerability Srbds:\s+(.+)',
        'Vulnerability Tsx async abort': r'Vulnerability Tsx async abort:\s+(.+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, file)
        if match:
            cpu_info[key] = match.group(1)

    return cpu_info

def parse_csv_data(csv_data):
    csv_file = io.StringIO(csv_data)
    reader = csv.DictReader(csv_file)
    gpu_info = {}
    for row in reader:
        for key, value in row.items():
            gpu_info[key] = value
    return gpu_info

def parse_nvidia_info(logfile):
    pattern = re.compile(r'Executing: nvidia-smi --query-gpu=.*? --format=csv\n(.*?)(?=\nINFO -|\Z)', re.DOTALL)
    match = pattern.search(logfile)
    
    if not match:
        print("No nvidia-smi output found in the logfile.")
        return None
    
    csv_data_block = match.group(1).strip()
    
    lines = csv_data_block.split('\n')
    if len(lines) < 2:
        print("Incomplete GPU information.")
        return None
    
    csv_data = '\n'.join(lines)
    gpu_info = parse_csv_data(csv_data)
    return gpu_info

def kernel_timing_df(lines):
    data = "\n".join(lines)
    return pd.read_csv(io.StringIO(data), sep=r'\s{2,}', header=0, engine="python")

def convert_to_float(value):
        try:
            return float(value)
        except ValueError:
            return value
        
def construct_metric(name, type, value, unit):
    return {
        "name": name,
        "type": type,
        "value": value,
        "unit": unit if unit else None,
    }

def process_kernel_data(info_dic, log_data):
    kernel_lines =[]
    capture_kernel_timing = False

    for line in log_data.splitlines():
        if not capture_kernel_timing and "Event: 9750" in line:
            capture_kernel_timing = True
            continue
        
        if capture_kernel_timing:
            if "exiting early" in line:
                break
            kernel_lines.append(line.strip())
    
    if kernel_lines:
        return parse_kernel_data(kernel_lines)
    else:
        print(f"No kernel information found for {info_dic['image_type']} with tag {info_dic['image_tag']}")
        return None

def parse_kernel_data(kernel_lines):
    metrics = []
    kernel_timing_lines = []
            
    for line in kernel_lines:
        if ':' in line:
            if 'GPU memory used(MB)' in line:
                special_case_match = re.match(r'(.+?)\((\w+)\):\s*([0-9.]+)', line.strip())
                if special_case_match:
                    name, unit, value = special_case_match.groups()
                    metric = construct_metric(name, int, int(value), unit)
                    metrics.append(metric)
            else:
                name, value = map(str.strip, line.split(':', 1))
                match = re.match(r"([0-9.]+)\s*(\w*)", value)
                if match:
                    value_num, unit = match.groups()
                    metric = construct_metric(name, float, float(value_num), unit)
                    metrics.append(metric)
        elif line.split()[0] in ["kernel", "sim_clean", "sim_A", "sim_ct", "sim_cp"]:
            elements = line.strip()
            kernel_timing_lines.append(elements)
        elif 'Time for Chain' in line:
            chain_time_match = re.match(r'Time for Chain (\d+) is ([0-9.]+) (\w+)', line.strip())
            if chain_time_match:
                chain_id, value_num, unit = chain_time_match.groups()
                metric = construct_metric(f"Time for Chain {chain_id}", float, float(value_num), unit)
                metrics.append(metric)
        elif 'launch count' in line:
            match = re.search(r'launch count\s+(\d+)\s*(\+\d+)?', line)
            if match:
                launch_count = int(match.group(1))
                metric = construct_metric(f"launch count", str, str(launch_count), None)
                metrics.append(metric)
        
    if kernel_timing_lines:
        name = "kernel timing"
        metric = construct_metric(name, pd.DataFrame, kernel_timing_df(kernel_timing_lines[1:]), None) 
        metrics.append(metric)

    return metrics

def parse_run_log(logfile):
    info = parse_logfile_name(logfile)
    with open(logfile, 'r') as f:
        logfile_content = f.read()
        info_data, kernel_data = logfile_content.split("- Setup")
        info['Datapath']=parse_datapath(info_data)
        info['System']=parse_system_info(info, info_data)
        info['CPU']=parse_cpu_info(info_data)
        info['GPU']=parse_nvidia_info(info_data)
        info['Run command']=parse_run_cmd(kernel_data)
        metrics = process_kernel_data(info, kernel_data)
    return {
        "info": info, 
        "metrics": metrics
    }

def json_serial(obj):
    """JSON serializer for objects not serializable by default"""

    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_json()
    elif isinstance(obj, type):
        return str(obj)
    
    raise TypeError ("Type %s not serializable" % type(obj))
