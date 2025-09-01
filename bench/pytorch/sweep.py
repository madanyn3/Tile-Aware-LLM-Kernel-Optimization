#!/usr/bin/env python3.8

import torch, os, csv
import common
from bench_prefill import runPrefill
from bench_decode import runDecode
from typing import List
from datetime import datetime

def sweep(x_D, x_H, x_QL, x_T):
    configs = []
    for D in x_D:
        for H in x_H:
            for QL in x_QL:
                configs.append(("prefill", dict(x_batch=1,x_head=H,x_qLen=QL,x_d=D)))
        for T in x_T:
            configs.append(("decode", dict(x_batch=1,x_head=H,x_kLen=T,x_d=D)))
    rows = []
    for kind,kwargs in configs:
        if kind == "prefill":
            r = runPrefill(**kwargs, x_dtype=torch.float16)
        else:
            r = runDecode(**kwargs, x_dtype=torch.float16)
        rows.append((kind, r))
    return rows

def createCSV (x_rows: List[dict]):
    l_prefill = []
    l_decode = []

    for item in x_rows:
        if item[0] == 'prefill':
            l_dict = item[1]
            row = f"{l_dict['B']}, {l_dict['H']}, {l_dict['Q_len']}, {l_dict['D']}, {l_dict['dtype']}, {l_dict['avg_ms_total']}, {l_dict['ms_per_token']}, {l_dict['tokens_per_s']}, {l_dict['mem_peak_MB_avg']}"
            l_prefill.append(row)
        elif item[0] == 'decode':
            l_dict = item[1]
            row = f"{l_dict['B']}, {l_dict['H']}, {l_dict['T']}, {l_dict['D']}, {l_dict['dtype']}, {l_dict['avg_ms_per_block']}, {l_dict['x_blockSteps']}, {l_dict['tokens_per_s']}, {l_dict['mem_peak_MB_avg']}"
            l_decode.append(row)

    l_outFile = f'{os.getcwd()}/sweep_output.csv'
    with open (l_outFile, 'w') as f:
        writer = csv.writer(f)
        header = f'sweep of prefill and decode conducted on {datetime.now()}; device = {common.getDevice()}'
        writer.writerow([header])
        writer.writerow(', , , , , prefill, , , , , '.split(','))
        writer.writerow(['B', 'H', 'Q_Len', 'D', 'dtype', 'avg_ms_total', 'ms_per_token', 'token_per_ms', 'mem_peak_MB_avg'])

        for row in l_prefill:
            writer.writerow(row.split(','))

        writer.writerow(['   '])
        writer.writerow(', , , , , decode, , , , , '.split(','))
        writer.writerow(['B', 'H', 'T', 'D', 'dtype', 'avg_ms_per_block', 'x_blockSteps', 'token_per_ms', 'mem_peak_MB_avg'])

        for row in l_decode:
            writer.writerow(row.split(','))
    print (f'output file created at: {l_outFile}')

if __name__ == "__main__":
    D = [64, 80, 96, 128]
    H = [8, 16, 32]
    QL = [512, 1024, 2048]
    T = [1024, 2048, 4096, 8192]

    # D = [16]
    # H = [4]
    # QL = [32]
    # T = [64]

    rows = sweep(D, H, QL, T)
    createCSV(rows)
