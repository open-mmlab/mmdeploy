# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
from texttable import Texttable


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze sdk profiler file tool.')
    parser.add_argument('profile_file', help='SDK profile file path')
    args = parser.parse_args()
    return args


def get_name(addr, prev, addr2name, used_addr, depth, skip):
    node_name = addr2name[addr] if not skip else ''
    if addr not in prev:
        return ' ' * depth * 4 + node_name
    prev_addr = prev[addr]
    if prev_addr in used_addr:
        depth += 1
        skip = True
    prev_name = get_name(prev[addr], prev, addr2name, used_addr, depth, skip)
    if len(prev_name.split()) == 0:
        return prev_name + node_name
    return prev_name + '/' + node_name


def main():
    args = parse_args()
    with open(args.profile_file) as f:
        data = f.read()
    graph, events = data.split('----\n')
    graph = graph.strip().split('\n')
    events = events.strip().split('\n')

    addr2name = {}
    addr2id = {}
    id2addr = {}
    next = {}
    prev = {}
    for i, line in enumerate(graph):
        info = line.split()
        name, addr = info[:2]
        addr2name[addr] = name
        addr2id[addr] = i
        id2addr[i] = addr
        next[addr] = []
        for child in info[2:]:
            next[addr].append(child)
            prev[child] = addr

    n_active = {i: 0 for i in range(len(addr2id))}
    n_call = {i: 0 for i in range(len(addr2id))}
    t_occupy = {i: 0 for i in range(len(addr2id))}
    t_usage = {i: 0 for i in range(len(addr2id))}
    t_time = {i: [] for i in range(len(addr2id))}
    used_id = set()
    used_addr = set()
    event_start = {}
    now = 0
    first_id = None

    for event in events:
        words = event.split()
        addr = words[0]
        id = addr2id[addr]
        used_addr.add(addr)
        used_id.add(id)
        kind, index, ts = map(int, words[1:])

        if first_id is None:
            first_id = id

        if id == first_id and kind == 0 and n_active[id] == 0:
            now = ts

        key = (id, index)
        delta = ts - now
        now = ts

        for i, n_act in n_active.items():
            if n_act > 0:
                t_occupy[i] += delta
                t_usage[i] += delta * n_act

        if kind == 0:
            event_start[key] = ts
            n_active[id] += 1
            n_call[id] += 1
        else:
            dt = ts - event_start[key]
            t_time[id].append(dt)
            event_start.pop(key)
            n_active[id] -= 1

    table = Texttable(max_width=0)
    table.header(
        ['name', 'occupy', 'usage', 'n_call', 't_mean', 't_50%', 't_90%'])

    for id in sorted(list(used_id)):
        occupy = t_occupy[id] / (t_occupy[first_id])
        usage = t_usage[id] / (t_occupy[first_id])
        times = sorted(t_time[id])
        t_mean = np.mean(times) / 1000
        t_50 = times[int(len(times) * 0.5)] / 1000
        t_90 = times[int(len(times) * 0.9)] / 1000
        name = get_name(id2addr[id], prev, addr2name, used_addr, 0, False)
        if len(next[id2addr[id]]) != 0:
            occupy = '-'
            usage = '-'
        table.add_row([name, occupy, usage, n_call[id], t_mean, t_50, t_90])
    print(table.draw())


if __name__ == '__main__':
    main()
