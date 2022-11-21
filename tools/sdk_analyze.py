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


def main():
    args = parse_args()
    with open(args.profile_file) as f:
        data = f.read()
    graph, events = data.split('----\n')
    graph = graph.strip().split('\n')
    events = events.strip().split('\n')

    name2id = {}
    id2name = {}
    for i, name in enumerate(graph):
        name2id[name] = i
        id2name[i] = name

    n_active = {i: 0 for i in range(len(name2id))}
    n_call = {i: 0 for i in range(len(name2id))}
    t_occupy = {i: 0 for i in range(len(name2id))}
    t_usage = {i: 0 for i in range(len(name2id))}
    t_time = {i: [] for i in range(len(name2id))}
    used_id = set()
    event_start = {}
    now = 0
    t_min = 0

    for event in events:
        words = event.split()
        name = words[0]
        id = name2id[name]
        used_id.add(id)
        kind, index, ts = map(int, words[1:])

        if now == 0:
            t_min = ts
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
        occupy = t_occupy[id] / (now - t_min)
        usage = t_usage[id] / (now - t_min)
        times = sorted(t_time[id])
        t_mean = np.mean(times) / 1000
        t_50 = times[int(len(times) * 0.5)] / 1000
        t_90 = times[int(len(times) * 0.9)] / 1000
        table.add_row(
            [id2name[id], occupy, usage, n_call[id], t_mean, t_50, t_90])
    print(table.draw())


if __name__ == '__main__':
    main()
