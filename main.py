from utils import Utils, GraphParallel, GraphSequential
from charm4py import charm, Chare, Future, Reducer, Array, coro, Channel, Group
from time import time
import numpy as np
import argparse
from math import floor

INPUT_FILE = 'rmat-2.txt'
OUTPUT_FILE = 'sssp-out.txt'
ROOT = 0
DEBUG = False
INFINITY = float("inf")
INT_MAX = 1<<31

# run file by " python3 -m charmrun.start +pN main.py [args]"

def combine_gathered_data(data):
    result = []
    for d in data:
        for j in d:
            result.append(j)
    return list(set(result))


class SSSP(Chare):
    def initialize(self):
        self.index, = self.thisIndex # 1D
        self.utils = Utils(charm.numPes())
        self.graph_global = GraphSequential(INPUT_FILE)
        self.local_vertices = [self.utils.get_local_vertex(i) for i in range(self.graph_global.n)
                               if self.utils.get_vertex_owner(i) == self.index]
        self.global_vertices = [self.utils.get_global_vertex(i, self.index) for i in self.local_vertices] # global indices
        self.delta = 1
        self.distances = {v : INFINITY for v in range(self.graph_global.n)}
        self.graph_local = self.get_local_topology()
        self.B = {}
        self.channels = self.init_channels()
        self.changed_data = {} #add initialization each iteration

    def init_changed_data(self):
        neighbours = list(self.channels.keys())
        for i in neighbours:
            self.changed_data[i] = []

    def init_channels(self):
        neighbours = list(set([self.utils.get_vertex_owner(v) for v in self.graph_local.end_v]))
        if self.index in neighbours:
            neighbours.remove(self.index)
        channels = {i : Channel(self, remote=self.thisProxy[(i,)]) for i in neighbours}
        return channels

    def print_graph_info(self):
        print(f'===============\nLocal topology. Index: {self.index}\n==========')
        self.graph_local.print_graph()
        print(f'===============\nOther info:\n---------------')
        print(f'Local vertices indices: {self.local_vertices}\n'
              f'Global vertices indices: {self.global_vertices}')
        print(f'Distances: {self.distances}')
        print(f'Delta: {self.delta}\n---------------')

    def get_local_topology(self):
        rows_i, end_v, weights = [0], [], []
        for i in range(self.graph_global.n):
            if i in self.global_vertices:
                rows_i.append(int(self.graph_global.rows_indices[i+1] - self.graph_global.rows_indices[i] + rows_i[-1]))
                end_v.extend(self.graph_global.end_v[self.graph_global.rows_indices[i]:self.graph_global.rows_indices[i+1]])
                weights.extend(self.graph_global.weights[self.graph_global.rows_indices[i]:self.graph_global.rows_indices[i+1]])
        n = len(rows_i) - 1
        m = rows_i[-1]
        return GraphParallel(n, m, rows_i, end_v, weights)

    def calculate_delta(self):
        delta = 1 / self.allreduce(self.get_max_vertex_degree(), Reducer.max).get()
        return delta

    def get_max_vertex_degree(self):
        return max([self.graph_local.rows_indices[i + 1] - self.graph_local.rows_indices[i]
                    for i in range(0, self.graph_local.local_n)])

    def send_changed_data(self):
        neighbours = self.channels.keys()
        channels = list(self.channels.values())
        for c in neighbours:
            message = self.changed_data[c]
            self.channels[c].send(message)
        for ch in charm.iwait(channels):
            data = ch.recv()
            if DEBUG:
                print(f'Received data: {data} of length {len(data)}')
            if len(data) > 0:
                for d in data:
                    v = d[0]
                    dist = d[1]
                    if dist < self.distances[v]:
                        bucket_idx = floor(dist / self.delta)
                        if self.distances[v] != INFINITY:
                            old_bucket_index = floor(self.distances[v] / self.delta)
                            if v in self.B[old_bucket_index]:
                                if bucket_idx != old_bucket_index:
                                    self.B[old_bucket_index].remove(v)
                        if bucket_idx not in self.B:
                            self.B[bucket_idx] = [v]
                        else:
                            self.B[bucket_idx].append(v)
                        self.distances[v] = dist

    def relax(self, v, x):
        if DEBUG:
            print(f'~~~~~~~~~~\nStarting relaxation on process {self.index}. Vertex={v}, distance={x}.')
        new_bucket_index = floor(x / self.delta)
        if self.distances[v] != INFINITY:
            # v is in some bucket already
            old_bucket_index = floor(self.distances[v] / self.delta)
            if DEBUG:
                print(f'Bucket indices on process {self.index}: old={old_bucket_index}, new={new_bucket_index}')
            if v in self.B[old_bucket_index]:
                if new_bucket_index != old_bucket_index:
                    self.B[old_bucket_index].remove(v)
            if new_bucket_index not in self.B:
                self.B[new_bucket_index] = [v]
            else:
                self.B[new_bucket_index].append(v)
        else:
            if DEBUG:
                print(f'Bucket indices on process {self.index}: old="inf", new={new_bucket_index}')
            if new_bucket_index not in self.B:
                self.B[new_bucket_index] = [v]
            else:
                if v not in self.B[new_bucket_index]:
                    self.B[new_bucket_index].append(v)
        self.distances[v] = x
        if DEBUG:
            print(f'End of relaxation on process {self.index}. Buckets: {self.B}\n~~~~~~~~~~')

    def process_bucket(self, i):
        A = np.asarray(self.B[i], dtype=np.uint32)
        while A.shape[0] > 0:
            C = []
            for u in A:
                local_u = self.utils.get_local_vertex(u)
                edges_bound = (self.graph_local.rows_indices[local_u], self.graph_local.rows_indices[local_u + 1])
                adjacent_vertices = self.graph_local.end_v[edges_bound[0]:edges_bound[1]]
                adjacent_weights = self.graph_local.weights[edges_bound[0]:edges_bound[1]]
                for k, v in enumerate(adjacent_vertices):
                    weight = adjacent_weights[k]
                    distance = self.distances[u] + weight
                    if distance < self.distances[v]:
                        proc_num = self.utils.get_vertex_owner(v)
                        if proc_num != self.index:
                            # should add v to changed_data
                            self.changed_data[proc_num].append([v, distance])
                            self.distances[v] = distance
                        else:
                            C.append(v)
                            self.relax(v, distance)
            A = np.asarray([c for c in C if c in A], dtype=np.uint32)

    @coro
    def run(self, calc_done):
        self.initialize()
        self.delta = self.calculate_delta()
        if DEBUG:
            self.print_graph_info()
        if ROOT in self.global_vertices:
            self.distances[ROOT] = 0
            self.B[0] = [ROOT]
        else:
            self.distances[ROOT] = 0
            self.B[0] = []
        i = 0
        start_time = time()
        while i < INT_MAX:
            self.init_changed_data()
            if i not in self.B:
                self.B[i] = []
            if DEBUG:
                print(f'Process bucket number {i} on process {self.index}. Buckets: {self.B}.')
            if DEBUG:
                print(f'--------\nBefore: Proc: {self.index}, iteration: {i}, \nbuckets: {self.B}, \ndistances: {self.distances}')
            self.process_bucket(i)
            if DEBUG:
                print(f'Proc: {self.index}, Changed data: {self.changed_data}')
            self.send_changed_data()

            if DEBUG:
                print('Update index for next iteration...')
            bucket_indices = [b for b in self.B.keys() if b > i and b != 0 and len(self.B[b]) != 0]
            min_b = INT_MAX if len(bucket_indices) == 0 else min(bucket_indices)
            old_i = i
            i = self.allreduce(min_b, Reducer.min).get()
            if DEBUG:
                print(f'--------\nAfter: Proc: {self.index}, iteration: {old_i}, \nbucket: {self.B}, \ndistances: {self.distances}')


            if DEBUG:
                print(f'==========\nDone processing bucket {old_i} on process {self.index}.\n'
                      f'Index: {self.index}\nDistances: {self.distances}\nBuckets: {self.B}\n'
                      f'Next index: {i}, all indices: {bucket_indices}\n==========')
        if self.index == 0:
            self.print_binary()
        calc_done.send([start_time, self.collect_result()])

    def collect_result(self):
        dist = np.array(list(self.distances.values()))
        if DEBUG:
            print(f'Local distances on process {self.index}: {dist}')
        finished_reduce = Future()
        self.reduce(finished_reduce, dist, Reducer.min)
        return finished_reduce.get()

    def print_result(self, dist):
        print(f'End of algorithm. Distances: {list(dist)}')

    def print_binary(self):
        file = open(OUTPUT_FILE, 'wb')
        dist = np.array(list(self.distances.values()), dtype=np.float64)
        for d in range(len(dist)):
            if np.isinf(dist[d]):
                dist[d] = -1
        dist.tofile(file)


def parse_arguments():
    parser = argparse.ArgumentParser(description='charm4py SSSP delta-stepping')
    parser.add_argument('-i', type=str, help='Input file')
    parser.add_argument('-o', type=str, help='Output file, default: sssp-out.txt', default='sssp-out.txt')
    parser.add_argument('-r', type=int, help='Root vertex, default: 0', default=0)
    parser.add_argument('-d', action='store_true', help='Debug output')
    return parser.parse_args()


def main(args):
    global INPUT_FILE, OUTPUT_FILE, ROOT
    args = parse_arguments()
    INPUT_FILE = args.i
    OUTPUT_FILE = args.o
    ROOT = args.r
    DEBUG = args.d

    charm.thisProxy.updateGlobals({'INPUT_FILE': INPUT_FILE,
                                   'OUTPUT_FILE': OUTPUT_FILE,
                                   'ROOT': ROOT,
                                   'DEBUG' : DEBUG},
                                    awaitable=True).get()
    calc_done = Future()
    array_proxy = Array(SSSP, charm.numPes())
    charm.awaitCreation(array_proxy)
    print('Starting computation')
    array_proxy.run(calc_done)
    start_time, distances = calc_done.get()
    end_time = time() # should be after calc_done.get()
    elapsed_time = end_time - start_time
    print(f'End of algorithm. Distances: {list(distances)}')
    print(f'Exited successfully. Elapsed time: {elapsed_time:.7f}')
    file = open('time_old.csv', 'a')
    file.write(f'{INPUT_FILE}\t{elapsed_time:.7f}\t{charm.numPes()}\n')
    file.close()
    exit()


# if __name__ == '__main__':
charm.start(main)
