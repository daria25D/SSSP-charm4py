from numpy import fromfile, uint32, uint64, float64, uint8, asarray


class Utils:
    def __init__(self, n_procs):
        self.lg_size = 0
        while 1 << self.lg_size != n_procs:
            self.lg_size += 1
        self.owner_const = (1 << self.lg_size) - 1

    def get_vertex_owner(self, vertex_global):
        return vertex_global & self.owner_const

    def get_global_vertex(self, vertex, proc):
        return (vertex << self.lg_size) + proc

    def get_local_vertex(self, vertex):
        return vertex >> self.lg_size


class GraphParallel:
    def __init__(self, local_n, local_m, r_i, end_v, weights, directed=False):
        self.local_n = local_n
        self.local_m = local_m
        self.rows_indices = asarray(r_i, dtype=uint64)
        self.end_v = asarray(end_v, dtype=uint32)
        self.weights = asarray(weights, dtype=float64)
        self.directed = directed
        self.num_of_edges = {i : self.rows_indices[i+1] - self.rows_indices[i] for i in range(self.local_n)}

    def read_graph(self, filename, file_format='binary', offset=0):
        if file_format == 'binary':
            file = open(filename, 'rb')
            self.local_n, = fromfile(file, dtype=uint32, count=1, offset=offset)  # uint32_t
            arity, = fromfile(file, dtype=uint64, count=1)  # uint64_t - error in C++ code, use rows_indices[-1] for local_m`

            self.directed = bool(fromfile(file, dtype=uint8, count=1)[0])
            align, = fromfile(file, dtype=uint8, count=1)

            self.rows_indices = fromfile(file, dtype=uint64, count=self.local_n + 1)
            self.end_v = fromfile(file, dtype=uint32, count=self.rows_indices[-1])
            self.local_m = self.rows_indices[-1]

            self.n_roots, = fromfile(file, dtype=uint32, count=1)
            self.roots = fromfile(file, dtype=uint32, count=self.n_roots)
            self.n_traversed_edges = fromfile(file, dtype=uint64, count=self.n_roots)

            self.weights = fromfile(file, dtype=float64, count=self.rows_indices[-1])
            self.offset = int(14 + self.rows_indices.shape[0] * 8 + \
                              self.rows_indices[-1] * 4 + \
                              4 + self.n_roots * 12 + \
                              self.rows_indices[-1] * 8)
            file.close()
        else:  # TODO write code for non-binary format
            pass

    def print_graph(self):
        print(f'local_n: {self.local_n}, local_m: {self.local_m}')
        print(f'directed: {self.directed}')
        print(f'Rows indices: {self.rows_indices}')
        print(f'EndV: {self.end_v}')
        print(f'Weights: {self.weights}')
        print(f'Number of edges for each vertex: {self.num_of_edges}')


class GraphSequential:
    def __init__(self, filename):
        self.read_graph(filename)

    def read_graph(self, filename):
        file = open(filename, 'rb')
        self.n, = fromfile(file, dtype=uint32, count=1)  # uint32_t
        arity, = fromfile(file, dtype=uint64,
                          count=1)  # uint64_t - error in C++ code, use rows_indices[-1] for local_m`

        self.directed = bool(fromfile(file, dtype=uint8, count=1)[0])
        align, = fromfile(file, dtype=uint8, count=1)

        self.rows_indices = fromfile(file, dtype=uint64, count=self.n + 1)
        self.end_v = fromfile(file, dtype=uint32, count=self.rows_indices[-1])
        self.m = self.rows_indices[-1]
        self.num_of_edges = {i : self.rows_indices[i+1] - self.rows_indices[i] for i in range(self.n)}

        self.n_roots, = fromfile(file, dtype=uint32, count=1)
        self.roots = fromfile(file, dtype=uint32, count=self.n_roots)
        self.n_traversed_edges = fromfile(file, dtype=uint64, count=self.n_roots)

        self.weights = fromfile(file, dtype=float64, count=self.rows_indices[-1])
        file.close()

    def print_graph(self):
        print(f'n: {self.n}, m: {self.m}')
        print(f'directed: {self.directed}')
        print(f'Rows indices: {self.rows_indices}')
        print(f'EndV: {self.end_v}')
        print(f'Weights: {self.weights}')
        print(f'Number of edges for each vertex: {self.num_of_edges}')