CC=gcc
MPICC=mpicc
CXX=g++
MPICXX=mpicxx
CFLAGS=-O3 -Wall -std=gnu99 -openmp -lrt
CXXFLAGS=-O3 -Wall
LDFLAGS=-lm -lrt 

all: compare gen_RMAT graphs_reference gen_RMAT_mpi 

gen_RMAT_serial: gen_RMAT.o
	$(CXX) $^ -o $@ $(LDFLAGS)

gen_RMAT_mpi: gen_RMAT_mpi.cpp
	$(MPICXX) $^ -o $@ -lrt

graphs_reference: main.o sssp_serial_reference.o
	$(CXX) $^ -o $@ $(LDFLAGS)

# write your own file
#sssp: main_mpi.o <file>.o
#	$(CXX) $^ -o $@ $(LDFLAGS)

compare: compare.o
	$(CXX) $^ -o $@ 

.cpp.o:
	$(CXX) $(CXXFLAGS) -o $@ -c $<

.c.o:
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -rf *.o gen_RMAT gen_RMAT_mpi graphs_reference compare

