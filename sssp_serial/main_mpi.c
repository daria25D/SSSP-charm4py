#include <stdio.h>
#include <stdlib.h>
#include <error.h>
#include <assert.h>
#include <string.h>
#include "mpi.h"
#include <math.h>
#include "defs.h"

int lgsize;
char* inFilename;
char* outFilename;
uint32_t rootNumberToValidate;

void usage(int argc, char **argv)
{
    printf("Usage:\n");
    printf("    %s -in <input> [options]\n", argv[0]);
    printf("Options:\n");
    printf("    -in <input> -- input graph filename\n");
    printf("    -out <output> -- output filename (distances from root vertex). By default output is '<input>.v.<process id>'\n");
    printf("    -root <root> -- root number for validation\n");
    exit(1);
}

void init (int argc, char** argv, graph_t* G)
{
    int i;
    inFilename = outFilename = NULL;
    rootNumberToValidate = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &G->nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &G->rank);
    

    for ( i = 1; i < argc; ++i) {
   		if (!strcmp(argv[i], "-in")) {
            inFilename = argv[++i];
        }
   		if (!strcmp(argv[i], "-out")) {
            outFilename = argv[++i];
        }
		if (!strcmp(argv[i], "-root")) {
			rootNumberToValidate = (int) atoi(argv[++i]);
        }
    }
    if (!inFilename) usage(argc, argv);
    if (!outFilename) {
        outFilename = (char *)malloc((strlen(inFilename) + 3) * sizeof(char));
        sprintf(outFilename, "%s.v", inFilename);
    }
    //sprintf(outFilename, "%s.%d", outFilename, G->rank);
    sprintf(inFilename, "%s.%d", inFilename, G->rank);
}

void readGraph(graph_t *G, char *filename)
{
    edge_id_t arity;
    uint8_t align;
    FILE *F = fopen(filename, "rb");
    if (!F) error(EXIT_FAILURE, 0, "Error in opening file %s", filename);

	assert(fread(&G->n, sizeof(vertex_id_t), 1, F) == 1);
    assert(fread(&arity, sizeof(edge_id_t), 1, F) == 1);
    //FIXME: check local_n, local_m 
    assert(fread(&G->local_n, sizeof(vertex_id_t), 1, F) == 1);
    assert(fread(&G->local_m, sizeof(edge_id_t), 1, F) == 1); 
    assert(fread(&G->nproc, sizeof(int), 1, F) == 1);
    /* Check for power of 2 */
    if ((G->nproc & (G->nproc - 1)) != 0) {
        if(G->rank == G->nproc -1) printf("Number of processes must be a power of two\n");
        MPI_Finalize();
        exit(1);
    }   
    lgsize = log(G->nproc)/log(2);
    assert (lgsize < G->nproc);
    assert(fread(&G->directed, sizeof(bool), 1, F) == 1);
    assert(fread(&align, sizeof(uint8_t), 1, F) == 1);

	G->rowsIndices = (edge_id_t *)malloc((G->local_n+1) * sizeof(edge_id_t));
    assert(G->rowsIndices);

	assert(fread(G->rowsIndices, sizeof(edge_id_t), G->local_n+1, F) == (G->local_n+1));
    G->endV = (vertex_id_t *)malloc(G->rowsIndices[G->local_n] * sizeof(vertex_id_t));
    assert(G->endV);

    assert(fread(G->endV, sizeof(vertex_id_t), G->rowsIndices[G->local_n], F) == G->rowsIndices[G->local_n]);

    assert(fread(&G->nRoots, sizeof(uint32_t), 1, F) == 1);
    
    G->roots = (vertex_id_t *)malloc(G->nRoots * sizeof(vertex_id_t));
    assert(G->roots);

    assert(fread(G->roots, sizeof(uint32_t), G->nRoots, F) == G->nRoots);
    //FIXME: next line
    assert(fread(G->roots, sizeof(uint32_t), G->nRoots, F) == G->nRoots);
    G->weights = (weight_t *)malloc(G->rowsIndices[G->local_n] * sizeof(weight_t));
    assert(G->weights);

    assert(fread(G->weights, sizeof(weight_t), G->rowsIndices[G->local_n], F) == G->rowsIndices[G->local_n]);

    fclose(F);
}

void calc_traversed_edges(graph_t *G, weight_t* dist, uint64_t* traversed_edges)
{
    edge_id_t i;
    edge_id_t nedges_local = 0;
    edge_id_t nedges_global = 0;
    for ( i = 0; i < G->local_n; i++) {
        if ( dist[i] != -1 ) {
            nedges_local +=  G->rowsIndices[i+1] - G->rowsIndices[i];
        }
    }
    MPI_Allreduce(&nedges_local, &nedges_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    *traversed_edges = nedges_global;
}

/* write distances from root vertex to each others to output file. -1 = infinity */
void writeDistance(char* filename, weight_t *dist, vertex_id_t n)
{
    FILE *F = fopen(filename, "wb");
    assert(fwrite(dist, sizeof(weight_t), n, F) == n);
    fclose(F);
}

void write_validate(graph_t *G, weight_t* local_dist)
{
    int i, j;
    weight_t* dist_recv;
    weight_t* dist;

    if (G->rank == 0) dist_recv = (weight_t *)malloc(G->n * sizeof(weight_t));
    
    MPI_Gather(local_dist, G->local_n, MPI_DOUBLE, dist_recv, G->local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (G->rank == 0) {
        dist = (weight_t *)malloc(G->n*sizeof(weight_t));
        for (i = 0; i < G->nproc; i++) {
            for (j = 0; j < G->local_n; j++) {
                dist[VERTEX_TO_GLOBAL(i, j)] = dist_recv[G->local_n * i + j];
            }
        }
        writeDistance(outFilename, dist, G->n);
        free(dist_recv);
        free(dist);

    }
}

void freeGraph(graph_t *G)
{
    free(G->rowsIndices);
    free(G->endV);
    free(G->roots);
    free(G->weights);
}

int main(int argc, char **argv) 
{
    weight_t* local_dist;
    graph_t g;
    uint64_t traversed_edges;
    uint64_t sssp_edges;
    uint32_t i;
    vertex_id_t j;
   
    /* initializing and reading the graph */
    MPI_Init (&argc, &argv);
    init(argc, argv, &g); 
    readGraph(&g, inFilename);

    local_dist = (weight_t *)malloc(g.local_n * sizeof(weight_t));

    /* doing SSSP */
    for ( i = 0; i < g.nRoots; ++i) {
        /* initializing for validation, -1 = infinity */
        for ( j = 0; j < g.local_n; j++) {
            local_dist[j] = -1;
        }
        
        //FIXME: timings
        //sssp(g.roots[i], &g, local_dist, &sssp_edges);

        calc_traversed_edges(&g, local_dist, &traversed_edges);
    
        if (rootNumberToValidate == i) {
            /* writing for validation */
            write_validate(&g, local_dist);
        } 
    }
    free(local_dist);
    freeGraph(&g);
    MPI_Finalize();
    return 0;
}
