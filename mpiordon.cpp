#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cmath>
#include "common.h"

square_t **squares;
square_t **previousSquares;
double interval;
double cutoff = 0.01;
int squareCounter = 0;

void initSquare(square_t *square){
    square->trueNeighbours = false;
    square->occupied = false;
    square->particles = nullptr;
}

int getsquaresToClear(){
    return squareCounter;
}

void resetSquareCounter(){
    squareCounter = 0;
}

void clearSquare(square_t *previousSquare){
    previousSquare->occupied = false;
    previousSquare->trueNeighbours = false;
    freeNodes(previousSquare->particles);
    previousSquare->particles = nullptr;
}

void freeNodes(particle_node_t* destroyNode){
    if(destroyNode->next == nullptr) free(destroyNode);
    else{
        freeNodes(destroyNode->next);
        free(destroyNode);
    }
}

void putInSquare(particle_t* particle){
    int x;
    int y;
    x = particle->sx = static_cast<int>(std::floor(particle->x / interval));
    y = particle->sy = static_cast<int>(std::floor(particle->y / interval));
    particle->inMiddle = true;

    particle_node_t * ny;
    ny = (particle_node_t*) malloc(sizeof(particle_node_t));
    ny->p = particle;
    //lock
    if(squares[x][y].particles == nullptr){
        ny->next = nullptr;
        squares[x][y].occupied = true;
        previousSquares[squareCounter++] = &squares[x][y];
    }else {
        particle_node_t * rest;
        rest = squares[x][y].particles;
        ny->next = rest;
    }
    squares[x][y].particles = ny;

    if(x * interval <= particle->x && particle->x <= x * interval + cutoff){
        squares[x][y].trueNeighbours = true;
        particle->inMiddle = false;
    } else if ((x + 1) * interval - cutoff <= particle->x && particle->x <= (x + 1) * interval) {
        squares[x][y].trueNeighbours = true;
        particle->inMiddle = false;
    } else if(y * interval <= particle->y && particle->y <= y * interval + cutoff){
        squares[x][y].trueNeighbours = true;
        particle->inMiddle = false;
    } else if ((y + 1) * interval - cutoff <= particle->y && particle->y <= (y + 1) * interval){
        squares[x][y].trueNeighbours = true;
        particle->inMiddle = false;
    }
    //unlock
}


//
//  benchmarking program
//
int main( int argc, char **argv )
{
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );



    set_size( n );
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    init_particles( n, particles );

    int sizesteps = getSizesteps();
    interval = getIntervall();
    int squaresToClear = 0;

    previousSquares = (square_t**) malloc(n * sizeof(square_t*));
    squares = (square_t**) malloc(sizesteps * sizeof(square_t*));
    for(int i = 0; i < sizesteps; i++){
        squares[i] = (square_t*) malloc(sizesteps * sizeof(square_t));
        if(rank == 0) {
            for (int j = 0; j < sizesteps; j++) {
                initSquare(&squares[i][j]);
            }
        }
    }



    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;

    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_contiguous(2, MPI_INTEGER, &PARTICLE);
    MPI_Type_contiguous(1, MPI_CXX_BOOL, &PARTICLE);
    MPI_Type_commit( &PARTICLE );

    MPI_Datatype PARTICLENODE;
    MPI_Type_contiguous(1, PARTICLE, &PARTICLENODE);
    MPI_Type_contiguous(1, PARTICLENODE, &PARTICLENODE);
    MPI_Type_commit(&PARTICLENODE);

    MPI_Datatype SQUARE;
    MPI_Type_contiguous(2, MPI_CXX_BOOL, &SQUARE);
    MPI_Type_contiguous(1, PARTICLENODE, &SQUARE);
    MPI_Type_commit(&SQUARE);

    //
    //  set up the data partitioning across processors
    //
    int particle_per_proc = (n + n_proc - 1) / n_proc;
    int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ ) {
        partition_offsets[i] = min( i * particle_per_proc, n );
    }

    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc; i++ ) {
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];
    }

    //
    //  allocate storage for local partition
    //
    int nlocal = partition_sizes[rank];
    particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  collect all global data locally (not good idea to do)
        //
        MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );

        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );


        if(rank == 0) {
            for (int i = 0; i < squaresToClear; i++) {
                clearSquare(previousSquares[i]);
            }
            //Barrier will be needed when parallel
            for (int i = 0; i < n; i++) {
                putInSquare(&particles[i]);
            }
        }

        MPI_Bcast(squares, sizesteps, SQUARE, 0, MPI_COMM_WORLD);

        //
        //  compute all forces
        //
        for( int i = 0; i < nlocal; i++ )
        {
            applyForces(&local[i], squares);
        }

        //
        //  move particles
        //
        for( int i = 0; i < nlocal; i++ )
            move( local[i] );

    }
    simulation_time = read_timer( ) - simulation_time;

    if( rank == 0 )
        printf( "n = %d, n_procs = %d, simulation time = %g s\n", n, n_proc, simulation_time );

    //
    //  release resources
    //
    free( partition_offsets );
    free( partition_sizes );
    free( local );
    free( particles );
    if( fsave )
        fclose( fsave );

    MPI_Finalize( );

    return 0;
}

