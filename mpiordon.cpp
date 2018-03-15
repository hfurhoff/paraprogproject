#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"

void putInSquare(particle_t* particle, square_t **squares, square_t **previousSquares, double interval, double cutoff, int *squareCounter){
    int x;
    int y;
    x = static_cast<int>(std::floor(particle->x / interval));
    y = static_cast<int>(std::floor(particle->y / interval));

    particle_node_t * ny;
    ny = (particle_node_t*) malloc(sizeof(particle_node_t));
    ny->p = particle;

    if(squares[x][y].particles == nullptr){
        ny->next = nullptr;
        squares[x][y].occupied = true;
        previousSquares[*squareCounter] = &squares[x][y];
        *squareCounter = *squareCounter + 1;
    }else {
        particle_node_t * rest;
        rest = squares[x][y].particles;
        ny->next = rest;
    }
    squares[x][y].particles = ny;
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

    printf("MPI RUN");

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );

    //
    //  set up the data partitioning across processors
    //
    int particle_per_proc = (n + n_proc - 1) / n_proc;
    int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ )
        partition_offsets[i] = min( i * particle_per_proc, n );

    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc; i++ )
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];

    //
    //  allocate storage for local partition
    //
    int nlocal = partition_sizes[rank];
    particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );

    MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );

    square_t **squares;
    square_t **previousSquares;
    double interval;
    double cutoff = 0.01;
    int squareCounter = 0;

    int sizesteps = getSizesteps();
    interval = getIntervall();
    int squaresToClear = 0;

    previousSquares = (square_t**) malloc(n * sizeof(square_t*));
    squares = (square_t**) malloc(sizesteps * sizeof(square_t*));
    for(int i = 0; i < sizesteps; i++){
        squares[i] = (square_t*) malloc(sizesteps * sizeof(square_t));
        for(int j = 0; j < sizesteps; j++){
            initSquare(&squares[i][j]);
        }
    }

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

        for(int i = 0; i < squaresToClear; i++) {
            clearSquare(previousSquares[i]);
        }

        for(int i = 0; i < n; i++ ){
            putInSquare(&particles[i], squares, previousSquares, interval, cutoff, &squareCounter);
        }

        squaresToClear = squareCounter;
        squareCounter = 0;

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
    for(int i = 0; i < squaresToClear; i++) {
        clearSquare(previousSquares[i]);
    }
    free(previousSquares);
    for(int i = 0; i < sizesteps; i++){
        free(squares[i]);
    }
    free(squares);
    free( partition_offsets );
    free( partition_sizes );
    free( local );
    free( particles );
    if( fsave )
        fclose( fsave );

    MPI_Finalize( );

    return 0;
}
