#ifndef _REENTRANT
#define _REENTRANT
#endif
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include "common.h"

//
//  global variables
//
int n;
unsigned int n_threads;
particle_t *particles;
FILE *fsave;
pthread_barrier_t barrier;

//
//  check that pthreads routine call was successful
//
#define P( condition ) {if( (condition) != 0 ) { printf( "\n FAILURE in %s, line %d\n", __FILE__, __LINE__ );exit( 1 );}}

square_t **squares;
square_t **previousSquares;
double interval;
double cutoff = 0.01;
int squareCounter = 0;
int squaresToClear = 0;
pthread_mutex_t countlock;
pthread_mutex_t **squarelock;

void initSquare(square_t *square){
    square->trueNeighbours = false;
    square->occupied = false;
    square->particles = nullptr;
}

void clearSquare(square_t *previousSquare){
    if(!previousSquare->occupied) return;
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

    //pthread_mutex_lock(&squarelock[x][y]);
    if(squares[x][y].particles == nullptr){
        ny->next = nullptr;
        squares[x][y].occupied = true;
        //pthread_mutex_lock(&countlock);
        previousSquares[squareCounter++] = &squares[x][y];
        //pthread_mutex_unlock(&countlock);
    }else {
        particle_node_t * rest;
        rest = squares[x][y].particles;
        ny->next = rest;
    }
    squares[x][y].particles = ny;
    //pthread_mutex_unlock(&squarelock[x][y]);

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
}
//
//  This is where the action happens
//
void *thread_routine( void *pthread_id )
{
    int thread_id = *(int*)pthread_id;

    int particles_per_thread = (n + n_threads - 1) / n_threads;
    int first = min(  thread_id    * particles_per_thread, n );
    int last  = min( (thread_id+1) * particles_per_thread, n );

    //
    //  simulate a number of time steps
    //
    for( int step = 0; step < NSTEPS; step++ )
    {

        if(thread_id == 0) {
            for (int i = 0; i < squaresToClear; i++) {
                clearSquare(previousSquares[i]);
            }
            for (int i = 0; i < n; i++) {
                putInSquare(&particles[i]);
            }
        }
        pthread_barrier_wait( &barrier );
        if(thread_id == 0){
            squaresToClear = squareCounter;
            squareCounter = 0;
        }
        for( int i = first; i < last; i++ )
        {
            applyForces(&particles[i], squares);
        }

        pthread_barrier_wait( &barrier );

        //
        //  move particles
        //
        for( int i = first; i < last; i++ )
            move( particles[i] );

        pthread_barrier_wait( &barrier );

        //
        //  save if necessary
        //
        if( thread_id == 0 && fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
    }

    return NULL;
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{
    //
    //  process command line
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-p <int> to set the number of threads\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    n = read_int( argc, argv, "-n", 1000 );
    n_threads = static_cast<unsigned int>(read_int(argc, argv, "-p", 2 ));
    char *savename = read_string( argc, argv, "-o", NULL );

    //
    //  allocate resources
    //
    fsave = savename ? fopen( savename, "w" ) : NULL;

    printf("PTHREADS RUN");
    particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    int sizesteps = getSizesteps();
    interval = getIntervall();

    previousSquares = (square_t**) malloc(n * sizeof(square_t*));
    squares = (square_t**) malloc(sizesteps * sizeof(square_t*));
    squarelock = (pthread_mutex_t**) malloc(sizesteps * sizeof(pthread_mutex_t*));
    for(int i = 0; i < sizesteps; i++){
        squares[i] = (square_t*) malloc(sizesteps * sizeof(square_t));
        squarelock[i] = (pthread_mutex_t*) malloc(sizesteps * sizeof(pthread_mutex_t));
        for(int j = 0; j < sizesteps; j++){
            initSquare(&squares[i][j]);
            P(pthread_mutex_init(&squarelock[i][j], NULL));
        }
    }

    pthread_attr_t attr;
    P( pthread_attr_init( &attr ) );
    P( pthread_barrier_init( &barrier, NULL, n_threads ) );
    P(pthread_mutex_init(&countlock, NULL));

    int *thread_ids = (int *) malloc( n_threads * sizeof( int ) );
    for( int i = 0; i < n_threads; i++ )
        thread_ids[i] = i;

    pthread_t *threads = (pthread_t *) malloc( n_threads * sizeof( pthread_t ) );

    printf("NUMBER OF THREADS = %d\n", n_threads);
    //
    //  do the parallel work
    //
    double simulation_time = read_timer( );
    for( int i = 1; i < n_threads; i++ )
    P( pthread_create( &threads[i], &attr, thread_routine, &thread_ids[i] ) );

    thread_routine( &thread_ids[0] );

    for( int i = 1; i < n_threads; i++ )
    P( pthread_join( threads[i], NULL ) );
    simulation_time = read_timer( ) - simulation_time;

    printf( "n = %d, n_threads = %d, simulation time = %g seconds\n", n, n_threads, simulation_time );

    //
    //  release resources
    //
    for(int i = 0; i < sizesteps; i++){
        free(squares[i]);
    }
    free(squares);
    free(previousSquares);
    P( pthread_barrier_destroy( &barrier ) );
    P( pthread_attr_destroy( &attr ) );
    free( thread_ids );
    free( threads );
    free( particles );
    if( fsave )
        fclose( fsave );

    return 0;
}
