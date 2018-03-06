#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
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

int getUsedSquares(){
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
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string(argc, argv, "-o", const_cast<char *>("data"));

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    int sizesteps = getSizesteps();
    interval = getIntervall();

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

        int usedSquares = squareCounter;
        for(int i = 0; i < usedSquares; i++) {
            clearSquare(previousSquares[i]);
        }
        resetSquareCounter();
        //Barrier
        for(int i = 0; i < n; i++ ){
            putInSquare(&particles[i]);
        }
        //Barrier
        for(int i = 0; i < n; i++){
            applyForces(&particles[i], squares);
        }

        //Barrier
        //
        //  move particles
        //
        for( int i = 0; i < n; i++ )
            move( particles[i] );

        //Barrier
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
    }
    simulation_time = read_timer( ) - simulation_time;

    printf( "\nn = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    if( fsave )
        fclose( fsave );

    return 0;
}
