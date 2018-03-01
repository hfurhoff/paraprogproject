#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"

//
//  benchmarking program
//
int main( int argc, char **argv )
{
    printf("in main");
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

    printf("before initSquares");
    initSquares();
    printf("initSquares works");
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        int x;
        x = static_cast<int>(std::floor(particles[0].x / (sqrt(0.0005 * n ) / 100)));
        int y;
        y = static_cast<int>(std::floor(particles[0].y / (sqrt(0.0005 * n ) / 100)));

        //printf("\nparticle zero in square {%d, %d}", x, y);
        //printf("\nparticle zero has speed{%g, %g}", particles[0].vx, particles[0].vy);
        printf("\nparticle zero has acc {%g, %g}", particles[0].ax, particles[0].ax);

        if(step > 0) clearEnvironment();
        //
        //  compute forcees
        //
        for(int i = 0; i < n; i++ ){
            putInSquare(&particles[i]);
        }
        //Barrier
        for(int i = 0; i < n; i++){

            applyForces(&particles[i]);
        }
        //Barrier
        //
        //  move particles
        //
        for( int i = 0; i < n; i++ )
            move( particles[i] );

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
