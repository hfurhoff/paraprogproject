#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "common.h"

double size;
double intervall;

int sizesteps;


#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005
//
//  time
//
double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

void initSquare(square_t *square){
    square->trueNeighbours = false;
    square->occupied = false;
    square->particles = nullptr;
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



//
//  keep density konstant
//
void set_size( int n )
{
    size = sqrt( density * n );
    sizesteps = static_cast<int>(std::floor(size / cutoff));
    intervall = size / sizesteps;

    printf("\nINTERVALL = %g\nSIZESTEPS = %d\nSIZE = %g\n", intervall, sizesteps, size);
}

int getSizesteps(){
    return sizesteps;
}

double getIntervall(){
    return intervall;
}

//
//  Initialize the particle positions and velocities
//
void init_particles( int n, particle_t *p )
{
    srand48( time( NULL ) );

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;

    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;

    for( int i = 0; i < n; i++ )
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];

        //
        //  distribute particles evenly to ensure proper spacing
        //
        p[i].x = size*(1.+(k%sx))/(1+sx);
        p[i].y = size*(1.+(k/sx))/(1+sy);

        //
        //  assign random velocities within a bound
        //
        p[i].vx = drand48()*2-1;
        p[i].vy = drand48()*2-1;
    }
    free( shuffle );
}

void applyForces(particle_t *particle, square_t (**squares)){
    int x;
    int y;
    x = static_cast<int>(std::floor(particle->x / intervall));
    y = static_cast<int>(std::floor(particle->y / intervall));
    particle->ax = particle-> ay = 0;
    particle_node_t *temp;

    int tempX;
    int tempY;
    int maxX,maxY;
    if(x > 0) tempX = x - 1; else tempX = x;
    if(y > 0) tempY = y - 1; else tempY = y;
    if(x < sizesteps - 1) maxX = x + 2; else maxX = sizesteps;
    if(y < sizesteps - 1) maxY = y + 2; else maxY = sizesteps;

    for (int i = tempX; i < maxX; i++) {
        for (int j = tempY; j < maxY; j++) {
            temp = squares[i][j].particles;
            while (temp != nullptr) {
                apply_force(*particle, *temp->p);
                temp = temp->next;
            }
        }
    }
    //}
}

//
//  interact two particles
//
void apply_force( particle_t &particle, particle_t &neighbor )
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff ) //0.0001
        return;
    r2 = fmax( r2, min_r*min_r );
    double r = sqrt( r2 );

    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

//
//  integrate the ODE
//
void move( particle_t &p )
{
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;

    //
    //  bounce from walls
    //
    while( p.x < 0 || p.x > size )
    {
        p.x  = p.x < 0 ? -p.x : 2*size-p.x;
        p.vx = -p.vx;
    }
    while( p.y < 0 || p.y > size )
    {
        p.y  = p.y < 0 ? -p.y : 2*size-p.y;
        p.vy = -p.vy;
    }
}

//
//  I/O routines
//
void save( FILE *f, int n, particle_t *p )
{
    static bool first = true;
    if( first )
    {
        fprintf( f, "%d %g\n", n, size );
        first = false;
    }
    for( int i = 0; i < n; i++ )
        fprintf( f, "%g %g\n", p[i].x, p[i].y );
}

//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}
