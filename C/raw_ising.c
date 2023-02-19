#include <stdlib.h>
#include <stdio.h>
#define C_ONLY
#include "numeth/gigarand/gigarand.c"
#include <math.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/time.h>

#define TIME_ESTIMATE_BLOCK 5000

long int block_time_ms = 0;
int time_estimate_count;
int time_blocks = 0;
struct timeval first_call, last_call;
int initialized = 0;
int print_config = 0;

void estimate_time(){
    struct timeval now;
    gettimeofday(&now, NULL);
    if (initialized == 0){
        block_time_ms = (long int) (1e3*(now.tv_sec - first_call.tv_sec) + 1e-3*(now.tv_usec - first_call.tv_usec));
        time_estimate_count++;
        initialized = 1;
    }else{
        block_time_ms = block_time_ms*(time_estimate_count);
        block_time_ms += (long int) (1e3*(now.tv_sec - last_call.tv_sec) + 1e-3*(now.tv_usec - last_call.tv_usec));
        time_estimate_count++;
        block_time_ms /= time_estimate_count;
    }
    last_call = now;
    // printf("block time: %ld\n", block_time_ms);
    // printf("remaining blocks: %d\n", time_blocks - time_estimate_count);
    printf("remaining: %.2lf seconds\n", (time_blocks - time_estimate_count)*block_time_ms*1e-3);

    return;
}
    

int mod(int k, int M){
    if (k > 0){
        if (k < M){return k;}
        else {return k%M;};
    };
    return M+k;
};

void ising(int l, float beta, float h, int n_steps){

    gettimeofday(&first_call, NULL);
    // declarations
    short int S[l][l];    
    short int proposal, neighborhood, log_r, accepted=0;
    float u;

    // check for overflow
    printf("Size of S is %e\n", (double)sizeof(S));

    // initialization of matrix
    for (int i=0; i<l; i++){
        for (int j=0; j<l; j++){
            S[i][j] = 2*(rand()%2) - 1;
        };
    };

    // erases the storage file
    char conf_filename_buffer[50];
    char stat_filename_buffer[50];

    snprintf(conf_filename_buffer, 50, "data/conf_h_%.4lf_beta_%.4lf_L_%d_T_%d.data", h, beta, l, n_steps );
    snprintf(stat_filename_buffer, 50, "data/stat_h_%.4lf_beta_%.4lf_L_%d_T_%d.data", h, beta, l, n_steps );
    
    FILE * configs = fopen(conf_filename_buffer, "w");
    FILE * stats = fopen(stat_filename_buffer, "w");
    fprintf(stats, "#M\tH\n");

    printf("Init. Done\n");

    // true cycle of metropolis hastings
    for (int iter=0; iter < n_steps; iter++){

        for (int i=0; i<l; i++){
            for (int j=0; j<l; j++){
                    proposal = 2*(rand()%2) - 1;
                    neighborhood = (
                                    S[i][ mod(j+1, l)] +
                                    S[mod(i+1, l)][j] +
                                    S[i][ mod(j-1, l)] +
                                    S[mod(i-1, l)][j]
                                    );
                    log_r = beta*(neighborhood + h)*(proposal - S[i][j]);
                    u = ran2();
                    if (log_r > log(u)){
                        accepted++ ;
                        S[i][j] = proposal;
                    }
                    else{
                        S[i][j] = S[i][j];
                    }

                    // TEST for save/read
                    // black frame -> white frame
                    // S[i][j]= T_BLOCK*blck_count + blck_iter + i + j;
                    if (print_config == 1){fprintf(configs, "%d ", S[i][j]);}
            };
            if (print_config == 1){fprintf(configs, "\n");}
        };
        // prints stats
        float H=0, M=0;
        float H_neigh;

        for (int i =0; i < l; i++){
            for (int j =0; j < l; j++){
                neighborhood = (S[i][mod(j+1, l)] + S[mod(i+1, l)][j] + S[i][mod(j-1, l)] + S[mod(i-1, l)][j]);
                H_neigh = -((1.0/4)*(neighborhood)+h)*S[i][j];
                H += H_neigh;
                M += S[i][j];
            };
        };
        fprintf(stats, "%lf %lf\n", M, H);

        if ( (iter != 0) &&(iter % TIME_ESTIMATE_BLOCK == 0)){
            estimate_time();
        }
    };


    
    // fclose(configs);
    fclose(stats);

    printf("Total time: %lf seconds\n", (double)(last_call.tv_sec - first_call.tv_sec));
    printf("--------------- END ------------\n");
};

int main (int argc, char **argv)
{
    int n_steps = 0, l = 0;
    float h = 0.0, beta = 0.0;
    int c;

    opterr = 0;


    while ((c = getopt (argc, argv, "l:b:h:t:c")) != -1)
    switch (c)
        {
        case 'l':
            l = atoi(optarg);
            break;
        case 'b':
            beta = atof(optarg);
            break;
        case 'h':
            h = atof(optarg);
            break;
        case 't':
            n_steps = atoi(optarg);
            break;
        case 'c':
            printf("Print configuration activated\n");
            print_config = 1;
            break;
        case '?':
            if (optopt == 'c'){
                printf("Print configuration activated");
                print_config = 1;
            }else if (isprint (optopt)){
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            }else{
                fprintf (stderr,
                        "Unknown option character `\\x%x'.\n",
                        optopt);
                return 1;
            }
            break;
        default:
        printf("Gimme an input nigga\n");
        abort ();
        }
    printf("Starting (L=%d, n_steps=%d, beta=%lf, h=%lf)\n", l, n_steps, beta, h);
    time_blocks = ((int) n_steps)/((int)TIME_ESTIMATE_BLOCK);
    ising(l, beta, h, n_steps);
}
