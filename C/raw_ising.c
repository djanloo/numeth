#include <stdlib.h>
#include <stdio.h>
#define C_ONLY
#include "numeth/gigarand/gigarand.c"
#include <math.h>
#include <time.h>


#define N 100
#define T_BLOCK 10
#define N_T_BLOCKS 1000

int mod(int k, int M){
    if (k > 0){
        if (k < M){return k;}
        else {return k%M;};
    };
    return M+k;
};

void ising(float beta, float J, float h){

    // time management
    time_t start, end;
    time_t blck_start, blck_end;
    double time_used;
    double blck_mean_time = 0;
    time(&start);
     
    // declarations
    int S[N][N][T_BLOCK];
    printf("size of S is %e\n", (double)sizeof(S));
    int proposal, neighborhood, log_r, accepted=0;
    float u;

    // initialization of matrix
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            S[i][j][0] = 2*(rand()%2) - 1;
        };
    };

    // erases the storage file
    FILE *f = fopen("ising.data", "w");
    fclose(f);

    printf("init fine\n");


    // true cycle of metropolis hastings
    for (int blck_count=0; blck_count < N_T_BLOCKS; blck_count++){
        time(&blck_start);

        for (int blck_iter=0; blck_iter < T_BLOCK; blck_iter++){

            for (int i=0; i<N; i++){
                for (int j=0; j<N; j++){
                        proposal = 2*(rand()%2) - 1;
                        neighborhood = (
                                        S[i][ mod(j+1, N)][blck_iter] +
                                        S[mod(i+1, N)][j][blck_iter] +
                                        S[i][ mod(j-1, N)][blck_iter] +
                                        S[mod(i-1, N)][j][blck_iter]
                                        );
                        log_r = beta*(J*neighborhood + h)*(proposal - S[i][j][blck_iter]);
                        u = ran2();
                        if (log_r > log(u)){
                            accepted++ ;
                            S[i][j][blck_iter+1] = proposal;
                        }
                        else{
                            S[i][j][blck_iter+1] = S[i][j][blck_iter];
                        }
                };
            };
        };

        // saves to file
        FILE *f = fopen("ising.data", "a");
        for (int iter =0; iter<T_BLOCK;iter++){
            for (int i =0; i<N;i++){
                for (int j =0; j<N;j++){
                    fprintf(f, "%d ", S[i][j][iter]);
                }
            }
            fprintf(f, "\n");
        }
        // fwrite(S, sizeof(int), sizeof(S), f);
        fclose(f);

        time(&blck_end);
        blck_mean_time = (blck_mean_time*blck_count + difftime(blck_end,blck_start))/(blck_count + 1);
        if (blck_count % 100 == 0){
            printf("Estimated remaining time: %.1lf", (N_T_BLOCKS - blck_count)*blck_mean_time);
            printf("\t (%d / %d)", blck_count, N_T_BLOCKS);
            printf("\t blck_time/N_T: %lf\n", blck_mean_time/(float)T_BLOCK);
        }

    }
    time(&end);
    printf("\tTotal elapsed time:%.1lf \n", difftime(end, start));
};

int main(){
    ising(2.0, 0.1, 0.1);
    return 0;
};
