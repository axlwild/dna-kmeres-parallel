//
// Created by Axel Cervantes on 20/01/21.
//

#ifndef EXAMEN_UTILS_H
#define EXAMEN_UTILS_H

#endif //EXAMEN_UTILS_H

void printMinDistances(float* mins, long minsSize, int N){
    printf("writing file...");
    FILE *f = fopen("/home/acervantes/kmerDist/min_distances.csv", "w");
    for (int i = N-1, aux = 0; i > 0; i--, aux++){
        for(int j = 1; j <= i; j++ ){
            fprintf(f, "%.2f\t", mins[aux]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return;
}