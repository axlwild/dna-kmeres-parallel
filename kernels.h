//
// Created by Axel Cervantes on 20/01/21.
//

#ifndef EXAMEN_KERNELS_H
#define EXAMEN_KERNELS_H

#endif //EXAMEN_KERNELS_H
#ifndef PERMS_KMERES
#define PERMS_KMERES 64
#endif
#ifndef K
#define K 3
#endif

// number of permutations of RNA K_meres and k-value
__constant__ char c_perms[PERMS_KMERES][4] = {
        "AAA", "AAC", "AAG","AAT",
        "ACA", "ACC", "ACG","ACT",
        "AGA", "AGC", "AGG", "AGT",
        "ATA", "ATC", "ATG", "ATT",

        "CAA", "CAC", "CAG", "CAT",
        "CCA", "CCC", "CCG", "CCT",
        "CGA", "CGC", "CGG", "CGT",
        "CTA", "CTC", "CTG", "CTT",

        "GAA", "GAC", "GAG", "GAT",
        "GCA", "GCC", "GCG", "GCT",
        "GGA", "GGC", "GGG", "GGT",
        "GTA", "GTC", "GTG", "GTT",

        "TAA", "TAC", "TAG", "TAT",
        "TCA", "TCC", "TCG", "TCT",
        "TGA", "TGC", "TGG", "TGT",
        "TTA", "TTC", "TTG", "TTT",
};

// https://www.geeksforgeeks.org/convert-given-upper-triangular-matrix-to-1d-array/
__device__ __host__ long getIdxTriangularMatrixRowMajor(long i, long j, long n){
    return (n * (i - 1) - (((i - 2) * (i - 1)) / 2)) + (j - i);
}


/*Versión 1: sumar en cada llamada los mínimos entre current_seq y los consecuentes*/
__global__ void minKmeres1(int *sums, float *mins, int num_seqs, int current_seq, int* indexes){
    // guardamos en memoria compartida
    // los kmeros de la entrada actual
    __shared__ int current_seq_kmeres[PERMS_KMERES];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < PERMS_KMERES ){
        current_seq_kmeres[idx] = sums[idx*num_seqs+current_seq];
    }
    __syncthreads();
    int entryLength;
    int compLength;
    if(idx > current_seq && idx < num_seqs){
        //if(idx == 46343){
        float sumMins = 0;
        entryLength = indexes[current_seq + 1] -  indexes[current_seq] - 1;
        compLength = indexes[idx + 1] -  indexes[idx] -1;

        if (entryLength < compLength)
            compLength = entryLength;
        for(int i = 0; i < PERMS_KMERES; i++){
            sumMins += min(current_seq_kmeres[i], sums[idx+num_seqs*i]);
        }
        //float sumbefore = sumMins;
        sumMins = 1 - sumMins/((compLength) - K + 1);
        //printf("Input #%d min_size %d, sum_mins=%f, before: %f\n", idx, compLength, sumMins, sumbefore);
        long aux = getIdxTriangularMatrixRowMajor(current_seq+1, idx - current_seq , (long)num_seqs);
        //  aux = getIdxTriangularMatrixRowMajor(1, 10001 , (long)num_seqs);

        mins[aux] = sumMins;
    }
}


__global__ void minKmeres2(int *sums, float *mins, int num_seqs, int current_seq, int* indexes){
    __shared__ int current_seq_kmeres[PERMS_KMERES];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idxGap = idx + current_seq ;
    int tid = threadIdx.x;
    if(tid < PERMS_KMERES){
        current_seq_kmeres[tid] = sums[tid*num_seqs+current_seq];
    }
    __syncthreads();
    int entryLength;
    int compLength;
    if(idxGap > current_seq && idxGap < num_seqs){
        float sumMins = 0;
        entryLength = indexes[current_seq + 1] -  indexes[current_seq] - 1;
        compLength = indexes[idxGap + 1] -  indexes[idxGap] -1;
        if (entryLength < compLength)
            compLength = entryLength;
        for(int i = 0; i < PERMS_KMERES; i++){
            sumMins += min(current_seq_kmeres[i], sums[idxGap+num_seqs*i]);
        }
        sumMins = 1 - sumMins/((compLength) - K + 1);
        long aux = getIdxTriangularMatrixRowMajor(current_seq+1, idxGap - current_seq , (long)num_seqs);
        mins[aux] = sumMins;
    }
}



__global__ void sumKmereCoincidencesGlobalMemory(char *data, int *indices, unsigned num_seqs, int *sum){
    // each block is comparing a sample with others
    int entry = blockIdx.x;
    // Each thread count all coincidences of a k-mere combination.
    int k_mere = threadIdx.x;
    if ((entry < num_seqs) && (threadIdx.x < PERMS_KMERES)){
        // Fase uno: sumamos todos los valores de la suma de los k-meros de cada entrada.
        // Cada bloque se encarga de cada cadena de entrada
        // Cada hilo se encarga de sumar cada permutación.
        const char *currentKmere = c_perms[k_mere];
        // Entonces cada hilo tendría que iterar toda la muestra solo una vez para calcular la suma.
        int entryLength = indices[entry + 1] -  indices[entry];
        // entonces iteramos por cada letra de la entrada hasta la N-k (los índices).
        // Podríamos guardar los índices en memoria constante para agilizar la lectura...


        // Si ponemos sequence como memoria compartida, podríamos cachar cadenas más grandes...
        char * sequence = data+indices[entry];
        char currentSubstringFromSample[4];
        int counter = 0;
        for (int i = 0; i < entryLength-3; i++){
            memcpy( currentSubstringFromSample, &sequence[i], 3 );
            currentSubstringFromSample[3] = '\0';
            if ((currentSubstringFromSample[0] == currentKmere[0] &&
                    (currentSubstringFromSample[1] == currentKmere[1] &&
                            (currentSubstringFromSample[2] == currentKmere[2])))){
                counter++;
            }
        }
        sum[entry+num_seqs*k_mere] = counter;
    }
}



