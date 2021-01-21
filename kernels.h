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
__constant__ char c_perms[64][4] = {
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
__device__ __host__ long getIdxTriangularMatrixRowMajor(long i, long j, int n){
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
        //printf("%d\n", current_seq_kmeres[idx]);
    }
    __syncthreads();
    int entryLength;
    int compLength;
    //if(idx > current_seq && idx < num_seqs){
    if(idx == 46343){
        float min = 0;
        float sumMins = 0;
        entryLength = indexes[current_seq + 1] -  indexes[current_seq] - 1;
        compLength = indexes[idx + 1] -  indexes[idx] -1;

        if (entryLength < compLength)
            compLength = entryLength;
        for(int i = 0; i < PERMS_KMERES; i++){
            int cur_kmere = sums[idx+num_seqs*i];
            /*if (idx == num_seqs - 1 && current_seq == 0)
            {
                printf("i = %d, Comparing: %d vs %d\n", i, cur_kmere, current_seq_kmeres[i]);
            }*/
            if(cur_kmere > current_seq_kmeres[i])
                min = current_seq_kmeres[i];
            else
                min = cur_kmere;
            sumMins += min;

            // hacemos la correción de la resta porque no necesitamos la diagonal principal.
        }
        //float sumbefore = sumMins;
        sumMins = 1 - sumMins/((compLength) - K + 1);
        //printf("Input #%d min_size %d, sum_mins=%f, before: %f\n", idx, compLength, sumMins, sumbefore);
        mins[getIdxTriangularMatrixRowMajor(current_seq+1, idx - current_seq , num_seqs)] = sumMins;
    }
}



// intento 1: .350 [s]
__global__ void sumKmereCoincidencesGlobalMemory(char *data, int *indices, unsigned num_seqs, int *sum){
    // each block is comparing a sample with others
    //int idx = threadIdx.x+blockDim.x*blockIdx.x;
    int entry = blockIdx.x;
    // Each thread count all coincidences of a k-mere combination.
    int k_mere = threadIdx.x;
    //printf("outside blockid: %d \n", blockIdx.x);
    if ((entry < num_seqs) && (threadIdx.x < PERMS_KMERES)){
        // Fase uno: sumamos todos los valores de la suma de los k-meros de cada entrada.
        // Cada bloque se encarga de cada cadena de entrada
        // Cada hilo se encarga de sumar cada permutación.
        const char *currentKmere = c_perms[k_mere];
        // Entonces cada hilo tendría que iterar toda la muestra solo una vez para calcular la suma.
        int entryLength = indices[entry + 1] -  indices[entry];
        // entonces iteramos por cada letra de la entrada hasta la N-k (los índices).
        // Podríamos guardar los índices en memoria constante para agilizar la lectura...
        bool is_same_kmere = true;
        char * sequence = data+indices[entry];
        //if (threadIdx.x == 0) printf("Sequence #%d: %s\n", entry, sequence);
        char currentSubstringFromSample[4];
        for (int i = 0; i < entryLength-3; i++){
            memcpy( currentSubstringFromSample, &sequence[i], 3 );
            currentSubstringFromSample[3] = '\0';
            is_same_kmere = true;
            for(int j = 0; j < 3; j++){
                if (currentSubstringFromSample[j] == currentKmere[j]){
                    continue;
                }
                is_same_kmere = false;
                break;
            }
            if(is_same_kmere){
                //printf("Entry(%d) + num_seqs(%d)* kmere(%d) = %d\n", entry, num_seqs, k_mere, entry+num_seqs*k_mere);
                sum[entry+num_seqs*k_mere] += 1;
            }
        }
        __syncthreads();
    }
}



