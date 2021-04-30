//
// Created by Axel Cervantes on 20/01/21.
//

#ifndef EXAMEN_KERNELS_H
#define EXAMEN_KERNELS_H

#endif //EXAMEN_KERNELS_H


#ifndef K
// la limitación de K se da más por la memoria compartida que por la constante.
// ya que usamos int para el contador en memoria compartida
#define K 3
#endif

#ifndef PERMS_KMERES
#define PERMS_KMERES (1 << (K*2))
#endif
// max shared memory per block
#define MAX_SHARED_MEM 49152
#define MAX_CONSTANT_MEM 65536
// Número máximo de palabras que pueden caber en memoria constante
#define MAX_WORDS MAX_SHARED_MEM / (K+1)
#define MAX_SHARED_INT MAX_SHARED_MEM / sizeof(int)
#define MAX_THREADS 1024
#define THREADS 1024 //min(PERMS_KMERES, 1024)
// number of permutations of RNA K_meres and k-value
__constant__ char c_perms[MAX_WORDS][K+1];
/*
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
*/
// https://www.geeksforgeeks.org/convert-given-upper-triangular-matrix-to-1d-array/
__device__ __host__ long getIdxTriangularMatrixRowMajor(long i, long j, long n){
    return (n * (i - 1) - (((i - 2) * (i - 1)) / 2)) + (j - i);
}


/*Versión 1: sumar en cada llamada los mínimos entre current_seq y los consecuentes*/
__global__ void minKmeres1(int *sums, float *mins, int num_seqs, int current_seq, int* indexes){
    // guardamos en memoria compartida
    // los kmeros de la entrada actual
    __shared__ int current_seq_kmeres[(int)(MAX_SHARED_MEM / sizeof(int))];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < PERMS_KMERES ){
        current_seq_kmeres[idx] = sums[idx*num_seqs+current_seq];
    }
    __syncthreads();
    int entryLength;
    int compLength;
    if(idx > current_seq && idx < num_seqs){
        float sumMins = 0;
        entryLength = indexes[current_seq +1] -  indexes[current_seq] - 1;
        compLength = indexes[idx + 1] -  indexes[idx] -1;

        if (entryLength < compLength)
            compLength = entryLength;
        for(int i = 0; i < PERMS_KMERES; i++){
            sumMins += min(current_seq_kmeres[i], sums[idx+num_seqs*i]);
        }
        //float sumbefore = sumMins;
        sumMins = 1 - sumMins/((compLength) - K + 1);
        //printf("Input #%d in_size %d, sum_mins=%f, before: %f\n", idx, compLength, sumMins, sumbefore);
        long aux = getIdxTriangularMatrixRowMajor(current_seq+1, idx - current_seq , (long)num_seqs);
        //  aux = getIdxTriangularMatrixRowMajor(1, 10001 , (long)num_seqs);

        mins[aux] = sumMins;
    }
}

// Idea: intentar aprovechar los bloques:
// si es par, procesa la secuencia par
// si es impar, procesa la secuencia impar.

__global__ void minKmeres2(int *sums, float *mins, int num_seqs, int num_seq_offset, int* indexes, int perm_offset, bool final, int blocks_per_entry){
    __shared__ int current_seq_kmeres[MAX_SHARED_INT]; // current_seq_kmeres[PERMS_KMERES];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // La secuencia cuyo k-mero se va a guardar en memoria compartida.
    int currentSequence = num_seq_offset + (blockIdx.x / blocks_per_entry);
    // idxGap es la secuencia contra la que se está comparando con el hilo actual.
    int idxGap = (idx / (blocks_per_entry * blockDim.x)) + // Ubica la entrada actual
                    (idx) % (blocks_per_entry * blockDim.x)  // Aumentaría el hilo actual correctamente
                    + 1 // hipótesis, el siguiente del que queremos calcular...
                    ;
    /*
        perm_offset sirve en caso de que haya más kmeros que memoria compartida.
    */
    int tid = threadIdx.x + perm_offset;
    //    if(tid < PERMS_KMERES){
        int auxSharedMem = 0;
        while(auxSharedMem + tid < MAX_SHARED_INT){
            if(auxSharedMem + tid < PERMS_KMERES && tid < PERMS_KMERES && currentSequence < num_seqs)
                current_seq_kmeres[threadIdx.x + auxSharedMem] = sums[(tid+auxSharedMem)*num_seqs+currentSequence];
            else
                current_seq_kmeres[threadIdx.x + auxSharedMem] = 0;
            auxSharedMem += blockDim.x;
        }
    //}
    __syncthreads();
    int entryLength;
    int compLength;
    // if(currentSequence == 1 && threadIdx.x == 0/*&& (blockIdx.x / blocks_per_entry) == 0*/){
    //     printf("ENTRANDO con currentSecuence: %d idxGap: %d. %d Secuencias\n", currentSequence, idxGap, num_seqs);
    // }
    /*
     * Cada hilo debe de hacer el cálculo desde el siguiente de la secuencia actual 
     * hasta num_seqs -1 
     */
    if((idxGap > currentSequence) && (idxGap < num_seqs) && (currentSequence < num_seqs)){
        float sumMins = 0;
        entryLength = indexes[currentSequence + 1] -  indexes[currentSequence] - 1;
        compLength = indexes[idxGap + 1] -  indexes[idxGap] -1;
        if (entryLength < compLength)
            compLength = entryLength;
        long aux = getIdxTriangularMatrixRowMajor(currentSequence+1, idxGap - currentSequence , (long)num_seqs);
        for(int i = 0; i + perm_offset < PERMS_KMERES; i++){
            if(current_seq_kmeres[i] <= sums[idxGap+num_seqs*(i+perm_offset)]){
                sumMins += (float)current_seq_kmeres[i];
            } 
            else {
                sumMins += (float)sums[idxGap+num_seqs*(i+perm_offset)];
            }
                
        }
            
        if(final){
            // if(aux == 1){
            //     printf("Mins[1]: %f\n", mins[1] + sumMins);
            // }
            mins[aux] = 1 - (mins[aux] + sumMins)/((compLength) - K + 1);
        }
        else{
            mins[aux] += sumMins;   
        }
        // if(aux == 1){
        //     printf("Mins[1]: %f\n", mins[1]);
        //     if(final) printf("Final Mins[0]: %f\n", mins[1]);
        // }
    }
}
/*
 * IDEA: hacer la matriz de mínimos transpuesta.
 * Guardar en memoria constante una fila completa (una entrada de kmeros mínimos)
 * 
 * En memoria compartida se guarda el mínimo entre Sk y Sk+1 para cada k-mero.
 * Usar esa memoria compartida para guardar en cada bloque los datos de reducción-
 * Usar paralelismo dinámico para llamar a otro kernel que haga la reducción.
 * Si eso no es pósible, tendríamos que doblar la memoria de las distancias para guardar los 
 * resultados de la reducción. (matriz de #secuencias x bloques)
 */ 
__global__ void minKmeres3(int *sums, float *mins, int num_seqs, int current_seq, int* indexes, int perm_offset, bool final){
    __shared__ int current_seq_kmeres[MAX_THREADS]; // current_seq_kmeres[PERMS0_KMERES];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idxGap = idx + current_seq ;
    int tid = threadIdx.x + perm_offset;
    if(tid < PERMS_KMERES && (threadIdx.x) < MAX_THREADS){
        // current_seq_kmeres[threadIdx.x] = sums[tid*NUM_SEQS+current_seq];
        current_seq_kmeres[threadIdx.x] = sums[current_seq*PERMS_KMERES+tid];
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
        long aux = getIdxTriangularMatrixRowMajor(current_seq+1, idxGap - current_seq , (long)num_seqs);
        for(int i = 0; i + perm_offset < PERMS_KMERES && i < MAX_THREADS; i++){
            if(current_seq_kmeres[i] <= sums[idxGap+num_seqs*(i+perm_offset)]){
                sumMins += (float)current_seq_kmeres[i];
            }
            else {
                sumMins += (float)sums[idxGap+num_seqs*(i+perm_offset)];
            }
        }
        if(final){
            mins[aux] = 1 - (mins[aux] + sumMins)/((compLength) - K + 1);
        }
        else{
            mins[aux] += sumMins;   
        }
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduceSum(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void sumKmereCoincidencesGlobalMemory(char *data, int *indices, unsigned num_seqs, int *sum, int perm_offset, int offset_cm){
    // each block is comparing a sample with others
    int entry = blockIdx.x;
    // Each thread count all coincidences of a k-mere combination.
    int k_mere = threadIdx.x+perm_offset;
    
    if ((entry < num_seqs) && ((k_mere) < PERMS_KMERES )){
        // Fase uno: sumamos todos los valores de la suma de los k-meros de cada entrada.
        // Cada bloque se encarga de cada cadena de entrada
        // Cada hilo se encarga de sumar cada permutación.
        const char *currentKmere = c_perms[offset_cm + threadIdx.x];
        // Entonces cada hilo tendría que iterar toda la muestra solo una vez para calcular la suma.
        int entryLength = indices[entry + 1] -  indices[entry];
        // entonces iteramos por cada letra de la entrada hasta la N-k (los índices).
        // Podríamos guardar los índices en memoria constante para agilizar la lectura...


        // Si ponemos sequence como memoria compartida, podríamos cachar cadenas más grandes...
        char * sequence = data+indices[entry];
        char currentSubstringFromSample[K+1];
        int counter = 0;
        // if(entry == 0 && k_mere == 16383){
        //     printf("Hola\n");
        //     printf("Secuencia: %s\n", sequence);
        //     printf("kmero: %s\n", currentKmere);
        //     printf("memory offset: %d\n", offset_cm);
        //     printf("idx: %d\n", threadIdx.x + offset_cm );
        //     // la memoria que lee es threadIdx.x + offset_cm * MAX_THREADS
        // }
            
        bool isRepetition = true;
        for (int i = 0; i < entryLength-K; i++){
            isRepetition = true;
            memcpy( currentSubstringFromSample, &sequence[i], K );
            currentSubstringFromSample[K] = '\0';
            for(int j = 0; j < K ; j++){
                if (currentSubstringFromSample[j] == currentKmere[j])
                    continue;
                isRepetition = false;
                break;
            }
            if(isRepetition)
                counter++;
        }
        //sum[entry+num_seqs*(k_mere+perm_offset)] = counter;
        sum[entry+num_seqs*(k_mere)] = counter;
        // Transpuesto:
        //sum[k_mere+PERMS_KMERES*(entry)] = counter;
    }
}



