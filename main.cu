#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <math.h>
#include <algorithm>
#include <sstream>
#include <typeinfo>
#include "cuda.h"
#include <cooperative_groups.h>
#include <limits.h>

using namespace std;
#define PERMS_KMERES 64
int          numberOfSequenses = 0;
char *       all_seqs;
unsigned int size_all_seqs = 0;


// Method definition
void importSeqs(string inputFile);
void printSeqs();
void getPermutations(char *str, char* permutations, int last, int index);
int permutationsCount(string permutation, string sequence, int k);
void sequentialKmerCount(vector<string> &seqs, vector<string> &permutations , int k);
void doParallelKmereDistance();
void doSequentialKmereDistance();
// Vectors to store ids and seqs
vector<string> ids;
vector<string> seqs;
vector<int> indexes_aux;

// Device variables.
char    * d_data; // all the strings.
int     * d_indices;
float   * d_distances;
int * d_sums; // coincidences of k-mer on each input
int * d_mins, *h_mins;
int * h_sums;
int minsSize;

string file = "/home/acervantes/kmerDist/plants.fasta";
//string file = "/home/acervantes/kmerDist/all_seqs.fasta";

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

char perms[64][4] = {
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
int permsSize = sizeof(perms) ;

vector<string> permutationsList (perms, end(perms));

float ** distancesSequential;
float ** distancesParallel;
/*
string join(const vector<string>& vec, const char* delim){
    stringstream res;
    copy(vec.begin(), vec.end(), ostream_iterator<string>(res, delimiter.c_str()));
    return res.str();
}
 */
string join(const std::vector<std::string> &lst, const std::string &delim){
    std::string ret;
    for(const auto &s : lst) {
        if(!ret.empty())
            ret += delim;
        ret += s;
    }
    return ret;
}
/**
 *
 * @param data:    buffer con todas las cadenas
 * @param indices: índices donde inicia cada cadena nueva en los datos
 * @param distances: matriz resultante de las distancias
 * @param num_seqs: número de cadenas de entrada.
 * @param suma: Matriz de rxc donde cada renglón equivale a las coincidencias de cada k-mero
 *              en una cadena de entrada (columna).
 */
__global__ void sumKmereCoincidences(char *data, int *indices, unsigned num_seqs, int *sum){
    // each block is comparing a sample with others
    //int idx = threadIdx.x+blockDim.x*blockIdx.x;
    int entry = blockIdx.x;
    // Each thread count all coincidences of a k-mere combination.
    int k_mere = threadIdx.x;
    //int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int idx    = blockIdx.x * PERMS_KMERES + threadIdx.x;
    extern __shared__ int sDataSum[];
    int maxIdxSM = (int) (49152 / sizeof(int));
    //printf("outside blockid: %d \n", blockIdx.x);
    // 12288
    if ((blockIdx.x < num_seqs) && (threadIdx.x < PERMS_KMERES)){
        if(idx < maxIdxSM )
            sDataSum[idx] = 0;
        __syncthreads();
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
                if(idx < maxIdxSM)
                    sDataSum[idx] += 1;
                else
                    sum[idx] += 1;
            }
        }
        __syncthreads();
        if (idx < maxIdxSM)
            sum[idx] = sDataSum[idx];
    }
}
// intento 1: .350 [s]
__global__ void sumKmereCoincidencesGlobalMemory(char *data, int *indices, unsigned num_seqs, int *sum){
    // each block is comparing a sample with others
    //int idx = threadIdx.x+blockDim.x*blockIdx.x;
    int entry = blockIdx.x;
    // Each thread count all coincidences of a k-mere combination.
    int k_mere = threadIdx.x;
    //int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int idx    = blockIdx.x * PERMS_KMERES + threadIdx.x;
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
                sum[idx] += 1;
            }
        }
        __syncthreads();
    }
}


/*
__global__ void minKmerzeDist(int *sums, double *distances, int num_seqs, double *mins){
    // Cada bloque se encargará de calcular un k-mero
    // Los hilos distribuirán la tarea de calcular el mínimo entre las entradas.
    // Distances poddría ser una matriz triangular o podría ser un arreglo cuyo acceso sea por una función
    // hash para ahorrar memoria.

    // En el bloque comparamos todas las coincidencias con los k-meros
    int current_kmere = blockIdx.x;
    int current_seq   = threadIdx.x;
    int idx           = blockIdx.x*blockDim.x+threadIdx.x;
    int comparisons   = current_seq - num_seqs;
    int min           = INT_MAX;
    // TODO: caso para más de 1024 entradas, se necesita seguir ejecutando hasta que todas las entradas se puedan comparar
    if(current_seq < num_seqs - 1){
        // Cada iteración es la comparación desde la secuencia 'start' hasta la última.
        for(int start = 0; start < num_seqs - 1; start++){
            //Se calculan los mínimos de una entrada contra las restantes
             if (start < current_seq)
                min = (sums[idx+start] < sums[idx + start + current_seq + 1]) ? sums[idx+start] : sums[idx + start + current_seq + 1];
            __syncthreads();

        }
    }
    __syncthreads();

    mins[idx] =

}*/

// https://www.geeksforgeeks.org/convert-given-upper-triangular-matrix-to-1d-array/
__device__ __host__ int getIdxTriangularMatrixRowMajor(int i, int j, int n){
    return (n * (i - 1) - (((i - 2) * (i - 1)) / 2)) + (j - i);
}

// extracted and modified from MK Programming Massively. 2nd Edition p.209
__global__ void parallelSum(float *results, int idxResult, int InputSize) {
    __shared__ int XY[PERMS_KMERES];
    extern __shared__ int min_sums[];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < InputSize) {
        XY[threadIdx.x] = min_sums[i];
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x+1) * 2* stride -1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }
    for (int stride = PERMS_KMERES/4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index + stride < PERMS_KMERES) {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    if (i == 0)
        results[idxResult] = XY[InputSize-1];
}

__global__ void minKmereDist(int *sums, float *distances, int num_seqs, int start){
    int current_seq   = blockIdx.x;
    int idx           = current_seq+blockDim.x*threadIdx.x;
    //int idxDist       = start*blockDim.x+current_seq;
    int min_sums[PERMS_KMERES] = {0};
    // Se guarda en memoria compartida las repeticiones de los kmeros de las dos entradas a comparar.
    __shared__ int seqs_sums[PERMS_KMERES*2];
    //TODO: ¿la variable totalsum pueden verlos los demás bloques?
    //int totalSum;

    if(current_seq < num_seqs - 1){

        for(int i = 0, j=0; i < PERMS_KMERES; i++){
            seqs_sums[j++]   = sums[idx];
            seqs_sums[j++]   = sums[idx+1];
        }
        __syncthreads();
        for(int i = 0; i < PERMS_KMERES; i++){
            if (seqs_sums[2*i] < seqs_sums[2*i+1]){
                min_sums[i] += seqs_sums[2*i];
            }
            else{
                printf("");
                min_sums[i] += seqs_sums[2*i+1];
            }
        }
        __syncthreads();
        //parallelSum<<<1,64>>>(distances,idxDist+1, PERMS_KMERES);
    }

}
/*
 * @param sums:    matriz que contiene el número de coincidencias de cada k-mero en cada cadena.
 * @param mins:    matriz de las sumas de los k-meros mínimos entre las cadenas i e i+1, respectivamente
 *                  (tamaño num_seqsxnum_seqs).
 *                  Por el momento se implementó un arreglo equivalente a una matriz triangular con el fin
 *                   de reducir la memoria.
 * @param num_seqs: secuencias totales a considerar.
 * @param num_kmeres: combinaciones totales de k-meros.
 *
 */
//  TODO: checar si num_kmeres se puede usar para inicializar un arreglo aquí.
__global__ void minKmeres(int *sums, int *mins, int num_seqs, int num_kmeres){
    int current_seq   = blockIdx.x;
    int current_kmere  = threadIdx.x;
    int current_comp;
    int idxMin;
    // es del tamaño de la ent
    __shared__ int PIVOT[PERMS_KMERES];
    int i;
    //j;
    int jump = 1;
    int current_sum = 0;

    // Guardamos en memoria compartida el pivote porque se accederá dos veces.
    if(current_seq < num_seqs && current_kmere < num_kmeres) {// initialize PIVOT
        PIVOT[current_kmere] = sums[current_seq+current_kmere*blockDim.x];
        __syncthreads();
        while(jump+current_seq < num_seqs){
            for(i = current_seq; i < num_seqs; i+=jump){
                current_comp = sums[current_seq+i+current_kmere*num_seqs];
                current_sum  = current_comp < PIVOT[current_kmere] ?
                               current_comp : PIVOT[current_kmere];
            }
            jump++;
        }
        __syncthreads(); // TODO: verificar que no afecta esto al resultado, pero reduce las colisiones.
        idxMin = getIdxTriangularMatrixRowMajor(current_seq+1, current_comp+1, num_kmeres);
        atomicAdd(&mins[idxMin], current_sum);
    }
}



int main(int argc, char **argv) {
    //char permutations[len];
    // absolute path of the input data
    importSeqs(file);
    std::cout << "Size all seqs:" << size_all_seqs << std::endl;
    // Device allocation
    doParallelKmereDistance();
    return 0;
}

void doSequentialKmereDistance(){
    // results files
    FILE *f_seq_res = fopen("/home/acervantes/kmerDist/sequential_results.csv", "w");

    distancesSequential = (float**) malloc(sizeof(float*) * numberOfSequenses);
    //distancesParallel   = (float**) malloc(sizeof(float*) * numberOfSequenses);
    for(int i = 0; i < numberOfSequenses; i++){
        distancesSequential[i] = (float*) malloc(numberOfSequenses*sizeof(float));
        //distancesParallel[i]   = (float*) malloc(numberOfSequenses*sizeof(float));
    }
    for (int i = 0; i < numberOfSequenses ; i++){
        for (int j = 0; j < numberOfSequenses ; j++) {
            distancesSequential[i][j] = -1;
        }
    }
    clock_t start_ser = clock();
    sequentialKmerCount(seqs, permutationsList, 3);
    clock_t end_ser = clock();
    double serialTimer = 0;
    serialTimer = double (end_ser-start_ser) / double(CLOCKS_PER_SEC);
    cout << "Elapsed time serial: " << serialTimer << "[s]" << endl;

    for (int i = 0; i < numberOfSequenses ; i++){
        for (int j = 0; j < numberOfSequenses ; j++) {
            fprintf(f_seq_res,"%f ",distancesSequential[i][j]);
            //printf("%f ",distancesSequential[i][j]);
            //distancesParallel[i][j] = 0;
        }
        fprintf(f_seq_res,"\n");
        //printf("\n");
    }
    fclose(f_seq_res);
}

void doParallelKmereDistance(){
    cudaError_t error;
    /**
     * Inicialización
     * */

    // Buffer de cadenas
    error = cudaMalloc((void **)&d_data, size_all_seqs);
    if (error){
        printf("Error #%d allocating device memory with data.", error);
        exit(1);
    }
    error = cudaMemcpy(d_data, all_seqs, size_all_seqs, cudaMemcpyHostToDevice);
    if (error){
        printf("Error copying data from host %d\n", error);
        exit(1);
    }

    // Los índices de las entradas de las cadenas.
    int *indexes = (int *) malloc((int)indexes_aux.size() * sizeof(int));
    for (int i = 0; i < indexes_aux.size(); i++){
        indexes[i] = indexes_aux[i];
    }
    int indexesBytesSize = indexes_aux.size()*sizeof(int);
    error = cudaMalloc((void **)&d_indices, indexesBytesSize);
    if (error){
        printf("Error malloc %d", error);
    }
    error = cudaMemcpy(d_indices, indexes, indexesBytesSize, cudaMemcpyHostToDevice);
    if (error) printf("Error copying data from device %d\n", error);

    // Suma de cada kmero de cada entrada.
    int numSumResults = numberOfSequenses*PERMS_KMERES; // sequences x 4**3
    int sumsSize = sizeof(int)*numSumResults;
    h_sums = (int*) malloc(sumsSize);
    for (int i = 0; i < numSumResults; i++){
        h_sums[i] = 0;
    }
    cudaMalloc((void **)&d_sums, sumsSize);
    error = cudaMemcpy(d_sums, h_sums, sumsSize, cudaMemcpyHostToDevice);
    if (error){
        printf("Error copying data from host %d", error);
        exit(1);
    }
    /* // defining a constant value is passed to the device directly.
    error = cudaMemcpyToSymbol(c_perms, &perms, 4*64 * sizeof(char) );
    if (error){
        printf("Errorsti : %d: %s\n", error, cudaGetErrorString(error));
    }
    */
    //int indexes[indexes_aux.size()];

    /**
     * Mins: será una estructura de datos para minimizar el uso de memoria.
     * En total, sería de n (n+1) / 2 donde n es el número de muestras.
     * */
    minsSize = (int) (numberOfSequenses*(numberOfSequenses+1) / 2);
    h_mins   = (int*) malloc(minsSize * sizeof(int));
    for(int i = 0; i < minsSize; i++)
        h_mins[i] = 0;
    int threads = 64;
    int blocks = numberOfSequenses;
    //int blocks = 10;
    //int threads = 64;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaError_t err_;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Running %d blocks and %d threads\n", blocks, threads);
    // Launch kernel
    //int smSize = 49152;
    //sumKmereCoincidences<<<blocks, threads, smSize>>>(d_data, d_indices, numberOfSequenses, d_sums);
    cudaEventRecord(start,0);
    sumKmereCoincidencesGlobalMemory<<<blocks, threads>>>(d_data, d_indices, numberOfSequenses, d_sums);
    //minKmereDist<<<10, 1024>>>(d_sums,d_distances, numberOfSequenses, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float parallelTimer = 0;
    cudaEventElapsedTime(&parallelTimer, start, stop);
    cout<< "Elapsed parallel timer: " << parallelTimer << " ms, " << parallelTimer / 1000 << " secs" <<endl;
    err_ = cudaGetLastError();
    if (err_)
        printf("LastError kmere dist: %d\n", err_);
    //minKmeres<<<blocks, threads>>>(d_sums, d_mins, numberOfSequenses, PERMS_KMERES);
    //cudaDeviceSynchronize();
    err_ = cudaGetLastError();
    if (err_)
        printf("LastError min kmeres: %d\n", err_);

    //unsigned long int sizeDistances = numberOfSequenses*numberOfSequenses * sizeof(float);
    //cudaMemcpy(h_distances, d_distances, sizeDistances, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_sums, d_sums, sumsSize, cudaMemcpyDeviceToHost);
    /*
    printf("Sums:\n");
    for(int j = 0, idx = 0; j < 64; j++){
        printf("%d: ", j);
        for(int i = 0; i < numberOfSequenses; i++){
            printf("%d,\t", h_sums[idx++]);
        }
        printf("\n");
    }*/

    free(distancesSequential);
    //free(distancesParallel);
    cudaFree(d_distances);
    cudaFree(d_indices);
    cudaFree(d_data);
    cudaFree(d_sums);
    cudaFree(d_mins);

    return;
    /*
    error = cudaMalloc((void **)&d_distances, sizeDistances);
    if (error){
        printf("Error al usar memoria con distancia %d ::", error);
        cout << sizeDistances << endl;
        return 0;
    }*/


    //    float *h_distances;
    //    h_distances =(float*) malloc(sizeDistances);
    //    int dimsDistances = numberOfSequenses*numberOfSequenses;
    //    for(int i=0; i<dimsDistances; i++){
    //        h_distances[i] = 0;
    //    }


    /*
    error = cudaMemcpy(d_distances, h_distances, sizeDistances, cudaMemcpyHostToDevice);
    if (error){
        printf("Error copying distances matrix from host %d", error);
    }*/

    /*
    error = cudaMalloc((void **)&d_mins, minsSize*sizeof(int));
    if (error){
        printf("Error #%d allocating memory to d_mins", error);
        exit(1);
    }
    error = cudaMemcpy(d_mins, h_mins, minsSize*sizeof(int), cudaMemcpyHostToDevice);
     if (error){
        printf("Error copying data from host %d", error);
        exit(1);
    }
     */

}
void importSeqs(string inputFile){
    int indexCounter = 0;
    ifstream input(inputFile);
    if (!input.good()) {
        std::cerr << "Error opening: " << inputFile << " . Check your file or pathh." << std::endl;
        exit(0);
    }
    string line;
    string acc = "";
    string globalAcc = "";
    bool newSeq = false;

    // Iterate over all secuences
    while (getline(input, line)) {

        // line may be empty so you *must* ignore blank lines
        // or you have a crash waiting to happen with line[0]
        if(line.empty()){
            continue;
        }
        //read the header of
        if (line[0] == '>') {
            // store id
            ids.push_back(line);
            newSeq = true;
            continue;
        }
        if (newSeq) {
            newSeq = false;
            acc = line;
            while (getline(input, line)) {
                if(line.empty() || line[0] == 13){
                    acc += "|";
                    seqs.push_back(acc);
                    indexes_aux.push_back(indexCounter);
                    indexCounter += acc.size();
                    globalAcc += acc;
                    acc = "";
                    break;
                }
                acc += line;
            }
            if (acc != ""){
                acc += "|";
                seqs.push_back(acc);
                indexes_aux.push_back(indexCounter);
                indexCounter += acc.size();
                globalAcc += acc;
                acc = "";
                indexes_aux.push_back(indexCounter);
            }
        }
    }
    int last_index = indexes_aux.size();
    numberOfSequenses = seqs.size();
    size_all_seqs = globalAcc.size()*sizeof(char);
    all_seqs = (char *) malloc(size_all_seqs);
    for(int i = 0; i < globalAcc.size(); i++){
        if (globalAcc[i] == '|'){
            all_seqs[i] = '\0';
            continue;
        }
        all_seqs[i] = globalAcc[i];
    }
    return;
}

void sequentialKmerCount(vector<string> &seqs, vector<string> &permutations , int k){
    string mers[4] = {"A","C","G","T"};
    int numberOfSequences = seqs.size();
    // |kmers| is at most 4**k = 4**3 = 64
    int max_combinations = pow(4,k);
    // Comparing example Ri with R(i+1) until Rn
    for(int i =  0; i < numberOfSequences - 1; i++){
        for(int j = i + 1; j < numberOfSequences; j++){
            if(i >= j)
                continue;
            // iterating over permutations (distance of Ri an Rj).
            int minLength = min(seqs[i].size(), seqs[j].size());
            int sum = 0;
            float distance = -1.0f;
            for(int p = 0; p < max_combinations; p++){
                int minimum = min(
                        permutationsCount(permutations[p], seqs[i],k),
                        permutationsCount(permutations[p], seqs[j],k)
                );
                sum += minimum;
            }

            distance = 1 - (float) sum / (minLength - k + 1);
            distancesSequential[i][j] = distance;
            distancesSequential[j][i] = distance;
        }
    }
    return;
}

int permutationsCount(string permutation, string sequence, int k){
    int sequence_len = sequence.size();
    int counter = 0;
    string current_kmere;
    for(int i = 0; i < sequence_len - k; i++){
        current_kmere = sequence.substr(i,i+k);
        if (permutation.compare(current_kmere) == 0){
            counter++;
        }
    }
    return counter;
}

void getPermutations(char *str, char* permutations, int last, int index){
    string stri;

    int i, len = strlen(str);
    for ( i = 0; i < len; i++ ) {
        permutations[index] = str[i] ;
        if (index == last){
            stri = permutations;
            permutationsList.push_back(stri);
        }
        else
            getPermutations (str, permutations, last, index+1);
    }
}

void printSeqs(){
    cout<< "total number of seqs: " << seqs.size() << endl;
    for (int i = 0; i<seqs.size(); i++){
        cout << ">" <<  seqs[i] << endl;
    }
}
