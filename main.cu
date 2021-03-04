#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>
#include <typeinfo>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda.h"
#include <cooperative_groups.h>
#include <climits>
#include "utils.h"
#include <map>
#include "kernels.h"
#include "utils.cpp"

#ifndef PERMS_KMERES
#define PERMS_KMERES 64
#endif


#define THREADS 64
#define N (54018*1024*128)
#define PRINT_ANSWERS false
#define PRINT_ANSWERS_FILE true

#define BLOCKS_STEP_1 40000
#define MAX_SEQS 1000
#define VERBOSE true

using namespace std;
int          numberOfSequenses = 0;
unsigned int size_all_seqs = 0;

long threads = THREADS;
//long blocks = 32768; // 2.61
//long blocks = 30000; // 2.32
//long blocks = 5000; // 2.0511
long blocks = 1000; // 2.012

// TODO: dinámico, si el número de hilos necesario es menor de 1024, usar esa cantidad
int threadsStep1 = 1024;
int blockThread1 = BLOCKS_STEP_1;
bool bug_log = false;
string file = "/home/acervantes/kmerDist/plants.fasta";
//string file = "/home/acervantes/kmerDist/all_seqs.fasta";
// to run this, execute importSeqsNoNL.
//string file = "/home/acervantes/kmerDist/genomic.fna";
// Method definition
void importSeqs(string inputFile); 
void importSeqsNoNL(string inputFile);

void printSeqs();
void getPermutations(char *str, char* permutations, int last, int index);


void sequentialKmerCount2(vector<string> &seqs, vector<string> &permutations , int k);
void doParallelKmereDistance();
void doSequentialKmereDistance();
void permutationsCountAll(string sequence, int * countResults, int max_combinations, int k);
long getIdxTriangularMatrixRowMajorSeq(long i, long j, long n);
// Vectors to store ids and seqs
vector<string> ids;
vector<string> seqs;
vector<int> indexes_aux;
// Device variables.
char    *data; // all the strings.
int     *indexes;
float   *distances;
int     *sums; // coincidences of k-mer on each input
float   *mins;
long minsSize;
long resultsArraySize;


// 4 cadenas sería 1 bit
// 1111 1111 = 1 byte
// 00 -> A
// 01 -> C
// 10 -> G
// 11 -> T

// AACG -> 00000110
// AACGA -> 00000110 00__ ____

//__constant__ char c_perms[][4];
//char perms[PERMS_KMERES][K+1];



std::map<std::string, int> permutationsMap;
vector<string> permutationsList;
char **perms;
float * distancesSequential;

int main() {

    const char *alphabet = "ACGT";
    int sizeAlphabet = 4;
    int permsSize    = pow(sizeAlphabet, K);
    perms = (char**) malloc(permsSize * sizeof(char*));
    for(int i = 0; i < permsSize; i++)
    {
        perms[i] = (char*) malloc((K+1)*sizeof(char));
    }
    permutation(alphabet, K, perms);
    for(int i = 0; i < permsSize; i++){
        permutationsList.push_back(perms[i]);
    }
    for(int i = 0; i < PERMS_KMERES; i++)
        permutationsMap[perms[i]] = i+1;
    /*
     * For k = 3 and |alphabet| = 4
     * we need we need 4**3 combinations size 3
     * 64 combinations size 3+1 bytes (end of string)
     * 192 bytes
     * */
    // We need to copy permutations to device constant memory
    // 65536 max constant memory
    if (VERBOSE){
        std::cout << "K = " << K << std::endl;
        std::cout << PERMS_KMERES * sizeof(char) * (K+1) << "/65536 bytes allocated" << std::endl;
        std::cout << "Máximum " << MAX_WORDS << " words in constant memory" << std::endl;
        std::cout << "(words size "<< K+1 << " bytes)";
        
    }




    // absolute path of the input data
    importSeqs(file);
    //importSeqsNoNL(file);
    resultsArraySize = numberOfSequenses*(numberOfSequenses+1) / 2 - numberOfSequenses;
    std::cout << "Size all seqs:" << size_all_seqs << std::endl;
    std::cout << seqs.size() << " sequences read ." << std::endl;

    doSequentialKmereDistance();
    printf("\n\aParallel:\n");
    // Device allocation
    doParallelKmereDistance();
    return 0;
}

void doSequentialKmereDistance(){
    // results files
    FILE *f_seq_res = fopen("/home/acervantes/kmerDist/sequential_results.csv", "w");
    //distancesSequential = (float**) malloc(sizeof(float*) * numberOfSequenses);
    //distancesParallel   = (float**) malloc(sizeof(float*) * numberOfSequenses);
    distancesSequential = (float*) calloc(resultsArraySize, sizeof(float));

    //    for (int i = 0; i < numberOfSequenses ; i++){
    //        for (int j = 0; j < numberOfSequenses ; j++) {
    //            distancesSequential[i][j] = -1;
    //        }
    //    }
    clock_t start_ser = clock();
    sequentialKmerCount2(seqs, permutationsList, 3);
    clock_t end_ser = clock();
    double serialTimer = 0;
    serialTimer = double (end_ser-start_ser) / double(CLOCKS_PER_SEC);
    cout << "Elapsed time serial: " << serialTimer << "[s]" << endl;
    if(PRINT_ANSWERS)
    for (long i = 0; i < resultsArraySize; i++){
        printf("%f\n", distancesSequential[i]);
    }

    if(PRINT_ANSWERS_FILE)
    for (long i = 0; i < resultsArraySize; i++){
        fprintf(f_seq_res,"%f\n", distancesSequential[i]);
    }
    /*for (long i = numberOfSequenses - 1, idx = 0; i > 0 ; i--){
        for (long j = 0; j < i ; j++, idx++) {
            fprintf(f_seq_res,"%f\t",distancesSequential[idx]);
            printf("%f(%ld)\t",distancesSequential[idx], idx);
            //distancesParallel[i][j] = 0;
        }
        fprintf(f_seq_res,"\n");
        printf("\n");
    }*/
    fclose(f_seq_res);
}

void doParallelKmereDistance(){
    FILE *f_res = fopen("/home/acervantes/kmerDist/parallel_results.csv", "w");
    cudaError_t error;
    /**
     * Inicialización
     * */
    // Los índices de las entradas de las cadenas.
    int numIndexes = indexes_aux.size();
    error = cudaMallocManaged(&indexes, numIndexes * sizeof(int));
    if (error){
        printf("Error malloc indexes: error #%d\n", error);
        exit(1);
    }
    for (int i = 0; i < indexes_aux.size(); i++){
        indexes[i] = indexes_aux[i];
    }

    // Suma de cada kmero de cada entrada.
    int numSumResults = numberOfSequenses*PERMS_KMERES; // sequences x 4**3
    int sumsSize = sizeof(int)*numSumResults;
    error = cudaMallocManaged(&sums, sumsSize);
    if (error){
        printf("Error malloc indexes: error #%d\n", error);
        exit(1);
    }
    for (int i = 0; i < numSumResults; i++){
        sums[i] = 0;
    }


    /**
     * Mins: será una estructura de datos para minimizar el uso de memoria.
     * En total, sería de n (n+1) / 2 donde n es el número de muestras.
     * Pero como tampoco es necesaria la matriz principal, se resta N elementos
     */
    minsSize = (long) ((long)numberOfSequenses*((long)numberOfSequenses+1) / 2) - numberOfSequenses;
    error = cudaMallocManaged(&mins, minsSize*sizeof(float));
    if (error){
        printf("Error malloc mins: error #%d\n", error);
        exit(1);
    }
    printf("%d sequences founded.\n", numberOfSequenses);
    printf("Allocating %ld elements of distance results.\n", minsSize);
    for(int i = 0; i < minsSize; i++)
        mins[i] = 0;
    //int blocks = 10;
    //int threads = 64;
    cudaEvent_t start;
    cudaEvent_t globalStart;
    cudaEvent_t stop;
    cudaEvent_t globalStop;
    cudaError_t err_;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&globalStart);
    cudaEventCreate(&globalStop);


    // Launch kernel
    //int smSize = 49152;
    //sumKmereCoincidences<<<blocks, threads, smSize>>>(d_data, d_indices, numberOfSequenses, d_sums);
    /**
     * Primera parte: se obtiene la matriz d_mins que contiene las distancias mínimas
     * de todos los kmeros de cada entrada.
     * Km1S1, Km1S2, Km1S3, ... , Km1Sn
     * Km2S1, Km2S2, Km2S3, ... , Km2Sn
     * Km3S1, Km3S2, Km3S3, ... , Km3Sn
     * .
     * .
     * .
     * Km64S, Km64S2, Km64S3, ... , Km64Sn
     * */
    cudaEventRecord(start, nullptr);
    cudaEventRecord(globalStart, nullptr);
    printf("Running %ld blocks and %ld threads\n", blocks, threads);
    /***
     *  Para evitar desborde de memoria compartida, se calendariza
     *  el ciclo para alcanzar a procesar todos los kmeros.
     *  Cada hilo analiza un kmero en memoria compartida.
     *
     */
    // TODO: asignación dinámica para valores mayores a K=6
    

    //for(int perm_offset = 0; perm_offset < PERMS_KMERES; perm_offset += MAX_WORDS){
    for(int perm_offset = 0; perm_offset < 1 ; perm_offset += MAX_WORDS){
        // Actualizamos los valores de las permutaciones de k-meros
        for(int i = 0; i < MAX_WORDS && i < PERMS_KMERES; i++){
            //std::cout << "Copying:" << perms[i] << std::endl;
            err_ = cudaMemcpyToSymbol(c_perms, perms[i+perm_offset], (K+1), i*(K+1));
            if(err_){
                std::cout << "Error i=" << i << " OFFSET: " << perm_offset << " error #" << err_ << std::endl;
                return;
            }
        }
        // TODO: Si el número de permutaciones (MAX_WORDS) excede el número de hilos (1024), calendarizar también
        sumKmereCoincidencesGlobalMemory<<<blockThread1, threadsStep1>>>
                (data, indexes, numberOfSequenses, sums, perm_offset);
        cudaDeviceSynchronize();
        err_ = cudaGetLastError();
        if (err_)
            printf("LastError sumCoincidences #%d\n", err_);
        else
            printf("Running ok\n");
        return;        
    }
    return;
    // sumKmereCoincidencesGlobalMemory<<<blockThread1, threadsStep1>>>(data, indexes, numberOfSequenses, sums);
    // cudaDeviceSynchronize();
    // err_ = cudaGetLastError();
    // if (err_)
    //     printf("LastError sumCoincidences #%d\n", err_);
    cudaFree(data);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    float parallelTimer = 0;
    cudaEventElapsedTime(&parallelTimer, start, stop);
    cout<< "Elapsed parallel timer step 1: " << parallelTimer << " ms, " << parallelTimer / 1000 << " secs" <<endl;
    /*printf("Sums:\n");
    for(int j = 0, idx = 0; j < PERMS_KMERES; j++){
        printf("%d: ", j);
        for(int i = 0; i < numberOfSequenses; i++){
            printf("%d,\t", sums[idx++]);
        }
        printf("\n");
    }
    printf("\n");*/
    /**
     * Paso 2: calcular las distancias de todo vs todo.
     *      Para toda cadena i:
     *        - Obtener distancia desde i+1 hasta n.
     * Paso 3: Al calcular las distancias, aplicar la fórmula.
     * Versión 1: Reducción de operaciones por llamada al kernel.
     * Versión 2: utilizar una localización de hilos tal que se ejecute siempre lo mismo y
     *            no se desperdicie memoria
     *
     * */
    //minKmereDist<<<10, 1024>>>(d_sums,d_distances, numberOfSequenses, 0);


    //minKmeres<<<blocks, 64>>>(d_sums, d_mins, numberOfSequenses);
    // sin ejecutar kernel tarda aprox 344 ms
    // ejecutando kernel 374 ms
    cudaEventRecord(start, nullptr);
    for(int i = 0; i < numberOfSequenses; i++){
        for(int perm_offset = 0; perm_offset < PERMS_KMERES; perm_offset += (int)MAX_SHARED_MEM / sizeof(int)){
            minKmeres2<<<blocks, threads>>>(sums, mins, numberOfSequenses, i, indexes, perm_offset);
            cudaDeviceSynchronize();
            err_ = cudaGetLastError();
            if (err_){
                printf("LastError kmere dist: %d iteration %d\n", err_, i);
                exit(1);
            }
        }
    }
    cudaDeviceSynchronize();
    err_ = cudaGetLastError();
    if (err_)
        printf("LastError kmere dist: %d\n", err_);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    parallelTimer = 0;
    cudaEventElapsedTime(&parallelTimer, start, stop);
    cout<< "Elapsed parallel step 2 timer: " << parallelTimer << " ms, " << parallelTimer / 1000 << " secs" <<endl;

    cudaEventRecord(globalStop,0);
    cudaEventSynchronize(globalStop);
    parallelTimer = 0;
    cudaEventElapsedTime(&parallelTimer, globalStart, globalStop);
    cout<< "Total time elapsed parallel: " << parallelTimer << " ms, " << parallelTimer / 1000 << " secs" <<endl;
    if(PRINT_ANSWERS)
    for (long i = 0; i < minsSize; i++){
        printf("%f\n", mins[i]);
    }
    if(PRINT_ANSWERS_FILE)
    for (long i = 0; i < resultsArraySize; i++){
        fprintf(f_res,"%f\n", mins[i]);
    }
    /*for (long i = numberOfSequenses - 1, idx = 0; i > 0 ; i--, idx++){
        for (long j = 0; j < i ; j++) {
            printf("%f\t",mins[idx]);
        }
        printf("\n");
    }*/

    /*printf("SumaMins:\n");
    for(int i = 0; i < minsSize; i++){
        printf("%f\t", mins[i]);
    }
    printf("\n");*/
    //printMinDistances(h_mins, minsSize, numberOfSequenses);
    /* // Para comprobar que los índices están bien.
    for(int i = 0; i < numIndexes - 1; i++){
        std::cout << "idx: " << indexes[i] << "\tCadena " << i << " :" << all_seqs+indexes[i]+1 << std::endl;
    }*/

    //unsigned long int sizeDistances = numberOfSequenses*numberOfSequenses * sizeof(float);
    //cudaMemcpy(h_distances, d_distances, sizeDistances, cudaMemcpyDeviceToHost);
    /*
    printf("Sums:\n");
    for(int j = 0, idx = 0; j < 64; j++){
        printf("%d: ", j);
        for(int i = 0; i < numberOfSequenses; i++){
            printf("%d,\t", h_sums[idx++]);
        }
        printf("\n");
    }
    */

    free(distancesSequential);
    //free(distancesParallel);
    cudaFree(distances);
    cudaFree(indexes);
    cudaFree(sums);
    cudaFree(mins);

    return;

}

void importSeqsNoNL(string inputFile){
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
                if(line[0] == '>') newSeq = true;
                if(line.empty() || line[0] == 13 || line[0] == '>'){
                    acc += "|";
                    seqs.push_back(acc);
                    indexes_aux.push_back(indexCounter);
                    indexCounter += acc.size();
                    globalAcc += acc;
                    acc = "";
                    break;
                }
                acc += line;
                if(seqs.size() >= MAX_SEQS) break;
            }
            if (acc != ""){
                acc += "|";
                seqs.push_back(acc);
                indexes_aux.push_back(indexCounter);
                indexCounter += acc.size();
                globalAcc += acc;
                acc = "";
                indexes_aux.push_back(indexCounter);
                if(seqs.size() >= MAX_SEQS) break;
            }
        }
    }
    int last_index = indexes_aux.size();
    numberOfSequenses = seqs.size();
    size_all_seqs = globalAcc.size()*sizeof(char);
    cudaError_t error;
    error = cudaMallocManaged(&data, size_all_seqs);
    if (error){
        printf("Error #%d allocating device memory with data.", error);
        exit(1);
    }
    for(int i = 0; i < globalAcc.size(); i++){
        if (globalAcc[i] == '|'){
            data[i] = '\0';
            continue;
        }
        data[i] = globalAcc[i];
    }
    return;
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
                if(seqs.size() >= MAX_SEQS) break;
            }
            if (acc != ""){
                acc += "|";
                seqs.push_back(acc);
                indexes_aux.push_back(indexCounter);
                indexCounter += acc.size();
                globalAcc += acc;
                acc = "";
                indexes_aux.push_back(indexCounter);
                if(seqs.size() >= MAX_SEQS) break;
            }
        }
    }
    int last_index = indexes_aux.size();
    numberOfSequenses = seqs.size();
    size_all_seqs = globalAcc.size()*sizeof(char);
    cudaError_t error;
    error = cudaMallocManaged(&data, size_all_seqs);
    if (error){
        printf("Error #%d allocating device memory with data.", error);
        exit(1);
    }
    for(int i = 0; i < globalAcc.size(); i++){
        if (globalAcc[i] == '|'){
            data[i] = '\0';
            continue;
        }
        data[i] = globalAcc[i];
    }
    return;
}

/*Versión 2: tarda menos pero utiliza más memoria*/
void sequentialKmerCount2(vector<string> &seqs, vector<string> &permutations , int k){
    string mers[4] = {"A","C","G","T"};
    long numberOfSequences = seqs.size();
    // |kmers| is at most 4**k = 4**3 = 64
    int max_combinations = pow(4,k);
    float distance;
    long sum;
    long minimum;
    long minLength;
    long i,j,p;
    long aux;
    int **countKmeres = new int*[numberOfSequences];
    // Getting distance of each kmere of each sequence
    for(i =  0; i < numberOfSequences; i++){
        countKmeres[i] = new int[max_combinations+1];
        permutationsCountAll(seqs[i], countKmeres[i], max_combinations, k);
    }
    for(i =  0; i < numberOfSequences - 1; i++){
        for(j = i + 1; j < numberOfSequences; j++){
            minLength = min(seqs[i].size() - 1, seqs[j].size() - 1);
            sum = 0;
            minimum = -1;
            aux =  getIdxTriangularMatrixRowMajorSeq(i +1 ,  (j - i), numberOfSequences);
            for(p = 1; p <= max_combinations; p++){
                minimum = min(countKmeres[i][p], countKmeres[j][p]);
                sum += minimum;
            }
            distance = 1 - (float) sum / (minLength - k + 1);
            distancesSequential[aux] = distance;
            //printf("Distance #%ld\t%f (i=%d, j=%d)\n", aux, distance, i, j );
            // distancesSequential[j][i] = distance;
        }
    }
    return;
}



void permutationsCountAll(string sequence, int * countResults, int max_combinations, int k){
    int sequence_len = sequence.size();
    string current_kmere;
    for(int i = 0; i < max_combinations + 1; i++)
        countResults[i] = 0;
    for(int i = 0; i < sequence_len - k ; i++){
        current_kmere = sequence.substr(i,k);
        // tomamos el índice 0 como error en caso de encontrar algún caracter fuera del alfabeto válido de entrada.
        countResults[permutationsMap[current_kmere]]++;
    }
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

long getIdxTriangularMatrixRowMajorSeq(long i, long j, long n){
    return (n * (i - 1) - (((i - 2) * (i - 1)) / 2)) + (j - i);
}