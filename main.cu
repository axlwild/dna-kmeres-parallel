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

using namespace std;

int          numberOfSequenses = 0;
char *       all_seqs;
unsigned int size_all_seqs = 0;


// Method definition
void importSeqs(string inputFile);
void printSeqs();
void getPermutations(char *str, char* permutations, int last, int index);
int permutationsCount(string permutation, string sequence, int k);
void sequentialKmerCount(vector<string> &seqs, vector<string> &permutations , int k);
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
int * h_sums;

// number of permutations of RNA K_meres and k-value
__constant__ char c_perms[64][4] = {
    "AAA", "AAC", "AAT","AAG",
    "ACA", "ACC", "ACT","ACG",
    "ATA", "ATC", "ATT","ATG",
    "AGA", "AGC", "AGT","AGG",

    "CAA", "CAC", "CAT","CAG",
    "CCA", "CCC", "CCT","CCG",
    "CTA", "CTC", "CTT","CTG",
    "CGA", "CGC", "CGT","CGG",

    "GAA", "GAC", "GAT","GAG",
    "GCA", "GCC", "GCT","GCG",
    "GTA", "GTC", "GTT","GTG",
    "GGA", "GGC", "GGT","GGG",

    "TAA", "TAC", "TAT","TAG",
    "TCA", "TCC", "TCT","TCG",
    "TTA", "TTC", "TTT","TTG",
    "TGA", "TGC", "TGT","TGG",
};
__constant__ int  c_size ;
char perms[64][4] = {
                        "AAA", "AAC", "AAT","AAG",
                        "ACA", "ACC", "ACT","ACG",
                        "ATA", "ATC", "ATT","ATG",
                        "AGA", "AGC", "AGT","AGG",

                        "CAA", "CAC", "CAT","CAG",
                        "CCA", "CCC", "CCT","CCG",
                        "CTA", "CTC", "CTT","CTG",
                        "CGA", "CGC", "CGT","CGG",

                        "GAA", "GAC", "GAT","GAG",
                        "GCA", "GCC", "GCT","GCG",
                        "GTA", "GTC", "GTT","GTG",
                        "GGA", "GGC", "GGT","GGG",

                        "TAA", "TAC", "TAT","TAG",
                        "TCA", "TCC", "TCT","TCG",
                        "TTA", "TTC", "TTT","TTG",
                        "TGA", "TGC", "TGT","TGG",
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

__global__ void parallelKDist(char *data, int *indices, float*distances, unsigned num_seqs, int *suma){
    // each block is comparing a sample with others
    //int idx = threadIdx.x+blockDim.x*blockIdx.x;
    __shared__ int entry;
    // Each thread count all coincidences of a k-mere combination.
    int k_mere = threadIdx.x;
    //printf("outside blockid: %d \n", blockIdx.x);
    if (blockIdx.x < (int)sizeof(indices) && threadIdx.x <= 64){
        entry = blockIdx.x;
        __syncthreads();
        // Fase uno: sumamos todos los valores de la suma de los k-meros de cada entrada.
        // Cada bloque se encarga de calcular la suma de cada entrada
        // Cada hilo se encarga de sumar cada permutación.
        // en suma guardamos las apariciones de las 64 posibles combinaciones.
        // Si cada hilo se encargara de cada k-mero no habría por qué hacer la operación atómica.
        const char *currentKmere = c_perms[k_mere];
        // Entonces cada hilo tendría que iterar toda la muestra solo una vez para calcular la suma.
        int entryLength = indices[entry + 1] -  indices[entry];

        // entonces iteramos por cada letra de la entrada hasta la N-k (los índices).
        // Podríamos guardar los índices en memoria constante para agilizar la lectura...
        bool is_same_kmere = true;
        char * sequence = data+indices[entry];
        /*if(blockIdx.x < 5 && threadIdx.x == 0){
            printf("Block #%d\tIndex: %d\tEntry:%d\nAll data inside: %s\n", blockIdx.x , indices[entry], entry, sequence);
            printf("Entry %d, EntryLength: %d, idx: %d\n", entry, entryLength, indices[entry]);
            printf("indices: %i\n", (int)indices[2]);
            printf("#sequence block %d: %s\n", blockIdx.x, sequence);
        }*/
        char currentSubstringFromSample[4];
        if(blockIdx.x == 0 && threadIdx.x == 0){
            printf("current string: %s\n", sequence);
        }
        for (int i = 0; i < entryLength-3; i++){
            memcpy( currentSubstringFromSample, &sequence[i], 3 );
            currentSubstringFromSample[3] = '\0';
            is_same_kmere = true;
            for(int j = 0; j < 3; j++){
                if (sequence[j] == currentKmere[j]){
                    continue;
                }
                is_same_kmere = false;
                break;
            }
            if(is_same_kmere){
                suma[entry*blockDim.x+threadIdx.x] += 1;
            }
        }
        // Fase dos.
        // Sumamos los mínimos de las cadenas comparadas.
        __syncthreads();
        int nextEntryLength;
        for(int j = entry + 1; j < num_seqs - 1; j++){
            nextEntryLength = indices[j + 1] -  indices[j];
            distances[entry*threadIdx.x+j] = 1 - min(suma[entry*blockDim.x+threadIdx.x],suma[j*blockDim.x+threadIdx.x])/ (min(nextEntryLength, entryLength) -3 + 1);
        }
    }
    else{

    }


}

int main(int argc, char **argv) {

    //    const char* prueba = "Esto\00Es\00Una\00Prueba";
    //    std::cout << prueba;
    //    return;

    //char permutations[len];
    cudaError_t error;
    // absolute path of the input data
    string file = "/home/acervantes/kmerDist/plants.fasta";
    //string file = "/home/acervantes/all_seqs.fasta";
    importSeqs(file);

    // Reserving memory for results
    numberOfSequenses = seqs.size();
    //printf("%d sequences founded", numberOfSequenses);
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
    doSequentialKmereDistance();

    // Device allocation
    int numResults = numberOfSequenses*numberOfSequenses*64;
    int sumsSize = sizeof(int)*numResults;
    h_sums = (int*) malloc(sumsSize);
    for (int i = 0; i < numResults; i++){
        h_sums[i] = -1;
    }
    cudaMalloc((void**)&d_sums, sumsSize);


    /* // defining a constant value is passed to the device directly.
    error = cudaMemcpyToSymbol(c_perms, &perms, 4*64 * sizeof(char) );
    if (error){
        printf("Errorsti : %d: %s\n", error, cudaGetErrorString(error));
    }
    */
    error = cudaMemcpyToSymbol(c_size, &permsSize, sizeof(int) );
    if (error){
        printf("Error %d: %s", error, cudaGetErrorString(error));
    }
    unsigned long int sizeDistances = numberOfSequenses*numberOfSequenses * sizeof(float);
    string data_aux = join(seqs, "\0");
    //int indexes[indexes_aux.size()];
    int *indexes = (int *) malloc((int)indexes_aux.size() * sizeof(int));
    for (int i = 0; i < indexes_aux.size(); i++){
        indexes[i] = indexes_aux[i];
    }
    int max_indexes = indexes_aux.size();
    int last_data_idx = sizeof(all_seqs) - 1;
    for (int i = 0; i < max_indexes - 1; i++){
        //printf("idx %d: %d\n", i, indexes[i]);
        int entryLength = (i != max_indexes -1)  ? indexes[i + 1] -  indexes[i] - 1 : sizeof(all_seqs) - (indexes[i]);
        const char * sequence = all_seqs+indexes[i] ;
        //printf("sequence size %d: %s\n", entryLength, sequence);
    }

    //std::copy(indexes_aux.begin(), indexes_aux.end(), indexes);
    //int size_seqs = sizeof(all_seqs) * sizeof(char);
    cudaMalloc((void **)&d_data, size_all_seqs);
    error = cudaMalloc((void **)&d_distances, sizeDistances);
    if (error){
        printf("Error al usar memoria con distancia %d ::", error);
        cout << sizeDistances << endl;
        return 0;
    }
    int indexesBytesSize = sizeof(indexes)*sizeof(int);
    error = cudaMalloc((void **)&d_indices, indexesBytesSize);
    if (error){
        printf("Error malloc %d", error);
    }
    float *h_distances;
    h_distances =(float*) malloc(sizeDistances);
    error = cudaMemcpy(d_data, all_seqs, size_all_seqs, cudaMemcpyHostToDevice);
    if (error){
        printf("Error copying data from host %d", error);
    }
    error = cudaMemcpy(d_sums, h_sums, sumsSize, cudaMemcpyHostToDevice);
    if (error){
        printf("Error copying data from host %d", error);
    }
    error = cudaMemcpy(d_indices, indexes, indexesBytesSize, cudaMemcpyHostToDevice);
    if (error){
        printf("Error copying data from device %d\n", error);
    }
    //int blocks = ceil(seqs.size() / 1024) + 1;
    //int threads = 1024;
    int blocks = 10;
    int threads = 64;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    // Launch kernel
    printf("Running %d blocks and %d threads\n", blocks, threads);
    parallelKDist<<<blocks, threads>>>(d_data, d_indices, d_distances, numberOfSequenses, d_sums);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float parallelTimer = 0;
    cudaEventElapsedTime(&parallelTimer, start, stop);
    cout<< "Elapsed parallel timer: " << parallelTimer << " ms, " << parallelTimer / 1000 << " secs" <<endl;
    cudaMemcpy(h_distances, d_distances, sizeDistances, cudaMemcpyDeviceToHost);

    for(int i = 0, idx = 0; i < numberOfSequenses; i++){
        for(int j = 0; j < numberOfSequenses; j++){
            printf("%f ", h_distances[idx++]);
        }
        printf("\n");
    }
    free(distancesSequential);
    //free(distancesParallel);
    cudaFree(d_distances);
    cudaFree(d_indices);
    cudaFree(d_data);
    cudaFree(d_sums);
    return 0;
}

void doSequentialKmereDistance(){
    // results files
    FILE *f_seq_res = fopen("/home/acervantes/kmerDist/sequential_results.csv", "w");
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
                if(line.empty() ){
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

/*
void kmerDistance(int k){

}
 */