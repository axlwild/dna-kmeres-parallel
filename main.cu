#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <math.h>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <sstream>
#include <typeinfo>
#include "cuda.h"

using namespace std;
// Method definition
void importSeqs(string inputFile);
void printSeqs();
void getPermutations(char *str, char* permutations, int last, int index);
int permutationsCount(string permutation, string sequence, int k);
void sequentialKmerCount(vector<string> &seqs, vector<string> &permutations , int k);
// Vectors to store ids and seqs
vector<string> ids;
vector<string> seqs;
vector<int> indexes_aux;

// Device variables.
char * d_data; // all the strings.
unsigned int * d_indices;
float * d_distances;
const char * data;

// coincidences of k-mer on each input
int * d_sums;
int * h_sums;

__constant__ const char* c_perms[] = {"AAA", "AAC", "AAT","AAG",
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
} ;
__constant__ int  c_size ;
const char * perms[] = {"AAA", "AAC", "AAT","AAG",
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

__global__ void parallelKDist(char *data, unsigned int *indices, float*distances, unsigned num_strings, int *suma){
    // each block is comparing a sample with others
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < num_strings && blockIdx.x < sizeof(indices) && threadIdx.x < 64){
        // Fase uno: sumamos todos los valores de la suma de los k-meros de cada entrada.
        // Cada bloque se encarga de calcular la suma de cada entrada
        int entry = blockIdx.x;
        // Cada hilo se encarga de sumar cada permutación.
        // en suma guardamos las apariciones de las 64 posibles combinaciones.
        // Si cada hilo se encargara de cada k-mero no habría por qué hacer la operación atómica.
        const char *currentKmere = c_perms[threadIdx.x];
        // Entonces cada hilo tendría que iterar toda la muestra solo una vez para calcular la suma.
        int entryLength = sizeof(data+indices[entry])/sizeof(char *);
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
        for(int j = entry + 1; j < num_strings; j++){
            nextEntryLength = sizeof(data+indices[j])/sizeof(char *);
            distances[entry*threadIdx.x+j] = 1 - min(suma[entry*blockDim.x+threadIdx.x],suma[j*blockDim.x+threadIdx.x])
                    / (min(nextEntryLength, entryLength) -3 + 1);
        }
    }


}

int main(int argc, char **argv) {
    //char permutations[len];
    int error;
    int numberOfSequenses = 0;
    //getPermutations(chars, permutations, len - 1, 0);
    /*
     * int len = strlen("ACGT") ;
     * for (int i = 0; i < permutationsList.size() ; i++){
        cout << permutationsList.at(i) << endl;
    }
    */
    // absolute path of the input data
    //string file = "/home/acervantes/plants.fasta";
    string file = "/home/acervantes/all_seqs.fasta";
    importSeqs(file);
    // Reserving memory for resultsf
    numberOfSequenses = seqs.size();
    distancesSequential = (float**) malloc(sizeof(float*) * numberOfSequenses);
    //distancesParallel   = (float**) malloc(sizeof(float*) * numberOfSequenses);
    for(int i = 0; i < numberOfSequenses; i++){
        distancesSequential[i] = (float*) malloc(numberOfSequenses*sizeof(float));
        //distancesParallel[i]   = (float*) malloc(numberOfSequenses*sizeof(float));
    }
    for (int i = 0; i < numberOfSequenses ; i++){
        for (int j = 0; j < numberOfSequenses ; j++) {
            distancesSequential[i][j] = 0;
            //distancesParallel[i][j] = 0;
        }
    }
    /*
    sequentialKmerCount(seqs, permutationsList, 3);
    for (int i = 0; i < numberOfSequenses ; i++){
        for (int j = 0; j < numberOfSequenses ; j++) {
            cout << distancesSequential[i][j] << "\t";
        }
        cout << endl;
    }
     */

    // Device allocation
    int sumsSize = sizeof(int)*numberOfSequenses*64;
    h_sums = (int*) malloc(sumsSize);
    cudaMalloc((void**)&d_sums, sumsSize);


    const char * c_perms[] = {"AAA", "AAC", "AAT","AAG",
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
    error = cudaMemcpyToSymbol(c_perms, perms, 64 * sizeof(char*) );
    if (error){
        printf("Errorsti %d", error);
    }
    error = cudaMemcpyToSymbol(c_size, &permsSize, sizeof(int) );
    if (error){
        printf("Errorsy %d", error);
    }

    int sizeDistances = numberOfSequenses*numberOfSequenses * sizeof(float);
    string data_aux = join(seqs, "\0");
    data = data_aux.c_str();
    int indexes[indexes_aux.size()];
    //indexes = (int * ) malloc(indexes_aux.size() * sizeof(int));
    for (int i = 0; i < indexes_aux.size(); i++){
        indexes[i] = indexes_aux[i];
    }
    //std::copy(indexes_aux.begin(), indexes_aux.end(), indexes);

    cudaMalloc((void **)&d_data, data_aux.size());
    cudaMalloc((void **)&d_distances, sizeDistances);
    error = cudaMalloc((void **)&d_indices, sizeof(indexes));
    if (error){
        printf("Errorsse %d", error);
    }
    float *h_distances;
    h_distances =(float*) malloc(sizeDistances);

    error = cudaMemcpy(d_data, data, data_aux.size(), cudaMemcpyHostToDevice);
    if (error){
        printf("Error %d", error);
    }
    error = cudaMemcpy(d_indices, indexes, sizeof(indexes), cudaMemcpyHostToDevice);
    if (error){
        printf("Errorsa %d\n", error);
    }
    int blocks = ceil(seqs.size() / 1024) + 1;
    int threads = 1024;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    // Launch kernel
    parallelKDist<<<blocks, threads>>>(d_data, d_indices, d_distances, numberOfSequenses, d_sums);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float parallelTimer = 0;
    cudaEventElapsedTime(&parallelTimer, start, stop);
    cout<< "Elapsed parallel timer: " << parallelTimer << " ms, " << parallelTimer / 1000 << " secs" <<endl;
    cudaMemcpy(h_distances, d_distances, sizeDistances, cudaMemcpyDeviceToHost);
    free(distancesSequential);
    //free(distancesParallel);
    cudaFree(d_distances);
    cudaFree(d_indices);
    cudaFree(d_data);
    cudaFree(d_sums);
    return 0;
}

void importSeqs(string inputFile){
    int indexCounter = 0;
    ifstream input(inputFile);
    if (!input.good()) {
        std::cerr << "Error opening: " << inputFile << " . Check your file or pathh." << std::endl;
        exit(0);
    }

    string line;

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
        }
        else {
            if (newSeq) {
                seqs.push_back(line);
                indexes_aux.push_back(indexCounter);
                indexCounter += line.size();
                newSeq = false;
            }
            else
                line += line;
            // store seqs

        }
    }
}

void sequentialKmerCount(vector<string> &seqs, vector<string> &permutations , int k){
    string mers[4] = {"A","C","G","T"};
    int numberOfSequences = seqs.size();
    // |kmers| is at most 4**k = 4**3 = 64
    int max_combinations = pow(4,k);
    // Comparing example Ri with i+1 until Rn
    for(int i =  0; i < numberOfSequences - 1; i++){
        for(int j = i + 1; j < numberOfSequences; j++){
            if(i == j)
                continue;
            // iterating over permutations (distance of Ri an Rj).
            int minLength = min(seqs[i].size(), seqs[j].size());
            int sum = 0;
            float distance = -1.0f;
            for(int p = 0; p < max_combinations; p++){
                int minimum = min(
                    permutationsCount(permutationsList[p], seqs[i],k),
                    permutationsCount(permutationsList[p], seqs[j],k)
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