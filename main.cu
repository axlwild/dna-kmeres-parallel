#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <math.h>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
int size;
int permutationsSize;
int countArraySize;
// Device
vector<string> * d_seqs;
vector<string> * d_permutations;
int *d_count;

char** seqs_ptr;
char** perms_ptr;

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
vector<string> permutationsList (perms, end(perms));

float ** distancesSequential;
float ** distancesParallel;


int main(int argc, char **argv) {
    int len = strlen("ACGT") ;
    //char permutations[len];
    int numberOfSequenses = 0;
    //getPermutations(chars, permutations, len - 1, 0);
    /*for (int i = 0; i < permutationsList.size() ; i++){
        cout << permutationsList.at(i) << endl;
    }
    */
    // absolute path of the input data
    string file = "/home/acervantes/plants.fasta";
    importSeqs(file);
    // Reserving memory for resultsf
    numberOfSequenses = seqs.size();
    distancesSequential = (float**) malloc(sizeof(float*) * numberOfSequenses);
    distancesParallel   = (float**) malloc(sizeof(float*) * numberOfSequenses);
    for(int i = 0; i < numberOfSequenses; i++){
        distancesSequential[i] = (float*) malloc(numberOfSequenses*sizeof(float));
        distancesParallel[i]   = (float*) malloc(numberOfSequenses*sizeof(float));
    }
    for (int i = 0; i < numberOfSequenses ; i++){
        for (int j = 0; j < numberOfSequenses ; j++) {
            distancesSequential[i][j] = 0;
            distancesParallel[i][j] = 0;
        }
    }

    sequentialKmerCount(seqs, permutationsList, 3);
    for (int i = 0; i < numberOfSequenses ; i++){
        for (int j = 0; j < numberOfSequenses ; j++) {
            cout << distancesSequential[i][j] << "\t";
        }
        cout << endl;
    }
    free(distancesSequential);
    free(distancesParallel);
    return 0;
}

void importSeqs(string inputFile){
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