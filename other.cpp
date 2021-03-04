//
// Created by Axel Cervantes on 04/03/21.
//

int permutationsCount(string permutation, string sequence, int k);
int permutationsCount(string permutation, string sequence, int k){
    int sequence_len = sequence.size();
    int counter = 0;
    string current_kmere;
    for(int i = 0; i < sequence_len - k; i++){
        current_kmere = sequence.substr(i,k);
        if (permutation.compare(current_kmere) == 0){
            counter++;
        }
    }
    return counter;
}

/*Versión secuencial 1: tarda más pero utiliza menos memoria*/
void sequentialKmerCount(vector<string> &seqs, vector<string> &permutations , int k);
void sequentialKmerCount(vector<string> &seqs, vector<string> &permutations , int k){
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
    int countKmereSi[max_combinations+1] = {0};
    int countKmereSj[max_combinations+1] = {0};
    // Comparing example Ri with R(i+1) until Rn
    for(i =  0; i < numberOfSequences - 1; i++){
        permutationsCountAll(seqs[i], countKmereSi, max_combinations, k);
        for(j = i + 1; j < numberOfSequences; j++){
            //if(i >= j)
            //    continue;
            // iterating over permutations (distance of Ri an Rj).
            //restamos uno por el | auxiliar que agregamos en todo al final
            minLength = min(seqs[i].size() - 1, seqs[j].size() - 1);
            sum = 0;
            minimum = -1;
            aux =  getIdxTriangularMatrixRowMajorSeq(i +1 ,  (j - i), numberOfSequences);
            // obtiene el vector de la cuenta de todas las permutaciones.
            permutationsCountAll(seqs[j], countKmereSj, max_combinations, k);
            for(p = 1; p <= max_combinations; p++){
                minimum = min(countKmereSi[p], countKmereSj[p]);
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