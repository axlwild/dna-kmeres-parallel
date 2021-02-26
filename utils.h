//
// Created by Axel Cervantes on 20/01/21.
//

#ifndef EXAMEN_UTILS_H
#define EXAMEN_UTILS_H

#endif //EXAMEN_UTILS_H
#include <iostream>
#include <string>
#include <vector>
#include <map>



/**
 * @param alphabet: string with all the alphabet letters.
 * @param length: length of permutated strings.
 * @param permutations: results buffer.
 */
void permutation(const char * alphabet, int length, char** permutations)
{
    int sizeAlphabet = strlen(alphabet);
    int permsSize    = pow(sizeAlphabet, length);
    //permutations = (char**) malloc(permsSize * sizeof(char*));
    //for(int i = 0; i < permsSize; i++)
    //    permutations[i] = (char*) malloc(length*sizeof(char));
    int * letterIdx = (int*) calloc(length, sizeof(int));

    for(int i = 0; i < permsSize; i++){
        for (int currChar = length-1; currChar >= 0; currChar--){
            permutations[i][currChar] = alphabet[letterIdx[currChar]];
        }
        //std::cout << permutations[i] << std::endl;
        bool update = true;
        for (int j = 0; j < length; j++){
            if (update) {
                if(letterIdx[j] < sizeAlphabet - 1){
                    update = false;
                    letterIdx[j]++;
                }
                else{
                    letterIdx[j] = 0;
                    update = true;
                }
            }
        }
    }
    return;
}

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

/**
 * @param string
 * @return
 * Consider the |alphabet| = 4. alphabet = {A,C,G,T}
 * Each character requires 2 bits.
 * As in each byte is going to fit 4 characters, we define a map with 4 characters as well.
 * Example:
 * AAAB = 00 00 00 01
 */
//char* stringToBitsRepresentation(char* string){
//    std::map<std::string, char> bitMap;
//    bitMap["AAAA"] = 0;
//    bitMap["AAAC"] = 1;
//    bitMap["AAAG"] = 2;
//    bitMap["AAAT"] = 3;
//    bitMap["AACA"] = 4;
//    bitMap["AACC"] = 5;
//    bitMap["AACG"] = 6;
//    bitMap["AACT"] = 7;
//    bitMap["AAGA"] = 8;
//    bitMap["AAGC"] = 9;
//    bitMap["AAGG"] = 10;
//    bitMap["AAGT"] = 11;
//    bitMap["AATA"] = 12;
//    bitMap["AATC"] = 13;
//    bitMap["AATG"] = 14;
//    bitMap["AATT"] = 15;
//}
