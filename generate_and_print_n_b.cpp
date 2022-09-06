#include "include/generate_n_b_couples.hpp"
#include "include/macros.hpp"

#include "time.h"

#include <algorithm>

void split_word(std::vector<int> &vec, std::string word, char del);

int main(int argc, char** argv) {

    std::vector<std::string> str_args(argv, argv + argc);

    int seed = time(NULL);
    srand(seed);

    // define parameters
    double to_multiply = 1.0;
    int to_sum = 80;
    int min_input_size = 80;
    int max_input_size = 480;

    //if blocking factors are calculated randomly
    int min_blocking_factor = 2;
    int max_num_of_b_per_n = 5;
    int max_num_tests = 50;
    //if data are given from argv

    std::vector<int> n; 
    std::vector<int> B; 
    if (argc > 2) {
        split_word(n, str_args[1], ',');
        split_word(B, str_args[2], ',');
    }

    std::vector<std::pair<int, int>> list_of_all_n_b;

    // print parameters
    printf("to_multiply:\t\t%f\n", to_multiply);
    printf("to_sum:\t\t\t%d\n", to_sum);
    printf("seed:\t\t\t%d\n", seed);
    printf("min_input_size:\t\t%d\n", min_input_size);
    printf("max_input_size:\t\t%d\n", max_input_size);

    if (argc == 1) {
        // generate blocking factors randomly
        printf("min_blocking_factor:\t%d\n", min_blocking_factor);
        printf("max_num_of_b_per_n:\t%d\n", max_num_of_b_per_n);
        printf("max_num_tests:\t\t%d\n", max_num_tests);
        
        // generate list randomly
        list_of_all_n_b = generate_list_of_all_n_b(min_input_size, max_input_size, max_num_of_b_per_n, to_multiply, to_sum, min_blocking_factor, max_num_tests, seed);

    } else {
        // read matrix size and blocking factors from argv
        for (int i = 0; i < n.size(); i++) {
            for (int j = 0; j < B.size(); j++) {
                if (n[i] % B[j] == 0) {
                    //printf("%d,%d\n", n[i], B[j]);
                    list_of_all_n_b.push_back(std::make_pair(n[i], B[j]));
                }
            }
        }
    }


    // define output file name
    std::string out_filename = "csv/list_of_n_b.csv";

    // print list to file
    print_list_to_file(list_of_all_n_b, out_filename);

    // print file name
    printf("output file:\t\t%s\n", out_filename.c_str());

    return 0;
    
}

void split_word(std::vector<int> &vec, std::string word, char del) {
    int size = 0;
    size = std::count(word.begin(), word.end(), del) + 1;
    for (int i = 0; i < size; i++) {
        int first_index = word.find(del);
        int val = std::stoi(word.substr(0, first_index));
        vec.push_back(val);
        word = word.substr(first_index + 1);
    }
}