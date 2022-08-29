#include <stdio.h>
#include <string>

int main() {
    
    std::string param = "--output-file=csv/all_performances.csv";
    int index_of_equal = param.find('=');
    std::string param_name = param.substr(0, index_of_equal + 1);
    std::string param_val = param.substr(index_of_equal + 1);
    printf("%d\n", index_of_equal);
    printf("%s\n", param_name.c_str());
    printf("%s\n", param_val.c_str());
}