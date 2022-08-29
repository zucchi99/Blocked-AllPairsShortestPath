#include <stdio.h>
#include <string>

int main() {
    
    std::string param = "--test=prova";
    int index_of_equal = param.find('=');
    std::string param_name = param.substr(0, param.size() - index_of_equal + 1);
    std::string param_val = param.substr(index_of_equal + 1);
    printf("%s\n", param_name.c_str());
    printf("%s\n", param_val.c_str());
}