#include <iostream>
#include <fstream>
#include <string>
using namespace std;
int main(){
    // Placeholder second stage; in a real pipeline, refine a.cpp output.
    // For now, pass-through.
    ifstream in("submission.csv");
    ofstream out("submission_refined.csv");
    string line;
    while(getline(in,line)){
        out << line << "\n";
    }
    return 0;
}
