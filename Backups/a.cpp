#include <iostream>
#include <fstream>
#include <string>
using namespace std;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    // Placeholder: generate a trivial CSV in required format
    // Columns: id,x,y,deg with 's' prefixes
    ofstream out("submission.csv");
    out << "id,x,y,deg\n";
    // Example single group 10 with two trees; replace with real solver output
    out << "010_0,s0,s0,s0\n";
    out << "010_1,s0.1,s0.1,s0\n";
    out.close();
    return 0;
}
