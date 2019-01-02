#include <iostream>

#include "hpchain.h"

using namespace std;

int main(int argc, char *argv[]){
	cout  << HPChain("HHPP").validate() << "\n";
	cout  << HPChain(string("HHHPPP")).validate() << "\n";
	cout  << HPChain().validate() << "\n";
	cout  << HPChain(string("ouch")).validate() << "\n";

	return 0;
}
