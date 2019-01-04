#include <iostream>
#include <cstdlib>
#include <ctime>

#include "hpchain.h"
#include "config.h"

using namespace std;

int main(int argc, char *argv[]){
	cout  << HPChain("HHPP").validate() << "\n";
	cout  << HPChain(string("HHHPPP")).validate() << "\n";
	cout  << HPChain().validate() << "\n";
	cout  << HPChain(string("ouch")).validate() << "\n";

	Config conf;

	cout << conf.hp_chain() << "\n";
	cout << conf.random_seed() << "\n";	

	if(conf.random_seed() < 0){
		srand(time(NULL));
	} else {
		srand(conf.random_seed());
	}

	return 0;
}
