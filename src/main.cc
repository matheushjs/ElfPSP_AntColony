#include <iostream>
#include <cstdlib>
#include <ctime>

#include "hpchain.h"
#include "config.h"
#include "vec3.h"

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

	vec3<int> p1(1, 2, 3);
	vec3<int> p2(1, 2, 3);
	vec3<int> p3 = p1 + p2;

	cout << p3.y << "\n";

	return 0;
}
