#include <iostream>
#include <cstdlib>
#include <ctime>

#include "hpchain.h"
#include "movchain.h"
#include "config.h"
#include "vec3.h"

using namespace std;

int main(int argc, char *argv[]){
	cout  << HPChain("HHPP").validate() << "\n";
	cout  << HPChain(string("HHHPPP")).validate() << "\n";
	cout  << HPChain().validate() << "\n";
	cout  << HPChain(string("ouch")).validate() << "\n";

	MovChain movchain;
	for(int i = 0; i < 10; i++) movchain.append_random();
	cout << movchain << "\n";

	Config conf;

	cout << conf.hp_chain() << "\n";
	cout << conf.cycles() << "\n";
	cout << conf.random_seed() << "\n";	

	if(conf.random_seed() < 0){
		srand(time(NULL));
	} else {
		srand(conf.random_seed());
	}

	vec3<int> p1(10, 0, 0);
	vec3<int> p2({0, 0, 10});

	cout << p1 + p2 << "\n";
	cout << p1 - p2 << "\n";
	cout << -p1 << "\n";
	cout << p1.dot(p2) << "\n";
	cout << p1.dot({1,1,1}) << "\n";
	cout << p1.dot() << "\n";
	cout << (p1-p2).norm2() << "\n";

	return 0;
}
