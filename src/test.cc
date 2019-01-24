#include <iostream>
#include <vector>
#include <ctime>

#include "hpchain.h"
#include "config.h"
#include "vec3.h"
#include "acosolution.h"

using namespace std;

void test_aco_solution(){
	cout << "\n== ACOSolution ==\n";

	vector<vec3<int>> prev({
			{1,0,0},{-1,0,0},
			{0,1,0},{0,-1,0}
		});
	vector<char> dir({
			ACOSolution::UP,
			ACOSolution::DOWN,
			ACOSolution::LEFT,
			ACOSolution::RIGHT,
			ACOSolution::FRONT,
		});
	vector<string> str({ "UP", "DOWN", "LEFT", "RIGHT", "FRONT" });

	for(vec3<int> &p: prev){
		cout << "\t" << p << " <--- Previous Direction \n";
		for(unsigned int i = 0; i < dir.size(); i++){
			cout << "\t\t" << str[i] << "\t"
			     << ACOSolution::DIRECTION_VECTOR(p, dir[i]) << "\n";
		}
	}
}

int main(int argc, char *argv[]){
	cout << "== HPChain ==\n"
	     << "\t1110*\n\t"
	     << HPChain("HHPP").validate()
	     << HPChain(string("HHHPPP")).validate()
	     << HPChain().validate()
	     << HPChain(string("ouch")).validate()
	     << "\n";

	Config conf;
	cout << "\n== Config ==\n\t";
	cout << conf.hp_chain() << "\n\t";
	cout << conf.cycles() << "\n\t";
	cout << conf.n_ants() << "\n\t";
	cout << conf.aco_alpha() << "\n\t";
	cout << conf.aco_beta() << "\n\t";
	cout << conf.random_seed() << "\n";	

	vec3<int> p1(10, 0, 0);
	vec3<int> p2({0, 0, 10});
	cout << "\n== Vec3 ==\n\t";
	cout << p1 + p2 << "\n\t";
	cout << p1 - p2 << "\n\t";
	cout << -p1 << "\n\t";
	cout << p1.dot(p2) << "\n\t";
	cout << p1.dot({1,1,1}) << "\n\t";
	cout << p1.dot() << "\n\t";
	cout << (p1-p2).norm2() << "\n";

	test_aco_solution();

	return 0;
}
