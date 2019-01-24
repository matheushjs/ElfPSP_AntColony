#include <iostream>
#include <fstream>
#include <ctime>

#include "hpchain.h"
#include "config.h"
#include "vec3.h"
#include "acopredictor.h"
#include "acosolution.h"

using namespace std;

inline ostream& operator<<(ostream& stream, const vector<vec3<int>> vec){
	for(vec3<int> v: vec) stream << v << " ";
	return stream;
}

inline ostream& operator<<(ostream& stream, const vector<char> directions){
	for(char d: directions) stream << ACOSolution::DIR_TO_CHAR(d);
	return stream;
}

int main(int argc, char *argv[]){
	Config conf;
	HPChain hpchain(conf.hp_chain());

	if(!hpchain.validate()){
		cerr << "Invallid HP Chain. Received: " << hpchain.get_chain() << "\n";
		std::exit(1);
	}

	ACOPredictor predictor(
		conf.hp_chain(),
		conf.cycles(),
		conf.n_ants(),
		conf.aco_alpha(),
		conf.aco_beta(),
		conf.aco_evaporation(),
		conf.random_seed()
	);

	ACOSolution result = predictor.predict();

	ofstream outFile = ofstream(conf.filename(), ios::trunc | ios::out);
	outFile << "{\n";
	outFile << "\t\"coords\": " << "\"" << result.vector() << "\",\n";
	outFile << "\t\"hpchain\": " << "\"" << hpchain.get_chain() << "\",\n";
	outFile << "\t\"directions\": " << "\"" << result.directions() << "\"\n";
	outFile << "}";

	return 0;
}
