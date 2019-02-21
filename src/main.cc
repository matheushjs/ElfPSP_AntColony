#include <vector>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>

/** \file main.cc */

#include "hpchain.h"
#include "config.h"
#include "vec3.h"
#include "acopredictor.h"
#include "acosolution.h"

using namespace std;

/** Prints a vector of integer coordinates. */
inline ostream& operator<<(ostream& stream, const vector<vec3<int>> vec){
	for(vec3<int> v: vec) stream << v << " ";
	return stream;
}

/** Prints a vector of relative directions. */
inline ostream& operator<<(ostream& stream, const vector<char> directions){
	for(char d: directions) stream << ACOSolution::DIR_TO_CHAR(d);
	return stream;
}

/** Main function. */
int main(int argc, char *argv[]){
	/* First parse command line arguments. */
	vector<string> args(argv, argv+argc);

	string argHPChain = "";
	int argNumCycles  = -1;
	int argNumAnts    = -1;

	/* Check arguments */
	for(unsigned i = 1; i < args.size(); i++){
		if(args[i] == "-h"){
			cout << "Usage: " << args[0] << " [-h] [-s hp_sequence] [-c num_cycles] [-a num_ants]\n"
				"\n"
				"    -h       displays this help message\n"
				"    -s arg   the HP sequence of the protein to predict\n"
				"    -c arg   number of cycles that the ant colony should perform\n"
				"    -a arg   number of ants in the ant colony\n";
			return 0;
		} else if(args[i-1] == "-s"){
			argHPChain = args[i];
			i++;
		} else if(args[i-1] == "-c"){
			argNumCycles = atoi(args[i].c_str());
			i++;
			if(argNumCycles <= 0){
				cerr << "Invalid number of cycles given!\n";
				exit(1);
			}
		} else if(args[i-1] == "-a"){
			argNumAnts = atoi(args[i].c_str());
			i++;
			if(argNumAnts <= 0){
				cerr << "Invalid number of ants given!\n";
				exit(2);
			}
		}
	}

	// Read configurations from configuration file
	Config conf;

	// Change configurations if command-line arguments were given
	if(argHPChain != "")
		conf.set_hp_chain(argHPChain);
	
	if(argNumCycles != -1)
		conf.set_cycles(argNumCycles);

	if(argNumAnts != -1)
		conf.set_nants(argNumAnts);

	// Validate hp chain
	HPChain hpchain(conf.hp_chain());

	if(!hpchain.validate()){
		cerr << "Invallid HP Chain. Received: " << hpchain.get_chain() << "\n";
		std::exit(2);
	}

	ACOPredictor predictor(conf);

	struct ACOPredictor::Results result = predictor.predict();

	ofstream outFile = ofstream(conf.filename(), ios::trunc | ios::out);
	outFile << "{\n";
	outFile << "\t\"coords\": " << "\"" << result.solution.vector() << "\",\n";
	outFile << "\t\"hpchain\": " << "\"" << hpchain.get_chain() << "\",\n";
	outFile << "\t\"directions\": " << "\"" << result.solution.directions() << "\"\n";
	outFile << "}";

	cout << "Contacts: " << result.contacts << "\n";

	return 0;
}
