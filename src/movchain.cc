#include <string>
#include <cstdlib>

#include "movchain.h"

using std::string;
using std::rand;

void MovChain::append(char movement){
	dMovChain.push_back(movement);
}

void MovChain::append_random(){
	int choice = rand() % 5;
	dMovChain.push_back(choice);
}

void MovChain::randomize(int idx){
	int choice = rand() % 5;
	dMovChain[idx] = choice;
}
