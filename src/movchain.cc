#include <string>
#include <cstdlib>

#include "movchain.h"

using std::string;
using std::rand;

// const char MovChain::sMovHash[] = "FLRUD";
const char MovChain::sMovHash[] = {
	MovChain::FRONT,
	MovChain::LEFT,
	MovChain::RIGHT,
	MovChain::UP,
	MovChain::DOWN
};

void MovChain::append(char movement){
	dMovChain.push_back(movement);
}

void MovChain::append_random(){
	int choice = rand() % 5;
	char chr = sMovHash[choice];
	dMovChain.push_back(chr);
}

void MovChain::randomize(int idx){
	int choice = rand() % 5;
	char chr = sMovHash[choice];
	dMovChain[idx] = chr;
}
