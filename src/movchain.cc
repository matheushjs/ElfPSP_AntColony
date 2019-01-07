#include <cstdlib>

#include "movchain.h"

using std::vector;
using std::rand;

void MovChain::append(char movement){
	dVector.push_back(movement);
}

void MovChain::append_random(){
	int choice = rand() % 5;
	dVector.push_back(choice);
}

void MovChain::randomize(int idx){
	int choice = rand() % 5;
	dVector[idx] = choice;
}

std::ostream& operator<<(std::ostream& stream, const MovChain &chain){
	char arr[] = { 'F', 'L', 'R', 'U', 'D' };
	vector<char> transformed(chain.length());

	for(unsigned int i = 0; i < transformed.size(); i++){
		int numericChar = chain.vector()[i];
		transformed[i] = arr[numericChar];
	}
	stream << transformed.data();
	
	return stream;
}
