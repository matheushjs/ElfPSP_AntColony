
#include "hpchain.h"

using std::string;

bool HPChain::validate(){
	for(char c: dChain)
		if(c != 'H' && c != 'P')
			return false;
	return true;
}
