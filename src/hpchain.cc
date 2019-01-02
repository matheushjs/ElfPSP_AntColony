
#include "hpchain.h"

using std::string;

HPChain::HPChain()                  : dChain(""){}
HPChain::HPChain(const char *chain) : dChain(chain){}
HPChain::HPChain(string chain)      : dChain(chain){}

string &HPChain::get_chain(){
	return dChain;
}

bool HPChain::validate(){
	for(char c: dChain)
		if(c != 'H' && c != 'P')
			return false;
	return true;
}
