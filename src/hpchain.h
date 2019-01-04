#pragma once

#include <string>

class HPChain {
	std::string dChain;

public:
	HPChain()                  : dChain(""){}
	HPChain(const char *chain) : dChain(chain){}
	HPChain(std::string chain) : dChain(chain){}

	std::string &get_chain(){ return dChain; }

	bool validate();
};
