#pragma once

#include <string>

class HPChain {
	std::string dChain;

public:
	HPChain();
	HPChain(const char *chain);
	HPChain(std::string chain);

	std::string &get_chain();

	bool validate();
};
