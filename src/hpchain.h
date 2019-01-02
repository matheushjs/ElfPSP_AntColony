#pragma once

#include <string>

class HPChain {
	const std::string dChain;

public:
	HPChain();
	HPChain(const char *chain);
	HPChain(std::string chain);

	const std::string get_chain();

	bool validate();
};
