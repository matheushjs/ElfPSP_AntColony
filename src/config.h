#ifndef _CONFIG_H
#define _CONFIG_H

#include <string>

class Config {
	static const char *sFilename;
	static const int sParamCount = 3;

	std::string dHPChain;
	int dCycles;
	int dRandSeed;

public:
	Config();

	const std::string&
	    hp_chain()    const { return dHPChain; }
	int cycles()      const { return dCycles; }
	int random_seed() const { return dRandSeed; }
};

#endif
