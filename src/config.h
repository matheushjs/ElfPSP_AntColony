#ifndef _CONFIG_H
#define _CONFIG_H

#include <string>

class Config {
	static const char *sFilename;
	static const int sParamCount = 2;

	std::string dHPChain;
	int dRandSeed;

public:
	Config();

	std::string &hp_chain()    { return dHPChain; }
	int         random_seed()  { return dRandSeed; }
};

#endif
