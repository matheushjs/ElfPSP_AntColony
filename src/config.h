#ifndef _CONFIG_H
#define _CONFIG_H

#include <string>

class Config {
	static const char *sFilename;
	static const int sParamCount = 6;

	std::string dHPChain;
	int         dCycles;
	int         dNAnts;
	double      dAcoAlpha;
	double      dAcoBeta;
	int         dRandSeed;

public:
	Config();

	const std::string&
	       hp_chain()    const { return dHPChain; }
	int    cycles()      const { return dCycles; }
	int    n_ants()      const { return dNAnts; }
	double aco_alpha()   const { return dAcoAlpha; }
	double aco_beta()    const { return dAcoBeta; }
	int    random_seed() const { return dRandSeed; }
};

#endif
