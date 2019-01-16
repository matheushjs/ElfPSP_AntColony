#ifndef _CONFIG_H
#define _CONFIG_H

#include <string>

/** \brief Class that reads the configuration file (e.g. configuration.yml) and stores the
 *    configuration parameters as internal variables.
 */
class Config {
	static const char *sFilename; /** Name of the configuration file. */

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
