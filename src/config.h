#ifndef _CONFIG_H
#define _CONFIG_H

/** \file config.h */

#include <string>

/** Reads the configuration file (eg configuration.yml) and stores the
 *    configuration parameters as internal variables.
 */
class Config {
	static const char *sFilename; /**< Name of the configuration file. */

	std::string dHPChain;
	int         dCycles;
	int         dNAnts;
	double      dAcoAlpha;
	double      dAcoBeta;
	double      dAcoEvaporation;
	int         dRandSeed;

public:
	/** The configuration file is read upon construction. */
	Config();

	/** HP chain of the protein to be predicted. */
	const std::string&
	       hp_chain()    const { return dHPChain; }

	/** Number of cycles that the ant colony must perform. */
	int    cycles()      const { return dCycles; }

	/** Number of ants in the colony. */
	int    n_ants()      const { return dNAnts; }

	/** Parameter alpha of the ACO probability calculation formula. */
	double aco_alpha()   const { return dAcoAlpha; }

	/** Parameter beta of the ACO probability calculation formula. */
	double aco_beta()    const { return dAcoBeta; }

	/** Evaporation rate of pheromones. */
	double aco_evaporation() const { return dAcoEvaporation; }

	/** Random seed defined by the user.
	 *
	 * If the user-provided seed is negative, that means we should take a random seed, such as `time(NULL).` */
	int    random_seed() const { return dRandSeed; }
};

#endif
