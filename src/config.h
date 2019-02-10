#ifndef _CONFIG_H
#define _CONFIG_H

/** \file config.h */

#include <string>

/** Reads the configuration file (eg configuration.yml) and stores the
 *    configuration parameters as internal variables.
 */
class Config {
	static const char *sFilename; /**< Name of the configuration file. */

	/** @{ */
	/** See getters. */
	std::string dHPChain;
	int         dCycles;
	int         dNAnts;
	double      dAcoAlpha;
	double      dAcoBeta;
	double      dAcoEvaporation;
	int         dLSFrequency;
	int         dExchangedAnts;
	int         dRandSeed;
	std::string dFilename;
	/** */

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

	/** Local search frequency. */
	int    ls_frequency() const { return dLSFrequency; }

	/** Number of ants that MPI nodes should exchange at every cycle. */
	int    exchanged_ants() const { return dExchangedAnts; }

	/** Random seed defined by the user.
	 *
	 * If the user-provided seed is negative, that means we should take a random seed, such as `time(NULL).` */
	int    random_seed() const { return dRandSeed; }

	/** Output filename for structural information of the protein. */
	const std::string &filename() const { return dFilename; }

	void set_hp_chain(std::string chain) { dHPChain = chain; }

	void set_cycles(int cycles) { dCycles = cycles; }

	void set_nants(int nants) { dNAnts = nants; }
};

#endif
