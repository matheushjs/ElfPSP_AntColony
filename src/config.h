#ifndef _CONFIG_H
#define _CONFIG_H

#include <string>

/** \brief Class that reads the configuration file (e.g. configuration.yml) and stores the
 *    configuration parameters as internal variables.
 */
class Config {
	static const char *sFilename; /** \brief Name of the configuration file. */

	std::string dHPChain;
	int         dCycles;
	int         dNAnts;
	double      dAcoAlpha;
	double      dAcoBeta;
	int         dRandSeed;

public:
	Config();

	/** \brief HP chain of the protein to be predicted. */
	const std::string&
	       hp_chain()    const { return dHPChain; }

	/** \brief Number of cycles that the ant colony must perform. */
	int    cycles()      const { return dCycles; }

	/** \brief Number of ants in the colony. */
	int    n_ants()      const { return dNAnts; }

	/** \brief Parameter alpha of the ACO probability calculation formula. */
	double aco_alpha()   const { return dAcoAlpha; }

	/** \brief Parameter beta of the ACO probability calculation formula. */
	double aco_beta()    const { return dAcoBeta; }

	/** \brief Random seed defined by the user.
	 *
	 * If the user-provided seed is negative, that means we should take a random seed, such as `time(NULL).` */
	int    random_seed() const { return dRandSeed; }
};

#endif
