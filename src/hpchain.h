#pragma once

/** \file hpchain.h */

#include <string>

/** Encapsulates a string representing an HP chain. */
class HPChain {
	std::string dChain;

public:
	/** Creates an empty HP chain. */
	HPChain()                  : dChain(""){}

	/** Creates an HP chain with the given string representation, which is copied internally. */
	HPChain(const char *chain) : dChain(chain){}
	
	/** Creates an HP chain with the given string representation, which is copied internally. */
	HPChain(std::string chain) : dChain(chain){}

	/** Reference to the internal string. */
	std::string &get_chain(){ return dChain; }

	/** Length of the chain. */
	int length() const { return dChain.length(); }

	/** Validates the internal HP chain.
	 *
	 * Returns 'true' if the internal string contains only 'H' and 'P' characters. */
	bool validate();

	/** Returns the i-th bead type.
	 * \return Either 'P' or 'H', the i-th bead's type.
	 */
	char operator[](unsigned int i) const { return dChain[i]; }
};
