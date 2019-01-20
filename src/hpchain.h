#pragma once

#include <string>

/** \brief Encapsulates a string representing an HP chain. */
class HPChain {
	std::string dChain;

public:
	/** \brief Creates an empty HP chain. */
	HPChain()                  : dChain(""){}

	/** \brief Creates an HP chain with the given string representation, which is copied internally. */
	HPChain(const char *chain) : dChain(chain){}
	
	/** \brief Creates an HP chain with the given string representation, which is copied internally. */
	HPChain(std::string chain) : dChain(chain){}

	/** \brief Reference to the internal string. */
	std::string &get_chain(){ return dChain; }

	/** \brief Length of the chain. */
	int length() const { return dChain.length(); }

	/** \brief Validates the internal HP chain.
	 *
	 * Returns 'true' if the internal string contains only 'H' and 'P' characters. */
	bool validate();
};
