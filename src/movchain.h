#pragma once

#include <string>

class MovChain {
	std::string dMovChain;

	static const char FRONT = 'F';
	static const char LEFT  = 'L';
	static const char RIGHT = 'R';
	static const char UP    = 'U';
	static const char DOWN  = 'D';

	static const char sMovHash[];

public:
	MovChain()                     : dMovChain(""){}
	MovChain(std::string movchain) : dMovChain(movchain){}
	MovChain(const char *movchain) : dMovChain(movchain){}

	std::string &get_chain(){ return dMovChain; }
	int length() const { return dMovChain.length(); }

	void append(char movement);
	void append_random();
	void randomize(int idx);
};
