#pragma once

#include <string>

class MovChain {
	std::string dMovChain;

	static const char FRONT = 0;
	static const char LEFT  = 1;
	static const char RIGHT = 2;
	static const char UP    = 3;
	static const char DOWN  = 4;

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
