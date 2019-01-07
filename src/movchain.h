#pragma once

#include <iostream>
#include <vector>

class MovChain {
	std::vector<char> dVector;

	static const char FRONT = 0;
	static const char LEFT  = 1;
	static const char RIGHT = 2;
	static const char UP    = 3;
	static const char DOWN  = 4;

public:
	MovChain(){}

	const std::vector<char> &vector() const { return dVector; }
	int length() const { return dVector.size(); }

	void append(char movement);
	void append_random();
	void randomize(int idx);
};

std::ostream& operator<<(std::ostream& stream, const MovChain &chain);
