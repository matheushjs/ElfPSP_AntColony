#pragma once

#include <vector>
#include "vec3.h"

struct ACOSolution {
	std::vector<vec3<int>> dVector;

	static const char UP    = 0;
	static const char DOWN  = 1;
	static const char LEFT  = 2;
	static const char RIGHT = 3;
	static const char FRONT = 4;

	ACOSolution();
};

std::ostream& operator<<(std::ostream& stream, const ACOSolution &sol);

/********/

inline ACOSolution::ACOSolution()
: dVector({{0,0,0}, {1,0,0}})
{}

inline
std::ostream& operator<<(std::ostream& stream, const ACOSolution &sol){
	for(const vec3<int> &point: sol.dVector){
		stream << point << " ";
	}
	return stream;
}
