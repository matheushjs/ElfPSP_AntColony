#pragma once

#include <vector>
#include "vec3.h"

struct ACOSolution {
	std::vector<vec3<int>> dVector;

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
