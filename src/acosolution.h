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

	static vec3<int> get_direction_vector(vec3<int> prevDirection, char dir);

	ACOSolution();
};

std::ostream& operator<<(std::ostream& stream, const ACOSolution &sol);

/********/

inline ACOSolution::ACOSolution()
: dVector({{0,0,0}, {1,0,0}})
{}

inline
vec3<int> ACOSolution::get_direction_vector(vec3<int> prevDirection, char dir){
	if(dir == FRONT){
		return prevDirection;
	}

	// Will be either 1 or -1
	int sign = prevDirection.x + prevDirection.y + prevDirection.z;

	struct { char x, z; }
		isZero = {
			.x = (prevDirection.x == 0),
			.z = (prevDirection.z == 0),
		};
	
	vec3<int> retval({0,0,0});
	if(dir == RIGHT){
		retval.x = sign * isZero.x;
		retval.y = sign * !isZero.x;
	} else if(dir == LEFT){
		retval.x = -sign * isZero.x;
		retval.y = -sign * !isZero.x;
	} else if(dir == UP){
		retval.z = 1 * isZero.z;
		retval.y = 1 * !isZero.z;
	} else /* if(dir == DOWN) */ {
		retval.z = -1 * isZero.z;
		retval.y = -1 * !isZero.z;
	}

	return retval;
}

inline
std::ostream& operator<<(std::ostream& stream, const ACOSolution &sol){
	for(const vec3<int> &point: sol.dVector){
		stream << point << " ";
	}
	return stream;
}
