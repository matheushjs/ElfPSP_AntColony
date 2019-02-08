#pragma once

/** \file acosolution.h */

#include <vector>
#include <random>
#include "vec3.h"

/** Encapsulates a solution that is built by an ant.
 * A solution is thus a vector of coordinates of the beads of a protein.
 */
class ACOSolution {
	std::vector<vec3<int>> dVector; /**< Vector of bead coordinates. */
	std::vector<char> dDirections; /**< Vector of directions taken by the ant. */
	bool dError; /**< May be used to indicate errors. */

public:

	/** Default constructor; starts as an empty solution. */
	ACOSolution();

	/** Static constants representing relative directions. */
	static const char UP    = 0; /**< Relative direction. */
	static const char DOWN  = 1; /**< Relative direction. */
	static const char LEFT  = 2; /**< Relative direction. */
	static const char RIGHT = 3; /**< Relative direction. */
	static const char FRONT = 4; /**< Relative direction. */

	/** Maps directions to characters U, D, L, R, F. `DIR_TO_CHAR(0)` yields 'U'. */
	static const char DIR_TO_CHAR(char d);

	/** @{ */ /** Vector of bead coordinates. */
	std::vector<vec3<int>> &vector() { return dVector; }
	const std::vector<vec3<int>> &vector() const { return dVector; }
	/** @} */

	/** @{ */ /** Vector of directions taken by the ant. */
	std::vector<char> &directions(){ return dDirections; }
	const std::vector<char> &directions() const { return dDirections; }
	/** @} */

	/** Gets an specific bead. */
	vec3<int> bead(unsigned int i){ return dVector[i]; }

	/** Gets an specific direction. */
	char direction(unsigned int i){ return dDirections[i]; }

	/** Given a previous direction and a relative direction, returns the corresponding delta vector.
	 * Say we have a sequence of beads b1 and b2, each with their own coordinates, and that we want to add
	 *   a new bead b3 whose relative direction is D from b2 and b1.
	 * In this case, the previous direction is (b2 - b1).
	 * This function thus returns the delta vector that should be added to bead b2 in order to obtain b3.
	 * That is, b3 = (b2 + delta), where delta is the result of this function.
	 *
	 * \param prevDirection direction built from the previous 2 beads as (b2 - b1).
	 * \param dir relative direction to follow, starting from the given previous direction.
	 * \return the delta direction such that (b2 + delta) is the position of the next bead (say, b3)
	 */
	static vec3<int> DIRECTION_VECTOR(vec3<int> prevDirection, char dir);

	/** Returns the relative direction between the last 2 beads.
	 * \return (b2 - b1) where b2 is the last bead and b1 is the second last bead.
	 */
	vec3<int> previous_direction() const;

	/** Returns the number of H-H contacts within the solution.
	 * \param chain The HPChain of the protein that this ACOSolution represents.
	 */
	int count_contacts(const HPChain &chain) const;

	void perturb_one(std::mt19937 &dRandGen);

	/** Returns current error state. */
	bool has_error() { return dError; }

	/** Sets error to signalize existence of an error. */
	void set_error() { dError = true; }

	/** Clears the internal error state. */
	void clear_error() { dError = false; };
};

/** Prints coordinates and directions of the solution.
 * Each coordinate of the solution is printed sequentially, separated by whitespace.
 * Then comes a newline.
 * Then each direction is printed sequentially, not separated by whitespace. */
std::ostream& operator<<(std::ostream& stream, const ACOSolution &sol);

/********/

inline ACOSolution::ACOSolution()
: dVector({{0,0,0}, {1,0,0}}), dError(false)
{}

inline
const char ACOSolution::DIR_TO_CHAR(char d){
	static const char map[] = { 'U', 'D', 'R', 'L', 'F' };
	return map[(int) d];
}

inline
vec3<int> ACOSolution::DIRECTION_VECTOR(vec3<int> prevDirection, char dir){
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
vec3<int> ACOSolution::previous_direction() const {
	return dVector.back() - dVector[dVector.size() - 2];
}

inline
int ACOSolution::count_contacts(const HPChain &chain) const {
	int contacts = 0;

	for(unsigned i = 0; i < dVector.size(); i++){
		if(chain[i] == 'P') continue;

		for(unsigned j = i+1; j < dVector.size(); j++){
			if(chain[j] == 'P') continue;
			int norm1 = (dVector[i] - dVector[j]).norm1();
			if(norm1 == 1) contacts++;
		}
	}

	return contacts;
}

inline
void ACOSolution::perturb_one(std::mt19937 &gen){
	std::uniform_real_distribution<> randDist(0.0, 1.0);

	// Decide where to perturb
	int randPos = randDist(gen) * dDirections.size();

	// Perturb that position
	dDirections[randPos] = randDist(gen) * 5;

	// Rebuild the vector
	dVector = {{0,0,0}, {1,0,0}};
	for(char dir: dDirections){
		vec3<int> prev = previous_direction();
		vec3<int> delta = ACOSolution::DIRECTION_VECTOR(prev, dir);
		dVector.push_back(dVector.back() + delta);
	}
}

inline
std::ostream& operator<<(std::ostream& stream, const ACOSolution &sol){
	for(const vec3<int> &point: sol.vector())
		stream << point << " ";
	stream << "\n";
	for(char c: sol.directions())
		stream << ACOSolution::DIR_TO_CHAR(c);

	return stream;
}
