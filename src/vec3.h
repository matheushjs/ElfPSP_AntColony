#pragma once

/** \file vec3.h */

#include <cmath>
#include <iostream>
#include <initializer_list>

/** \brief Represents a triple (x, y, z) of any type.
 */
template <typename T>
struct vec3 {
	/** (x,y,z) coordinates. @{ */
	T x, y, z; /** @} */

	/** \brief  Internal coordinates are not initialized and may contain junk. */
	vec3(){}

	/** \brief Example `v = vec3(1, 2, 3)`. */
	vec3(T a, T b, T c) : x(a), y(b), z(c){}

	/** \brief Example `v = {1, 2, 3}`. */
	vec3(std::initializer_list<T> l) : x(*l.begin()), y(*(l.begin()+1)), z(*(l.begin()+2)){}

	/** \brief Element-wise sum. */
	vec3<T> operator+(const vec3<T> other) const;

	/** \brief Element-wise subtraction. */
	vec3<T> operator-(const vec3<T> other) const;

	/** \brief Unary minus operation. Returns the opposite vector. */
	vec3<T> operator-() const;
	
	/** \brief Returns true if both vectors have equal coordinates. */
	bool operator==(const vec3<T> other) const;

	/** \brief Returns the dot product of two vectors. */
	T dot(const vec3<T> other) const;

	/** \brief Returns the dot product of the vector with itself. */
	T dot() const;

	/** \brief Returns the norm1 of the vector (absolute value norm). */
	T norm1() const;

	/** \brief Returns the euclidean norm of the vector. */
	T norm2() const;
};

/** \brief Allows a vector to be printed to `std::cout`. */
template <typename T>
std::ostream& operator<<(std::ostream& stream, const vec3<T> vec);

/******/

template <typename T>
inline vec3<T> vec3<T>::operator+(const vec3<T> other) const {
	vec3<T> result;
	result.x = this->x + other.x;
	result.y = this->y + other.y;
	result.z = this->z + other.z;
	return result;
}

template <typename T>
inline vec3<T> vec3<T>::operator-(const vec3<T> other) const {
	vec3<T> result;
	result.x = this->x - other.x;
	result.y = this->y - other.y;
	result.z = this->z - other.z;
	return result;
}

template <typename T>
inline vec3<T> vec3<T>::operator-() const {
	vec3<T> result;
	result.x = -this->x;
	result.y = -this->y;
	result.z = -this->z;
	return result;
}

template <typename T>
inline bool vec3<T>::operator==(const vec3<T> other) const {
	return (this->x == other.x) && (this->y == other.y) && (this->z == other.z);
}

template <typename T>
inline T vec3<T>::dot(const vec3<T> other) const {
	return this->x * other.x + this->y * other.y + this->z * other.z;
}

template <typename T>
inline T vec3<T>::dot() const {
	return this->x * this->x + this->y * this->y + this->z * this->z;
}

template <typename T>
inline T vec3<T>::norm1() const {
	return std::abs(this->x) + std::abs(this->y) + std::abs(this->z);
}

template <typename T>
inline T vec3<T>::norm2() const {
	return std::sqrt(this->dot());
}

template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const vec3<T> vec){
	stream << "(" << vec.x << "," << vec.y << "," << vec.z << ")";
	return stream;
}
