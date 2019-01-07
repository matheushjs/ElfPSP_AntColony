#pragma once

#include <cmath>
#include <iostream>
#include <initializer_list>

template <typename T>
struct vec3 {
	T x, y, z;

	vec3(){}
	vec3(T a, T b, T c) : x(a), y(b), z(c){}
	vec3(std::initializer_list<T> l) : x(*l.begin()), y(*(l.begin()+1)), z(*(l.begin()+2)){}

	vec3<T> operator+(const vec3<T> other);
	vec3<T> operator-(const vec3<T> other);
	vec3<T> operator-();
	T dot(const vec3<T> other);
	T dot();
	T norm1();
	T norm2();
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, const vec3<T> vec);

/******/

template <typename T>
inline vec3<T> vec3<T>::operator+(const vec3<T> other){
	vec3<T> result;
	result.x = this->x + other.x;
	result.y = this->y + other.y;
	result.z = this->z + other.z;
	return result;
}

template <typename T>
inline vec3<T> vec3<T>::operator-(const vec3<T> other){
	vec3<T> result;
	result.x = this->x - other.x;
	result.y = this->y - other.y;
	result.z = this->z - other.z;
	return result;
}

template <typename T>
inline vec3<T> vec3<T>::operator-(){
	vec3<T> result;
	result.x = -this->x;
	result.y = -this->y;
	result.z = -this->z;
	return result;
}

template <typename T>
inline T vec3<T>::dot(const vec3<T> other){
	return this->x * other.x + this->y * other.y + this->z * other.z;
}

template <typename T>
inline T vec3<T>::dot(){
	return this->x * this->x + this->y * this->y + this->z * this->z;
}

template <typename T>
inline T vec3<T>::norm1(){
	return std::abs(this->x) + std::abs(this->y) + std::abs(this->z);
}

template <typename T>
inline T vec3<T>::norm2(){
	return std::sqrt(this->dot());
}

template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const vec3<T> vec){
	stream << "(" << vec.x << "," << vec.y << "," << vec.z << ")";
	return stream;
}
