#pragma once

template <typename T>
struct vec3 {
	T x, y, z;

	vec3(){}
	vec3(T a, T b, T c) : x(a), y(b), z(c){}
};

template <typename T>
inline vec3<T> operator+(vec3<T> a, vec3<T> b){
	vec3<T> result;
	result.x = a.x + b.x;
	result.y = a.y + b.y;
	result.z = a.z + b.z;
	return result;
}
