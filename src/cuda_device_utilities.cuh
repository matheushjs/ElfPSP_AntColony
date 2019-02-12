#pragma once

__device__ inline
void print(int3 a){
	printf("(%d, %d, %d)", a.x, a.y, a.z);
}

__device__ inline
int3 operator-(int3 a, int3 b){
	return {a.x-b.x, a.y-b.y, a.z-b.z};
}

__device__ inline
int3 operator+(int3 a, int3 b){
	return {a.x+b.x, a.y+b.y, a.z+b.z};
}

__device__ inline
int norm1(int3 a){
	return abs(a.x) + abs(a.y) + abs(a.z);
}

__device__ inline
void moveSeed(int &movingSeed){
	movingSeed ^= (movingSeed << 13);
	movingSeed ^= (movingSeed >> 17);
	movingSeed ^= (movingSeed << 15);
}

__device__ inline
int randomize(int &movingSeed){
	moveSeed(movingSeed);
	return movingSeed;
}

__device__ inline
double randomize_d(int &movingSeed){
	moveSeed(movingSeed);
	return (unsigned int) movingSeed / (double) UINT_MAX;
}
