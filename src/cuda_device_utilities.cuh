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

__device__ inline
double atomicAdd_d(double* address, double val){
	unsigned long long int* address_as_ull =
							  (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
						__double_as_longlong(val +
							__longlong_as_double(assumed)));
	// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}

template <typename T>
class CUDAPointer {
	T *dPointer;
	size_t dNElems;
public:
	CUDAPointer(size_t nElems) : dNElems(nElems) {
		cudaMalloc(&dPointer, sizeof(T)*nElems);
	}

	~CUDAPointer(){ cudaFree(dPointer); }
	
	T *get(){ return dPointer; }
	
	operator T*() { return dPointer; }
	
	operator void*() { return (void *) dPointer; }
	
	void memcpyAsync(const T *src){
		cudaMemcpyAsync(dPointer, src, sizeof(T)*dNElems, cudaMemcpyHostToDevice);
	}
	
	void copyTo(T *dest, int nElems){
		cudaMemcpy(dest, dPointer, sizeof(T)*nElems, cudaMemcpyDeviceToHost);
	}
	
	void copyTo(T *dest){ copyTo(dest, dNElems); }
};
