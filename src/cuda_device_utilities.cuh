#pragma once

/** \file cuda_device_utilities.cuh */


/** \defgroup cuda_device_utilities_cuh
 *  \{ */

/** Prints an int3 from within device code. */
__device__ inline
void print(int3 a){
	printf("(%d, %d, %d)", a.x, a.y, a.z);
}

/** Subtracts two int3. */
__device__ inline
int3 operator-(int3 a, int3 b){
	return {a.x-b.x, a.y-b.y, a.z-b.z};
}

/** Adds two int3. */
__device__ inline
int3 operator+(int3 a, int3 b){
	return {a.x+b.x, a.y+b.y, a.z+b.z};
}

/** Returns the norm 1 of an int3 (norm of the absolute value). */
__device__ inline
int norm1(int3 a){
	return abs(a.x) + abs(a.y) + abs(a.z);
}

/** Randomizes the given number, which also serves as the random generation seed. */
__device__ inline
void moveSeed(int &movingSeed){
	movingSeed ^= (movingSeed << 13);
	movingSeed ^= (movingSeed >> 17);
	movingSeed ^= (movingSeed << 15);
}

/** A wrapper for `moveSeed`, which also returns the generated random number. */
__device__ inline
int randomize(int &movingSeed){
	moveSeed(movingSeed);
	return movingSeed;
}

/** Version of `randomize` that returns a number in range [0,1]. */
__device__ inline
double randomize_d(int &movingSeed){
	moveSeed(movingSeed);
	return (unsigned int) movingSeed / (double) UINT_MAX;
}

/** Atomic addition, which only uses the built-in atomicCAS function.
 * This function aims at older versions of GPUs, which don't have atomicAdd().
 */
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

/** \} */ // CudaDeviceUtilities

/** Encapsulates a pointer to a memory region within the GPU memory space.
 * This templated class provides utilities for easy allocation and deallocation of memory in the GPU.
 * Memory is deallocated when the CUDAPointer object runs out of scope.
 */
template <typename T>
class CUDAPointer {
	T *dPointer; /**< Pointer to the memory region. */
	size_t dNElems; /**< Number of elements, of the templated type T, that fit in the allocated memory. */
public:
	/** Constructor that allocates memory within the GPU.
	  * \param nElems Number of elements, of the templated type T, that should fit in the allocated memory. */
	CUDAPointer(size_t nElems) : dNElems(nElems) {
		cudaMalloc(&dPointer, sizeof(T)*nElems);
	}

	/** Frees memory allocated upon construction. */
	~CUDAPointer(){ cudaFree(dPointer); }
	
	/** Returns the raw pointer stored internally. */
	T *get(){ return dPointer; }
	
	/** Enables casting a CUDAPointer to the associated raw pointer type T*. */
	operator T*() { return dPointer; }
	
	/** Enables casting a CUDAPointer to void*. */
	operator void*() { return (void *) dPointer; }
	
	/** Asynchronously copy elements from host memory to GPU memory.
	 * \param src Pointer to memory in host memory that holds the elements that should be copied.
	 *            The pointer must have the same number of elements as allocated upon construction
	 *            of the CUDAPointer object in question.
	 */
	void memcpyAsync(const T *src){
		cudaMemcpyAsync(dPointer, src, sizeof(T)*dNElems, cudaMemcpyHostToDevice);
	}
	
	/** Synchronously copy elements from GPU memory to host memory.
	 * \param dest Pointer to memory in host memory to which we should copy elements from GPU memory.
	 * \param nElems Number of elements to be copied. No checks regarding illegal memory access is done,
	 *               so the user is responsible for that.
	 */
	void copyTo(T *dest, int nElems){
		cudaMemcpy(dest, dPointer, sizeof(T)*nElems, cudaMemcpyDeviceToHost);
	}
	
	/** Overload of `copyTo(T*, int)` that copies the whole memory allocated upon construction to the host memory. */
	void copyTo(T *dest){ copyTo(dest, dNElems); }
};
