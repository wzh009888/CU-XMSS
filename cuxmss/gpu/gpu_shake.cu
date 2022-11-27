#include "gpu_shake.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>

__device__ void dev_SHA3_sponge(const unsigned char *inp, size_t len,
				unsigned char *out, unsigned char pad,
				size_t bitlen, size_t outbitlen);

__device__ void dev_SHAKE128(unsigned char *output, int outputByteLen,
			     const unsigned char *input, unsigned int inputByteLen)
{
	dev_SHA3_sponge(input, inputByteLen, output, '\x1f', 128, outputByteLen * 8);
} // dev_SHAKE128

__global__ void global_SHAKE128(unsigned char *output, int outputByteLen,
				const unsigned char *input, unsigned int inputByteLen)
{
	dev_SHA3_sponge(input, inputByteLen, output, '\x1f', 128, outputByteLen * 8);
} // global_SHAKE128

__device__ void dev_shake256(unsigned char *output, int outputByteLen,
			     const unsigned char *input, unsigned int inputByteLen)
{
	dev_SHA3_sponge(input, inputByteLen, output, '\x1f', 256,
			outputByteLen * 8);
} // dev_shake256

__global__ void global_shake256(unsigned char *output, int outputByteLen,
				const unsigned char *input, unsigned int inputByteLen)
{
	dev_shake256(output, outputByteLen, input, inputByteLen);
} // global_shake256

__global__ void global_parallel_shake256(const u8 *msg, u8 *md,
					 size_t in_size, size_t out_size, size_t n)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	dev_SHA3_sponge(msg + tid * in_size, in_size, md + out_size * tid,
			'\x1f', 256, out_size * 8);
} // global_parallel_shake256

#ifndef USING_PTX
#define BIT_INTERLEAVE (0)
#else // ifndef USING_PTX
#define BIT_INTERLEAVE (1)
#endif // ifndef USING_PTX

unsigned char this_rhotates[5][5] = {
	{  0, 1,  62,  28,   27	    },
	{ 36, 44, 6,   55,   20	    },
	{  3, 10, 43,  25,   39	    },
	{ 41, 45, 15,  21,   8	    },
	{ 18, 2,  61,  56,   14	    }
};
__constant__ unsigned char rhotates[5][5];

unsigned long this_iotas[24] = {
	BIT_INTERLEAVE ? 0x0000000000000001ULL : 0x0000000000000001ULL,
	BIT_INTERLEAVE ? 0x0000008900000000ULL : 0x0000000000008082ULL,
	BIT_INTERLEAVE ? 0x8000008b00000000ULL : 0x800000000000808aULL,
	BIT_INTERLEAVE ? 0x8000808000000000ULL : 0x8000000080008000ULL,
	BIT_INTERLEAVE ? 0x0000008b00000001ULL : 0x000000000000808bULL,
	BIT_INTERLEAVE ? 0x0000800000000001ULL : 0x0000000080000001ULL,
	BIT_INTERLEAVE ? 0x8000808800000001ULL : 0x8000000080008081ULL,
	BIT_INTERLEAVE ? 0x8000008200000001ULL : 0x8000000000008009ULL,
	BIT_INTERLEAVE ? 0x0000000b00000000ULL : 0x000000000000008aULL,
	BIT_INTERLEAVE ? 0x0000000a00000000ULL : 0x0000000000000088ULL,
	BIT_INTERLEAVE ? 0x0000808200000001ULL : 0x0000000080008009ULL,
	BIT_INTERLEAVE ? 0x0000800300000000ULL : 0x000000008000000aULL,
	BIT_INTERLEAVE ? 0x0000808b00000001ULL : 0x000000008000808bULL,
	BIT_INTERLEAVE ? 0x8000000b00000001ULL : 0x800000000000008bULL,
	BIT_INTERLEAVE ? 0x8000008a00000001ULL : 0x8000000000008089ULL,
	BIT_INTERLEAVE ? 0x8000008100000001ULL : 0x8000000000008003ULL,
	BIT_INTERLEAVE ? 0x8000008100000000ULL : 0x8000000000008002ULL,
	BIT_INTERLEAVE ? 0x8000000800000000ULL : 0x8000000000000080ULL,
	BIT_INTERLEAVE ? 0x0000008300000000ULL : 0x000000000000800aULL,
	BIT_INTERLEAVE ? 0x8000800300000000ULL : 0x800000008000000aULL,
	BIT_INTERLEAVE ? 0x8000808800000001ULL : 0x8000000080008081ULL,
	BIT_INTERLEAVE ? 0x8000008800000000ULL : 0x8000000000008080ULL,
	BIT_INTERLEAVE ? 0x0000800000000001ULL : 0x0000000080000001ULL,
	BIT_INTERLEAVE ? 0x8000808200000000ULL : 0x8000000080008008ULL
};
__constant__ unsigned long iotas[24];

#ifndef USING_PTX
# define dev_ROL64(v, n) (((v) << (n)) | ((v) >> (64 - (n))))
#else // ifdef USING_PTX
__forceinline__ __device__ static u32 ROL32(u32 v, u32 n)
{
	asm ("shf.l.clamp.b32 %0, %0, %0, %1;\n\t" \
	     : "+r" (v) : "r" (n));
	return v;
} // ROL32

__forceinline__ __device__ uint64_t dev_ROL64(unsigned long val, int offset)
{
	uint32_t hi = (uint32_t)(val >> 32), lo = (uint32_t)val;

	if (offset & 1) {
		uint32_t tmp = hi;

		offset >>= 1;
		hi = ROL32(lo, offset);
		lo = ROL32(tmp, offset + 1);
	} else {
		offset >>= 1;
		lo = ROL32(lo, offset);
		hi = ROL32(hi, offset);
	}

	return ((uint64_t)hi << 32) | lo;
} /* dev_ROL64 */

__forceinline__ __device__ uint64_t dev_ROL64_odd(unsigned long val, int offset)
{
	uint32_t hi = (uint32_t)(val >> 32), lo = (uint32_t)val;
	uint32_t tmp = hi;

	offset >>= 1;
	hi = ROL32(lo, offset);
	lo = ROL32(tmp, offset + 1);

	return ((uint64_t)hi << 32) | lo;
} // dev_ROL64_odd

__forceinline__ __device__ uint64_t dev_ROL64_even(unsigned long val, int offset)
{
	uint32_t hi = (uint32_t)(val >> 32), lo = (uint32_t)val;

	offset >>= 1;
	lo = ROL32(lo, offset);
	hi = ROL32(hi, offset);

	return ((uint64_t)hi << 32) | lo;
} // dev_ROL64_even

#define ROL64(val, offset) \
	{ \
		uint32_t hi = (uint32_t)(val >> 32), lo = (uint32_t)val; \
		if (offset & 1) { \
			uint32_t tmp = hi; \
			int off = (offset >> 1); \
			hi = ROL32(lo, off); \
			lo = ROL32(tmp, off + 1); \
		} else { \
			int off = (offset >> 1); \
			lo = ROL32(lo, off); \
			hi = ROL32(hi, off); \
		} \
		val = ((uint64_t)hi << 32) | lo; \
	}

#define ROL64_odd(val, offset) \
	{ \
		uint32_t hi = (uint32_t)(val >> 32), lo = (uint32_t)val; \
		uint32_t tmp = hi; \
		int off = (offset >> 1); \
		hi = ROL32(lo, off); \
		lo = ROL32(tmp, off + 1); \
		val = ((uint64_t)hi << 32) | lo; \
	}

#define ROL64_even(val, offset) \
	{ \
		uint32_t hi = (uint32_t)(val >> 32), lo = (uint32_t)val; \
		int off = (offset >> 1); \
		lo = ROL32(lo, off); \
		hi = ROL32(hi, off); \
		val = ((uint64_t)hi << 32) | lo; \
	}

#endif // ifdef USING_PTX

#define KECCAK_REF // fastest

// #define KECCAK_1X

// #define KECCAK_1X_ALT

// #define KECCAK_2X
// #define KECCAK_COMPLEMENTING_TRANSFORM   // used with KECCAK_2X

// KECCAK_4X // else

#if defined(KECCAK_REF)
/*
 * This is straightforward or "maximum clarity" implementation aiming
 * to resemble section 3.2 of the FIPS PUB 202 "SHA-3 Standard:
 * Permutation-Based Hash and Extendible-Output Functions" as much as
 * possible. With one caveat. Because of the way C stores matrices,
 * references to A[x,y] in the specification are presented as A[y][x].
 * Implementation unrolls inner x-loops so that modulo 5 operations are
 * explicitly pre-computed.
 */
__device__ void dev_Theta(unsigned long A[5][5])
// __device__ void dev_Theta(unsigned long A[5][5])
{
	unsigned long c[5], d[5];
	int y;

	c[0] = A[0][0];
	c[1] = A[0][1];
	c[2] = A[0][2];
	c[3] = A[0][3];
	c[4] = A[0][4];

#ifdef USING_UNROLL
	#pragma unroll
#endif // ifdef USING_UNROLL
	for (y = 1; y < 5; y++) {
	#ifdef USING_UNROLL
		#pragma unroll
	#endif // ifdef USING_UNROLL
		for (int x = 0; x < 5; x++) {
			c[x] ^= A[y][x];
		}
	}

#ifndef USING_PTX
	d[0] = dev_ROL64(c[1], 1) ^ c[4];
	d[1] = dev_ROL64(c[2], 1) ^ c[0];
	d[2] = dev_ROL64(c[3], 1) ^ c[1];
	d[3] = dev_ROL64(c[4], 1) ^ c[2];
	d[4] = dev_ROL64(c[0], 1) ^ c[3];
#else // ifdef USING_PTX
	d[0] = dev_ROL64_odd(c[1], 1) ^ c[4];
	d[1] = dev_ROL64_odd(c[2], 1) ^ c[0];
	d[2] = dev_ROL64_odd(c[3], 1) ^ c[1];
	d[3] = dev_ROL64_odd(c[4], 1) ^ c[2];
	d[4] = dev_ROL64_odd(c[0], 1) ^ c[3];
#endif // ifdef USING_PTX
#ifdef USING_UNROLL
	#pragma unroll 5
#endif // ifdef USING_UNROLL
	for (y = 0; y < 5; y++) {
		A[y][0] ^= d[0];
		A[y][1] ^= d[1];
		A[y][2] ^= d[2];
		A[y][3] ^= d[3];
		A[y][4] ^= d[4];
	}
} /* dev_Theta */

__device__ void dev_Rho(unsigned long A[5][5])
{
#ifndef USING_PTX
	size_t y;
#ifdef USING_UNROLL
	#pragma unroll 5
#endif // ifdef USING_UNROLL
	for (y = 0; y < 5; y++) {
		A[y][0] = dev_ROL64(A[y][0], rhotates[y][0]);
		A[y][1] = dev_ROL64(A[y][1], rhotates[y][1]);
		A[y][2] = dev_ROL64(A[y][2], rhotates[y][2]);
		A[y][3] = dev_ROL64(A[y][3], rhotates[y][3]);
		A[y][4] = dev_ROL64(A[y][4], rhotates[y][4]);
	}
#else // ifdef USING_PTX
	unsigned long *AA = (unsigned long *)&A[0];
	u8 *BB = (u8 *)&rhotates[0];
	u32 ss[13] = { 0, 2, 3, 5, 6, 7, 9, 11, 19, 20, 21, 23, 24 };
	u32 cc[12] = { 1, 4, 8, 10, 12, 13, 14, 15, 16, 17, 18, 22 };

// if USING_UNROLL is used, the results are wrong
// #ifdef USING_UNROLL
// 	#pragma unroll
// #endif // ifdef USING_UNROLL
	for (int i = 0; i < 13; i++)
		ROL64_even(AA[ss[i]], BB[ss[i]]);

// #ifdef USING_UNROLL
// 	#pragma unroll
// #endif // ifdef USING_UNROLL
	for (int i = 0; i < 12; i++)
		ROL64_odd(AA[cc[i]], BB[cc[i]]);
#endif // ifdef USING_PTX
} /* dev_Rho */

__forceinline__ __device__ void dev_Pi(unsigned long A[5][5])
// __device__ void dev_Pi(unsigned long A[5][5])
{
	unsigned long T[5][5];

	/*
	 * T = A
	 * A[y][x] = T[x][(3*y+x)%5]
	 */
	memcpy(T, A, sizeof(T));

#ifdef USING_UNROLL
	#pragma unroll 5
#endif // ifdef USING_UNROLL
	for (int y = 0; y < 5; y++) {
	#ifdef USING_UNROLL
		#pragma unroll 5
	#endif // ifdef USING_UNROLL
		for (int x = 0; x < 5; x++) {
			A[y][x] = T[x][(3 * y + x) % 5];
		}
	}
	// A[0][0] = T[0][0];
	// A[0][1] = T[1][1];
	// A[0][2] = T[2][2];
	// A[0][3] = T[3][3];
	// A[0][4] = T[4][4];
	//
	// A[1][0] = T[0][3];
	// A[1][1] = T[1][4];
	// A[1][2] = T[2][0];
	// A[1][3] = T[3][1];
	// A[1][4] = T[4][2];
	//
	// A[2][0] = T[0][1];
	// A[2][1] = T[1][2];
	// A[2][2] = T[2][3];
	// A[2][3] = T[3][4];
	// A[2][4] = T[4][0];
	//
	// A[3][0] = T[0][4];
	// A[3][1] = T[1][0];
	// A[3][2] = T[2][1];
	// A[3][3] = T[3][2];
	// A[3][4] = T[4][3];
	//
	// A[4][0] = T[0][2];
	// A[4][1] = T[1][3];
	// A[4][2] = T[2][4];
	// A[4][3] = T[3][0];
	// A[4][4] = T[4][1];
} /* dev_Pi */

__forceinline__ __device__ void dev_Chi(unsigned long A[5][5])
// __device__ void dev_Chi(unsigned long A[5][5])
{
	unsigned long C[5];
	size_t y;

	#ifdef USING_UNROLL
		#pragma unroll 5
	#endif // ifdef USING_UNROLL
	for (y = 0; y < 5; y++) {
		for (int x = 0; x < 5; x++) {
			C[x] = A[y][x] ^ (~A[y][(x + 1) % 5] & A[y][(x + 2) % 5]);
		}
		for (int x = 0; x < 5; x++) {
			A[y][x] = C[x];
		}
		// C[0] = A[y][0] ^ (~A[y][1] & A[y][2]);
		// C[1] = A[y][1] ^ (~A[y][2] & A[y][3]);
		// C[2] = A[y][2] ^ (~A[y][3] & A[y][4]);
		// C[3] = A[y][3] ^ (~A[y][4] & A[y][0]);
		// C[4] = A[y][4] ^ (~A[y][0] & A[y][1]);
		//
		// A[y][0] = C[0];
		// A[y][1] = C[1];
		// A[y][2] = C[2];
		// A[y][3] = C[3];
		// A[y][4] = C[4];
	}
} /* dev_Chi */

__forceinline__ __device__ void dev_Iota(unsigned long A[5][5], size_t i)
// __device__ void dev_Iota(unsigned long A[5][5], size_t i)
{
	// assert(i < (sizeof(iotas) / sizeof(iotas[0])));
	A[0][0] ^= iotas[i];
} /* dev_Iota */

__device__ void dev_KeccakF1600(unsigned long A[5][5])
{
	size_t i;

	for (i = 0; i < 24; i++) {
		dev_Theta(A);
		dev_Rho(A);
		dev_Pi(A);
		dev_Chi(A);
		dev_Iota(A, i);
	}
} /* dev_KeccakF1600 */

#elif defined(KECCAK_1X)
/*
 * This implementation is optimization of above code featuring unroll
 * of even y-loops, their fusion and code motion. It also minimizes
 * temporary storage. Compiler would normally do all these things for
 * you, purpose of manual optimization is to provide "unobscured"
 * reference for assembly implementation [in case this approach is
 * chosen for implementation on some platform]. In the nutshell it's
 * equivalent of "plane-per-plane processing" approach discussed in
 * section 2.4 of "Keccak implementation overview".
 */
__device__ void dev_Round(unsigned long A[5][5], size_t i)
{
	unsigned long C[5], E[2];       /* registers */
	unsigned long d[5], T[2][5];    /* memory    */

	// assert(i < (sizeof(iotas) / sizeof(iotas[0])));

	C[0] = A[0][0] ^ A[1][0] ^ A[2][0] ^ A[3][0] ^ A[4][0];
	C[1] = A[0][1] ^ A[1][1] ^ A[2][1] ^ A[3][1] ^ A[4][1];
	C[2] = A[0][2] ^ A[1][2] ^ A[2][2] ^ A[3][2] ^ A[4][2];
	C[3] = A[0][3] ^ A[1][3] ^ A[2][3] ^ A[3][3] ^ A[4][3];
	C[4] = A[0][4] ^ A[1][4] ^ A[2][4] ^ A[3][4] ^ A[4][4];

	d[0] = dev_ROL64(C[1], 1) ^ C[4];
	d[1] = dev_ROL64(C[2], 1) ^ C[0];
	d[2] = dev_ROL64(C[3], 1) ^ C[1];
	d[3] = dev_ROL64(C[4], 1) ^ C[2];
	d[4] = dev_ROL64(C[0], 1) ^ C[3];

	T[0][0] = A[3][0] ^ d[0]; /* borrow T[0][0] */
	T[0][1] = A[0][1] ^ d[1];
	T[0][2] = A[0][2] ^ d[2];
	T[0][3] = A[0][3] ^ d[3];
	T[0][4] = A[0][4] ^ d[4];

	C[0] =       A[0][0] ^ d[0];/* rotate by 0 */
	C[1] = dev_ROL64(A[1][1] ^ d[1], rhotates[1][1]);
	C[2] = dev_ROL64(A[2][2] ^ d[2], rhotates[2][2]);
	C[3] = dev_ROL64(A[3][3] ^ d[3], rhotates[3][3]);
	C[4] = dev_ROL64(A[4][4] ^ d[4], rhotates[4][4]);

	A[0][0] = C[0] ^ (~C[1] & C[2]) ^ iotas[i];
	A[0][1] = C[1] ^ (~C[2] & C[3]);
	A[0][2] = C[2] ^ (~C[3] & C[4]);
	A[0][3] = C[3] ^ (~C[4] & C[0]);
	A[0][4] = C[4] ^ (~C[0] & C[1]);

	T[1][0] = A[1][0] ^ (C[3] = d[0]);
	T[1][1] = A[2][1] ^ (C[4] = d[1]); /* borrow T[1][1] */
	T[1][2] = A[1][2] ^ (E[0] = d[2]);
	T[1][3] = A[1][3] ^ (E[1] = d[3]);
	T[1][4] = A[2][4] ^ (C[2] = d[4]); /* borrow T[1][4] */

	C[0] = dev_ROL64(T[0][3],        rhotates[0][3]);
	C[1] = dev_ROL64(A[1][4] ^ C[2], rhotates[1][4]);       /* d[4] */
	C[2] = dev_ROL64(A[2][0] ^ C[3], rhotates[2][0]);       /* d[0] */
	C[3] = dev_ROL64(A[3][1] ^ C[4], rhotates[3][1]);       /* d[1] */
	C[4] = dev_ROL64(A[4][2] ^ E[0], rhotates[4][2]);       /* d[2] */

	A[1][0] = C[0] ^ (~C[1] & C[2]);
	A[1][1] = C[1] ^ (~C[2] & C[3]);
	A[1][2] = C[2] ^ (~C[3] & C[4]);
	A[1][3] = C[3] ^ (~C[4] & C[0]);
	A[1][4] = C[4] ^ (~C[0] & C[1]);

	C[0] = dev_ROL64(T[0][1],        rhotates[0][1]);
	C[1] = dev_ROL64(T[1][2],        rhotates[1][2]);
	C[2] = dev_ROL64(A[2][3] ^ d[3], rhotates[2][3]);
	C[3] = dev_ROL64(A[3][4] ^ d[4], rhotates[3][4]);
	C[4] = dev_ROL64(A[4][0] ^ d[0], rhotates[4][0]);

	A[2][0] = C[0] ^ (~C[1] & C[2]);
	A[2][1] = C[1] ^ (~C[2] & C[3]);
	A[2][2] = C[2] ^ (~C[3] & C[4]);
	A[2][3] = C[3] ^ (~C[4] & C[0]);
	A[2][4] = C[4] ^ (~C[0] & C[1]);

	C[0] = dev_ROL64(T[0][4],        rhotates[0][4]);
	C[1] = dev_ROL64(T[1][0],        rhotates[1][0]);
	C[2] = dev_ROL64(T[1][1],        rhotates[2][1]);/* originally A[2][1] */
	C[3] = dev_ROL64(A[3][2] ^ d[2], rhotates[3][2]);
	C[4] = dev_ROL64(A[4][3] ^ d[3], rhotates[4][3]);

	A[3][0] = C[0] ^ (~C[1] & C[2]);
	A[3][1] = C[1] ^ (~C[2] & C[3]);
	A[3][2] = C[2] ^ (~C[3] & C[4]);
	A[3][3] = C[3] ^ (~C[4] & C[0]);
	A[3][4] = C[4] ^ (~C[0] & C[1]);

	C[0] = dev_ROL64(T[0][2],        rhotates[0][2]);
	C[1] = dev_ROL64(T[1][3],        rhotates[1][3]);
	C[2] = dev_ROL64(T[1][4],        rhotates[2][4]);       /* originally A[2][4] */
	C[3] = dev_ROL64(T[0][0],        rhotates[3][0]);       /* originally A[3][0] */
	C[4] = dev_ROL64(A[4][1] ^ d[1], rhotates[4][1]);

	A[4][0] = C[0] ^ (~C[1] & C[2]);
	A[4][1] = C[1] ^ (~C[2] & C[3]);
	A[4][2] = C[2] ^ (~C[3] & C[4]);
	A[4][3] = C[3] ^ (~C[4] & C[0]);
	A[4][4] = C[4] ^ (~C[0] & C[1]);
} /* dev_Round */

__device__ void dev_KeccakF1600(unsigned long A[5][5])
{
	size_t i;

	for (i = 0; i < 24; i++) {
		dev_Round(A, i);
	}
} /* dev_KeccakF1600 */

#elif defined(KECCAK_1X_ALT)
/*
 * This is variant of above KECCAK_1X that reduces requirement for
 * temporary storage even further, but at cost of more updates to A[][].
 * It's less suitable if A[][] is memory bound, but better if it's
 * register bound.
 */

__device__ void dev_Round(unsigned long A[5][5], size_t i)
{
	unsigned long C[5], d[5];

	assert(i < (sizeof(iotas) / sizeof(iotas[0])));

	C[0] = A[0][0] ^ A[1][0] ^ A[2][0] ^ A[3][0] ^ A[4][0];
	C[1] = A[0][1] ^ A[1][1] ^ A[2][1] ^ A[3][1] ^ A[4][1];
	C[2] = A[0][2] ^ A[1][2] ^ A[2][2] ^ A[3][2] ^ A[4][2];
	C[3] = A[0][3] ^ A[1][3] ^ A[2][3] ^ A[3][3] ^ A[4][3];
	C[4] = A[0][4] ^ A[1][4] ^ A[2][4] ^ A[3][4] ^ A[4][4];

	d[1] = C[0] ^  dev_ROL64(C[2], 1);
	d[2] = C[1] ^  dev_ROL64(C[3], 1);
	d[3] = C[2] ^= dev_ROL64(C[4], 1);
	d[4] = C[3] ^= dev_ROL64(C[0], 1);
	d[0] = C[4] ^= dev_ROL64(C[1], 1);

	A[0][1] ^= d[1];
	A[1][1] ^= d[1];
	A[2][1] ^= d[1];
	A[3][1] ^= d[1];
	A[4][1] ^= d[1];

	A[0][2] ^= d[2];
	A[1][2] ^= d[2];
	A[2][2] ^= d[2];
	A[3][2] ^= d[2];
	A[4][2] ^= d[2];

	A[0][3] ^= C[2];
	A[1][3] ^= C[2];
	A[2][3] ^= C[2];
	A[3][3] ^= C[2];
	A[4][3] ^= C[2];

	A[0][4] ^= C[3];
	A[1][4] ^= C[3];
	A[2][4] ^= C[3];
	A[3][4] ^= C[3];
	A[4][4] ^= C[3];

	A[0][0] ^= C[4];
	A[1][0] ^= C[4];
	A[2][0] ^= C[4];
	A[3][0] ^= C[4];
	A[4][0] ^= C[4];

	C[1] = A[0][1];
	C[2] = A[0][2];
	C[3] = A[0][3];
	C[4] = A[0][4];

	A[0][1] = dev_ROL64(A[1][1], rhotates[1][1]);
	A[0][2] = dev_ROL64(A[2][2], rhotates[2][2]);
	A[0][3] = dev_ROL64(A[3][3], rhotates[3][3]);
	A[0][4] = dev_ROL64(A[4][4], rhotates[4][4]);

	A[1][1] = dev_ROL64(A[1][4], rhotates[1][4]);
	A[2][2] = dev_ROL64(A[2][3], rhotates[2][3]);
	A[3][3] = dev_ROL64(A[3][2], rhotates[3][2]);
	A[4][4] = dev_ROL64(A[4][1], rhotates[4][1]);

	A[1][4] = dev_ROL64(A[4][2], rhotates[4][2]);
	A[2][3] = dev_ROL64(A[3][4], rhotates[3][4]);
	A[3][2] = dev_ROL64(A[2][1], rhotates[2][1]);
	A[4][1] = dev_ROL64(A[1][3], rhotates[1][3]);

	A[4][2] = dev_ROL64(A[2][4], rhotates[2][4]);
	A[3][4] = dev_ROL64(A[4][3], rhotates[4][3]);
	A[2][1] = dev_ROL64(A[1][2], rhotates[1][2]);
	A[1][3] = dev_ROL64(A[3][1], rhotates[3][1]);

	A[2][4] = dev_ROL64(A[4][0], rhotates[4][0]);
	A[4][3] = dev_ROL64(A[3][0], rhotates[3][0]);
	A[1][2] = dev_ROL64(A[2][0], rhotates[2][0]);
	A[3][1] = dev_ROL64(A[1][0], rhotates[1][0]);

	A[1][0] = dev_ROL64(C[3],    rhotates[0][3]);
	A[2][0] = dev_ROL64(C[1],    rhotates[0][1]);
	A[3][0] = dev_ROL64(C[4],    rhotates[0][4]);
	A[4][0] = dev_ROL64(C[2],    rhotates[0][2]);

	C[0] = A[0][0];
	C[1] = A[1][0];
	d[0] = A[0][1];
	d[1] = A[1][1];

	A[0][0] ^= (~A[0][1] & A[0][2]);
	A[1][0] ^= (~A[1][1] & A[1][2]);
	A[0][1] ^= (~A[0][2] & A[0][3]);
	A[1][1] ^= (~A[1][2] & A[1][3]);
	A[0][2] ^= (~A[0][3] & A[0][4]);
	A[1][2] ^= (~A[1][3] & A[1][4]);
	A[0][3] ^= (~A[0][4] & C[0]);
	A[1][3] ^= (~A[1][4] & C[1]);
	A[0][4] ^= (~C[0]    & d[0]);
	A[1][4] ^= (~C[1]    & d[1]);

	C[2] = A[2][0];
	C[3] = A[3][0];
	d[2] = A[2][1];
	d[3] = A[3][1];

	A[2][0] ^= (~A[2][1] & A[2][2]);
	A[3][0] ^= (~A[3][1] & A[3][2]);
	A[2][1] ^= (~A[2][2] & A[2][3]);
	A[3][1] ^= (~A[3][2] & A[3][3]);
	A[2][2] ^= (~A[2][3] & A[2][4]);
	A[3][2] ^= (~A[3][3] & A[3][4]);
	A[2][3] ^= (~A[2][4] & C[2]);
	A[3][3] ^= (~A[3][4] & C[3]);
	A[2][4] ^= (~C[2]    & d[2]);
	A[3][4] ^= (~C[3]    & d[3]);

	C[4] = A[4][0];
	d[4] = A[4][1];

	A[4][0] ^= (~A[4][1] & A[4][2]);
	A[4][1] ^= (~A[4][2] & A[4][3]);
	A[4][2] ^= (~A[4][3] & A[4][4]);
	A[4][3] ^= (~A[4][4] & C[4]);
	A[4][4] ^= (~C[4]    & d[4]);
	A[0][0] ^= iotas[i];
} /* dev_Round */

__device__ void dev_KeccakF1600(unsigned long A[5][5])
{
	size_t i;

	for (i = 0; i < 24; i++) {
		dev_Round(A, i);
	}
} /* dev_KeccakF1600 */

#elif defined(KECCAK_2X)
/*
 * This implementation is variant of KECCAK_1X above with outer-most
 * dev_Round loop unrolled twice. This allows to take temporary storage
 * out of dev_Round procedure and simplify references to it by alternating
 * it with actual data (see dev_Round loop below). Originally it was meant
 * rather as reference for an assembly implementation, but it seems to
 * play best with compilers [as well as provide best instruction per
 * processed byte ratio at minimal dev_Round unroll factor]...
 */
__device__ void dev_Round(unsigned long R[5][5], unsigned long A[5][5], size_t i)
{
	unsigned long C[5], d[5];

	assert(i < (sizeof(iotas) / sizeof(iotas[0])));

	C[0] = A[0][0] ^ A[1][0] ^ A[2][0] ^ A[3][0] ^ A[4][0];
	C[1] = A[0][1] ^ A[1][1] ^ A[2][1] ^ A[3][1] ^ A[4][1];
	C[2] = A[0][2] ^ A[1][2] ^ A[2][2] ^ A[3][2] ^ A[4][2];
	C[3] = A[0][3] ^ A[1][3] ^ A[2][3] ^ A[3][3] ^ A[4][3];
	C[4] = A[0][4] ^ A[1][4] ^ A[2][4] ^ A[3][4] ^ A[4][4];

	d[0] = dev_ROL64(C[1], 1) ^ C[4];
	d[1] = dev_ROL64(C[2], 1) ^ C[0];
	d[2] = dev_ROL64(C[3], 1) ^ C[1];
	d[3] = dev_ROL64(C[4], 1) ^ C[2];
	d[4] = dev_ROL64(C[0], 1) ^ C[3];

	C[0] =       A[0][0] ^ d[0];/* rotate by 0 */
	C[1] = dev_ROL64(A[1][1] ^ d[1], rhotates[1][1]);
	C[2] = dev_ROL64(A[2][2] ^ d[2], rhotates[2][2]);
	C[3] = dev_ROL64(A[3][3] ^ d[3], rhotates[3][3]);
	C[4] = dev_ROL64(A[4][4] ^ d[4], rhotates[4][4]);

#ifdef KECCAK_COMPLEMENTING_TRANSFORM
	R[0][0] = C[0] ^ ( C[1] | C[2]) ^ iotas[i];
	R[0][1] = C[1] ^ (~C[2] | C[3]);
	R[0][2] = C[2] ^ ( C[3] & C[4]);
	R[0][3] = C[3] ^ ( C[4] | C[0]);
	R[0][4] = C[4] ^ ( C[0] & C[1]);
#else  /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */
	R[0][0] = C[0] ^ (~C[1] & C[2]) ^ iotas[i];
	R[0][1] = C[1] ^ (~C[2] & C[3]);
	R[0][2] = C[2] ^ (~C[3] & C[4]);
	R[0][3] = C[3] ^ (~C[4] & C[0]);
	R[0][4] = C[4] ^ (~C[0] & C[1]);
#endif /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */

	C[0] = dev_ROL64(A[0][3] ^ d[3], rhotates[0][3]);
	C[1] = dev_ROL64(A[1][4] ^ d[4], rhotates[1][4]);
	C[2] = dev_ROL64(A[2][0] ^ d[0], rhotates[2][0]);
	C[3] = dev_ROL64(A[3][1] ^ d[1], rhotates[3][1]);
	C[4] = dev_ROL64(A[4][2] ^ d[2], rhotates[4][2]);

#ifdef KECCAK_COMPLEMENTING_TRANSFORM
	R[1][0] = C[0] ^ (C[1] |  C[2]);
	R[1][1] = C[1] ^ (C[2] &  C[3]);
	R[1][2] = C[2] ^ (C[3] | ~C[4]);
	R[1][3] = C[3] ^ (C[4] |  C[0]);
	R[1][4] = C[4] ^ (C[0] &  C[1]);
#else  /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */
	R[1][0] = C[0] ^ (~C[1] & C[2]);
	R[1][1] = C[1] ^ (~C[2] & C[3]);
	R[1][2] = C[2] ^ (~C[3] & C[4]);
	R[1][3] = C[3] ^ (~C[4] & C[0]);
	R[1][4] = C[4] ^ (~C[0] & C[1]);
#endif /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */

	C[0] = dev_ROL64(A[0][1] ^ d[1], rhotates[0][1]);
	C[1] = dev_ROL64(A[1][2] ^ d[2], rhotates[1][2]);
	C[2] = dev_ROL64(A[2][3] ^ d[3], rhotates[2][3]);
	C[3] = dev_ROL64(A[3][4] ^ d[4], rhotates[3][4]);
	C[4] = dev_ROL64(A[4][0] ^ d[0], rhotates[4][0]);

#ifdef KECCAK_COMPLEMENTING_TRANSFORM
	R[2][0] =  C[0] ^ ( C[1] | C[2]);
	R[2][1] =  C[1] ^ ( C[2] & C[3]);
	R[2][2] =  C[2] ^ (~C[3] & C[4]);
	R[2][3] = ~C[3] ^ ( C[4] | C[0]);
	R[2][4] =  C[4] ^ ( C[0] & C[1]);
#else  /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */
	R[2][0] = C[0] ^ (~C[1] & C[2]);
	R[2][1] = C[1] ^ (~C[2] & C[3]);
	R[2][2] = C[2] ^ (~C[3] & C[4]);
	R[2][3] = C[3] ^ (~C[4] & C[0]);
	R[2][4] = C[4] ^ (~C[0] & C[1]);
#endif /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */

	C[0] = dev_ROL64(A[0][4] ^ d[4], rhotates[0][4]);
	C[1] = dev_ROL64(A[1][0] ^ d[0], rhotates[1][0]);
	C[2] = dev_ROL64(A[2][1] ^ d[1], rhotates[2][1]);
	C[3] = dev_ROL64(A[3][2] ^ d[2], rhotates[3][2]);
	C[4] = dev_ROL64(A[4][3] ^ d[3], rhotates[4][3]);

#ifdef KECCAK_COMPLEMENTING_TRANSFORM
	R[3][0] =  C[0] ^ ( C[1] & C[2]);
	R[3][1] =  C[1] ^ ( C[2] | C[3]);
	R[3][2] =  C[2] ^ (~C[3] | C[4]);
	R[3][3] = ~C[3] ^ ( C[4] & C[0]);
	R[3][4] =  C[4] ^ ( C[0] | C[1]);
#else  /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */
	R[3][0] = C[0] ^ (~C[1] & C[2]);
	R[3][1] = C[1] ^ (~C[2] & C[3]);
	R[3][2] = C[2] ^ (~C[3] & C[4]);
	R[3][3] = C[3] ^ (~C[4] & C[0]);
	R[3][4] = C[4] ^ (~C[0] & C[1]);
#endif /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */

	C[0] = dev_ROL64(A[0][2] ^ d[2], rhotates[0][2]);
	C[1] = dev_ROL64(A[1][3] ^ d[3], rhotates[1][3]);
	C[2] = dev_ROL64(A[2][4] ^ d[4], rhotates[2][4]);
	C[3] = dev_ROL64(A[3][0] ^ d[0], rhotates[3][0]);
	C[4] = dev_ROL64(A[4][1] ^ d[1], rhotates[4][1]);

#ifdef KECCAK_COMPLEMENTING_TRANSFORM
	R[4][0] =  C[0] ^ (~C[1] & C[2]);
	R[4][1] = ~C[1] ^ ( C[2] | C[3]);
	R[4][2] =  C[2] ^ ( C[3] & C[4]);
	R[4][3] =  C[3] ^ ( C[4] | C[0]);
	R[4][4] =  C[4] ^ ( C[0] & C[1]);
#else  /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */
	R[4][0] = C[0] ^ (~C[1] & C[2]);
	R[4][1] = C[1] ^ (~C[2] & C[3]);
	R[4][2] = C[2] ^ (~C[3] & C[4]);
	R[4][3] = C[3] ^ (~C[4] & C[0]);
	R[4][4] = C[4] ^ (~C[0] & C[1]);
#endif /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */
} /* dev_Round */

__device__ void dev_KeccakF1600(unsigned long A[5][5])
{
	unsigned long T[5][5];
	size_t i;

#ifdef KECCAK_COMPLEMENTING_TRANSFORM
	A[0][1] = ~A[0][1];
	A[0][2] = ~A[0][2];
	A[1][3] = ~A[1][3];
	A[2][2] = ~A[2][2];
	A[3][2] = ~A[3][2];
	A[4][0] = ~A[4][0];
#endif /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */

	for (i = 0; i < 24; i += 2) {
		dev_Round(T, A, i);
		dev_Round(A, T, i + 1);
	}

#ifdef KECCAK_COMPLEMENTING_TRANSFORM
	A[0][1] = ~A[0][1];
	A[0][2] = ~A[0][2];
	A[1][3] = ~A[1][3];
	A[2][2] = ~A[2][2];
	A[3][2] = ~A[3][2];
	A[4][0] = ~A[4][0];
#endif /* ifdef KECCAK_COMPLEMENTING_TRANSFORM */
} /* dev_KeccakF1600 */

#else   /* define KECCAK_INPLACE to compile this code path */
/*
 * This implementation is KECCAK_1X from above combined 4 times with
 * a twist that allows to omit temporary storage and perform in-place
 * processing. It's discussed in section 2.5 of "Keccak implementation
 * overview". It's likely to be best suited for processors with large
 * register bank... On the other hand processor with large register
 * bank can as well use KECCAK_1X_ALT, it would be as fast but much
 * more compact...
 */
__device__ void dev_FourRounds(unsigned long A[5][5], size_t i)
{
	unsigned long B[5], C[5], d[5];

	assert(i <= (sizeof(iotas) / sizeof(iotas[0]) - 4));

	/* dev_Round 4*n */
	C[0] = A[0][0] ^ A[1][0] ^ A[2][0] ^ A[3][0] ^ A[4][0];
	C[1] = A[0][1] ^ A[1][1] ^ A[2][1] ^ A[3][1] ^ A[4][1];
	C[2] = A[0][2] ^ A[1][2] ^ A[2][2] ^ A[3][2] ^ A[4][2];
	C[3] = A[0][3] ^ A[1][3] ^ A[2][3] ^ A[3][3] ^ A[4][3];
	C[4] = A[0][4] ^ A[1][4] ^ A[2][4] ^ A[3][4] ^ A[4][4];

	d[0] = dev_ROL64(C[1], 1) ^ C[4];
	d[1] = dev_ROL64(C[2], 1) ^ C[0];
	d[2] = dev_ROL64(C[3], 1) ^ C[1];
	d[3] = dev_ROL64(C[4], 1) ^ C[2];
	d[4] = dev_ROL64(C[0], 1) ^ C[3];

	B[0] =       A[0][0] ^ d[0];/* rotate by 0 */
	B[1] = dev_ROL64(A[1][1] ^ d[1], rhotates[1][1]);
	B[2] = dev_ROL64(A[2][2] ^ d[2], rhotates[2][2]);
	B[3] = dev_ROL64(A[3][3] ^ d[3], rhotates[3][3]);
	B[4] = dev_ROL64(A[4][4] ^ d[4], rhotates[4][4]);

	C[0] = A[0][0] = B[0] ^ (~B[1] & B[2]) ^ iotas[i];
	C[1] = A[1][1] = B[1] ^ (~B[2] & B[3]);
	C[2] = A[2][2] = B[2] ^ (~B[3] & B[4]);
	C[3] = A[3][3] = B[3] ^ (~B[4] & B[0]);
	C[4] = A[4][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[0][3] ^ d[3], rhotates[0][3]);
	B[1] = dev_ROL64(A[1][4] ^ d[4], rhotates[1][4]);
	B[2] = dev_ROL64(A[2][0] ^ d[0], rhotates[2][0]);
	B[3] = dev_ROL64(A[3][1] ^ d[1], rhotates[3][1]);
	B[4] = dev_ROL64(A[4][2] ^ d[2], rhotates[4][2]);

	C[0] ^= A[2][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[3][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[4][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[0][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[1][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[0][1] ^ d[1], rhotates[0][1]);
	B[1] = dev_ROL64(A[1][2] ^ d[2], rhotates[1][2]);
	B[2] = dev_ROL64(A[2][3] ^ d[3], rhotates[2][3]);
	B[3] = dev_ROL64(A[3][4] ^ d[4], rhotates[3][4]);
	B[4] = dev_ROL64(A[4][0] ^ d[0], rhotates[4][0]);

	C[0] ^= A[4][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[0][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[1][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[2][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[3][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[0][4] ^ d[4], rhotates[0][4]);
	B[1] = dev_ROL64(A[1][0] ^ d[0], rhotates[1][0]);
	B[2] = dev_ROL64(A[2][1] ^ d[1], rhotates[2][1]);
	B[3] = dev_ROL64(A[3][2] ^ d[2], rhotates[3][2]);
	B[4] = dev_ROL64(A[4][3] ^ d[3], rhotates[4][3]);

	C[0] ^= A[1][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[2][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[3][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[4][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[0][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[0][2] ^ d[2], rhotates[0][2]);
	B[1] = dev_ROL64(A[1][3] ^ d[3], rhotates[1][3]);
	B[2] = dev_ROL64(A[2][4] ^ d[4], rhotates[2][4]);
	B[3] = dev_ROL64(A[3][0] ^ d[0], rhotates[3][0]);
	B[4] = dev_ROL64(A[4][1] ^ d[1], rhotates[4][1]);

	C[0] ^= A[3][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[4][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[0][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[1][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[2][4] = B[4] ^ (~B[0] & B[1]);

	/* dev_Round 4*n+1 */
	d[0] = dev_ROL64(C[1], 1) ^ C[4];
	d[1] = dev_ROL64(C[2], 1) ^ C[0];
	d[2] = dev_ROL64(C[3], 1) ^ C[1];
	d[3] = dev_ROL64(C[4], 1) ^ C[2];
	d[4] = dev_ROL64(C[0], 1) ^ C[3];

	B[0] =       A[0][0] ^ d[0];/* rotate by 0 */
	B[1] = dev_ROL64(A[3][1] ^ d[1], rhotates[1][1]);
	B[2] = dev_ROL64(A[1][2] ^ d[2], rhotates[2][2]);
	B[3] = dev_ROL64(A[4][3] ^ d[3], rhotates[3][3]);
	B[4] = dev_ROL64(A[2][4] ^ d[4], rhotates[4][4]);

	C[0] = A[0][0] = B[0] ^ (~B[1] & B[2]) ^ iotas[i + 1];
	C[1] = A[3][1] = B[1] ^ (~B[2] & B[3]);
	C[2] = A[1][2] = B[2] ^ (~B[3] & B[4]);
	C[3] = A[4][3] = B[3] ^ (~B[4] & B[0]);
	C[4] = A[2][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[3][3] ^ d[3], rhotates[0][3]);
	B[1] = dev_ROL64(A[1][4] ^ d[4], rhotates[1][4]);
	B[2] = dev_ROL64(A[4][0] ^ d[0], rhotates[2][0]);
	B[3] = dev_ROL64(A[2][1] ^ d[1], rhotates[3][1]);
	B[4] = dev_ROL64(A[0][2] ^ d[2], rhotates[4][2]);

	C[0] ^= A[4][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[2][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[0][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[3][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[1][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[1][1] ^ d[1], rhotates[0][1]);
	B[1] = dev_ROL64(A[4][2] ^ d[2], rhotates[1][2]);
	B[2] = dev_ROL64(A[2][3] ^ d[3], rhotates[2][3]);
	B[3] = dev_ROL64(A[0][4] ^ d[4], rhotates[3][4]);
	B[4] = dev_ROL64(A[3][0] ^ d[0], rhotates[4][0]);

	C[0] ^= A[3][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[1][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[4][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[2][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[0][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[4][4] ^ d[4], rhotates[0][4]);
	B[1] = dev_ROL64(A[2][0] ^ d[0], rhotates[1][0]);
	B[2] = dev_ROL64(A[0][1] ^ d[1], rhotates[2][1]);
	B[3] = dev_ROL64(A[3][2] ^ d[2], rhotates[3][2]);
	B[4] = dev_ROL64(A[1][3] ^ d[3], rhotates[4][3]);

	C[0] ^= A[2][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[0][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[3][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[1][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[4][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[2][2] ^ d[2], rhotates[0][2]);
	B[1] = dev_ROL64(A[0][3] ^ d[3], rhotates[1][3]);
	B[2] = dev_ROL64(A[3][4] ^ d[4], rhotates[2][4]);
	B[3] = dev_ROL64(A[1][0] ^ d[0], rhotates[3][0]);
	B[4] = dev_ROL64(A[4][1] ^ d[1], rhotates[4][1]);

	C[0] ^= A[1][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[4][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[2][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[0][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[3][4] = B[4] ^ (~B[0] & B[1]);

	/* dev_Round 4*n+2 */
	d[0] = dev_ROL64(C[1], 1) ^ C[4];
	d[1] = dev_ROL64(C[2], 1) ^ C[0];
	d[2] = dev_ROL64(C[3], 1) ^ C[1];
	d[3] = dev_ROL64(C[4], 1) ^ C[2];
	d[4] = dev_ROL64(C[0], 1) ^ C[3];

	B[0] =       A[0][0] ^ d[0];/* rotate by 0 */
	B[1] = dev_ROL64(A[2][1] ^ d[1], rhotates[1][1]);
	B[2] = dev_ROL64(A[4][2] ^ d[2], rhotates[2][2]);
	B[3] = dev_ROL64(A[1][3] ^ d[3], rhotates[3][3]);
	B[4] = dev_ROL64(A[3][4] ^ d[4], rhotates[4][4]);

	C[0] = A[0][0] = B[0] ^ (~B[1] & B[2]) ^ iotas[i + 2];
	C[1] = A[2][1] = B[1] ^ (~B[2] & B[3]);
	C[2] = A[4][2] = B[2] ^ (~B[3] & B[4]);
	C[3] = A[1][3] = B[3] ^ (~B[4] & B[0]);
	C[4] = A[3][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[4][3] ^ d[3], rhotates[0][3]);
	B[1] = dev_ROL64(A[1][4] ^ d[4], rhotates[1][4]);
	B[2] = dev_ROL64(A[3][0] ^ d[0], rhotates[2][0]);
	B[3] = dev_ROL64(A[0][1] ^ d[1], rhotates[3][1]);
	B[4] = dev_ROL64(A[2][2] ^ d[2], rhotates[4][2]);

	C[0] ^= A[3][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[0][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[2][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[4][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[1][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[3][1] ^ d[1], rhotates[0][1]);
	B[1] = dev_ROL64(A[0][2] ^ d[2], rhotates[1][2]);
	B[2] = dev_ROL64(A[2][3] ^ d[3], rhotates[2][3]);
	B[3] = dev_ROL64(A[4][4] ^ d[4], rhotates[3][4]);
	B[4] = dev_ROL64(A[1][0] ^ d[0], rhotates[4][0]);

	C[0] ^= A[1][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[3][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[0][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[2][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[4][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[2][4] ^ d[4], rhotates[0][4]);
	B[1] = dev_ROL64(A[4][0] ^ d[0], rhotates[1][0]);
	B[2] = dev_ROL64(A[1][1] ^ d[1], rhotates[2][1]);
	B[3] = dev_ROL64(A[3][2] ^ d[2], rhotates[3][2]);
	B[4] = dev_ROL64(A[0][3] ^ d[3], rhotates[4][3]);

	C[0] ^= A[4][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[1][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[3][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[0][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[2][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[1][2] ^ d[2], rhotates[0][2]);
	B[1] = dev_ROL64(A[3][3] ^ d[3], rhotates[1][3]);
	B[2] = dev_ROL64(A[0][4] ^ d[4], rhotates[2][4]);
	B[3] = dev_ROL64(A[2][0] ^ d[0], rhotates[3][0]);
	B[4] = dev_ROL64(A[4][1] ^ d[1], rhotates[4][1]);

	C[0] ^= A[2][0] = B[0] ^ (~B[1] & B[2]);
	C[1] ^= A[4][1] = B[1] ^ (~B[2] & B[3]);
	C[2] ^= A[1][2] = B[2] ^ (~B[3] & B[4]);
	C[3] ^= A[3][3] = B[3] ^ (~B[4] & B[0]);
	C[4] ^= A[0][4] = B[4] ^ (~B[0] & B[1]);

	/* dev_Round 4*n+3 */
	d[0] = dev_ROL64(C[1], 1) ^ C[4];
	d[1] = dev_ROL64(C[2], 1) ^ C[0];
	d[2] = dev_ROL64(C[3], 1) ^ C[1];
	d[3] = dev_ROL64(C[4], 1) ^ C[2];
	d[4] = dev_ROL64(C[0], 1) ^ C[3];

	B[0] =       A[0][0] ^ d[0];/* rotate by 0 */
	B[1] = dev_ROL64(A[0][1] ^ d[1], rhotates[1][1]);
	B[2] = dev_ROL64(A[0][2] ^ d[2], rhotates[2][2]);
	B[3] = dev_ROL64(A[0][3] ^ d[3], rhotates[3][3]);
	B[4] = dev_ROL64(A[0][4] ^ d[4], rhotates[4][4]);

	/* C[0] = */ A[0][0] = B[0] ^ (~B[1] & B[2]) ^ iotas[i + 3];
	/* C[1] = */ A[0][1] = B[1] ^ (~B[2] & B[3]);
	/* C[2] = */ A[0][2] = B[2] ^ (~B[3] & B[4]);
	/* C[3] = */ A[0][3] = B[3] ^ (~B[4] & B[0]);
	/* C[4] = */ A[0][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[1][3] ^ d[3], rhotates[0][3]);
	B[1] = dev_ROL64(A[1][4] ^ d[4], rhotates[1][4]);
	B[2] = dev_ROL64(A[1][0] ^ d[0], rhotates[2][0]);
	B[3] = dev_ROL64(A[1][1] ^ d[1], rhotates[3][1]);
	B[4] = dev_ROL64(A[1][2] ^ d[2], rhotates[4][2]);

	/* C[0] ^= */ A[1][0] = B[0] ^ (~B[1] & B[2]);
	/* C[1] ^= */ A[1][1] = B[1] ^ (~B[2] & B[3]);
	/* C[2] ^= */ A[1][2] = B[2] ^ (~B[3] & B[4]);
	/* C[3] ^= */ A[1][3] = B[3] ^ (~B[4] & B[0]);
	/* C[4] ^= */ A[1][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[2][1] ^ d[1], rhotates[0][1]);
	B[1] = dev_ROL64(A[2][2] ^ d[2], rhotates[1][2]);
	B[2] = dev_ROL64(A[2][3] ^ d[3], rhotates[2][3]);
	B[3] = dev_ROL64(A[2][4] ^ d[4], rhotates[3][4]);
	B[4] = dev_ROL64(A[2][0] ^ d[0], rhotates[4][0]);

	/* C[0] ^= */ A[2][0] = B[0] ^ (~B[1] & B[2]);
	/* C[1] ^= */ A[2][1] = B[1] ^ (~B[2] & B[3]);
	/* C[2] ^= */ A[2][2] = B[2] ^ (~B[3] & B[4]);
	/* C[3] ^= */ A[2][3] = B[3] ^ (~B[4] & B[0]);
	/* C[4] ^= */ A[2][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[3][4] ^ d[4], rhotates[0][4]);
	B[1] = dev_ROL64(A[3][0] ^ d[0], rhotates[1][0]);
	B[2] = dev_ROL64(A[3][1] ^ d[1], rhotates[2][1]);
	B[3] = dev_ROL64(A[3][2] ^ d[2], rhotates[3][2]);
	B[4] = dev_ROL64(A[3][3] ^ d[3], rhotates[4][3]);

	/* C[0] ^= */ A[3][0] = B[0] ^ (~B[1] & B[2]);
	/* C[1] ^= */ A[3][1] = B[1] ^ (~B[2] & B[3]);
	/* C[2] ^= */ A[3][2] = B[2] ^ (~B[3] & B[4]);
	/* C[3] ^= */ A[3][3] = B[3] ^ (~B[4] & B[0]);
	/* C[4] ^= */ A[3][4] = B[4] ^ (~B[0] & B[1]);

	B[0] = dev_ROL64(A[4][2] ^ d[2], rhotates[0][2]);
	B[1] = dev_ROL64(A[4][3] ^ d[3], rhotates[1][3]);
	B[2] = dev_ROL64(A[4][4] ^ d[4], rhotates[2][4]);
	B[3] = dev_ROL64(A[4][0] ^ d[0], rhotates[3][0]);
	B[4] = dev_ROL64(A[4][1] ^ d[1], rhotates[4][1]);

	/* C[0] ^= */ A[4][0] = B[0] ^ (~B[1] & B[2]);
	/* C[1] ^= */ A[4][1] = B[1] ^ (~B[2] & B[3]);
	/* C[2] ^= */ A[4][2] = B[2] ^ (~B[3] & B[4]);
	/* C[3] ^= */ A[4][3] = B[3] ^ (~B[4] & B[0]);
	/* C[4] ^= */ A[4][4] = B[4] ^ (~B[0] & B[1]);
} /* dev_FourRounds */

__device__ void dev_KeccakF1600(unsigned long A[5][5])
{
	size_t i;

	for (i = 0; i < 24; i += 4) {
		dev_FourRounds(A, i);
	}
} /* dev_KeccakF1600 */

#endif /* if defined(KECCAK_REF) */

#ifdef USING_PTX
__device__ static uint64_t BitInterleave(uint64_t Ai)
{
	if (BIT_INTERLEAVE) {
		uint32_t hi = (uint32_t)(Ai >> 32), lo = (uint32_t)Ai;
		uint32_t t0, t1;

		t0 = lo & 0x55555555;
		t0 |= t0 >> 1;  t0 &= 0x33333333;
		t0 |= t0 >> 2;  t0 &= 0x0f0f0f0f;
		t0 |= t0 >> 4;  t0 &= 0x00ff00ff;
		t0 |= t0 >> 8;  t0 &= 0x0000ffff;

		t1 = hi & 0x55555555;
		t1 |= t1 >> 1;  t1 &= 0x33333333;
		t1 |= t1 >> 2;  t1 &= 0x0f0f0f0f;
		t1 |= t1 >> 4;  t1 &= 0x00ff00ff;
		t1 |= t1 >> 8;  t1 <<= 16;

		lo &= 0xaaaaaaaa;
		lo |= lo << 1;  lo &= 0xcccccccc;
		lo |= lo << 2;  lo &= 0xf0f0f0f0;
		lo |= lo << 4;  lo &= 0xff00ff00;
		lo |= lo << 8;  lo >>= 16;

		hi &= 0xaaaaaaaa;
		hi |= hi << 1;  hi &= 0xcccccccc;
		hi |= hi << 2;  hi &= 0xf0f0f0f0;
		hi |= hi << 4;  hi &= 0xff00ff00;
		hi |= hi << 8;  hi &= 0xffff0000;

		Ai = ((uint64_t)(hi | lo) << 32) | (t1 | t0);
	}

	return Ai;
} // BitInterleave

__device__ static uint64_t BitDeinterleave(uint64_t Ai)
{
	if (BIT_INTERLEAVE) {
		uint32_t hi = (uint32_t)(Ai >> 32), lo = (uint32_t)Ai;
		uint32_t t0, t1;

		t0 = lo & 0x0000ffff;
		t0 |= t0 << 8;  t0 &= 0x00ff00ff;
		t0 |= t0 << 4;  t0 &= 0x0f0f0f0f;
		t0 |= t0 << 2;  t0 &= 0x33333333;
		t0 |= t0 << 1;  t0 &= 0x55555555;

		t1 = hi << 16;
		t1 |= t1 >> 8;  t1 &= 0xff00ff00;
		t1 |= t1 >> 4;  t1 &= 0xf0f0f0f0;
		t1 |= t1 >> 2;  t1 &= 0xcccccccc;
		t1 |= t1 >> 1;  t1 &= 0xaaaaaaaa;

		lo >>= 16;
		lo |= lo << 8;  lo &= 0x00ff00ff;
		lo |= lo << 4;  lo &= 0x0f0f0f0f;
		lo |= lo << 2;  lo &= 0x33333333;
		lo |= lo << 1;  lo &= 0x55555555;

		hi &= 0xffff0000;
		hi |= hi >> 8;  hi &= 0xff00ff00;
		hi |= hi >> 4;  hi &= 0xf0f0f0f0;
		hi |= hi >> 2;  hi &= 0xcccccccc;
		hi |= hi >> 1;  hi &= 0xaaaaaaaa;

		Ai = ((uint64_t)(hi | lo) << 32) | (t1 | t0);
	}

	return Ai;
} // BitDeinterleave
#endif // ifdef USING_PTX

__device__ size_t dev_SHA3_absorb(unsigned long A[5][5], const unsigned char *inp, size_t len,
				  size_t r)
{
	unsigned long *A_flat = (unsigned long *)A;
	size_t i, w = r / 8;

	// assert(r < (25 * sizeof(A[0][0])) && (r % 8) == 0);

	while (len >= r) {
		for (i = 0; i < w; i++) {
		#ifdef USING_INTEGER
			unsigned long Ai;
			memcpy(&Ai, &inp[0], 8);
		#else // ifdef USING_INTEGER
			unsigned long Ai = (unsigned long)inp[0]   | (unsigned long)inp[1] << 8  |
					   (unsigned long)inp[2] << 16 | (unsigned long)inp[3] << 24 |
					   (unsigned long)inp[4] << 32 | (unsigned long)inp[5] << 40 |
					   (unsigned long)inp[6] << 48 | (unsigned long)inp[7] << 56;
		#endif // ifdef USING_INTEGER
			inp += 8;
			#ifdef USING_PTX
			A_flat[i] ^= BitInterleave(Ai);
			#else // ifdef USING_PTX
			A_flat[i] ^= Ai;
			#endif // ifdef USING_PTX
		}
		dev_KeccakF1600(A);
		len -= r;
	}

	return len;
} /* SHA3_absorb */

/*
 * sha3_squeeze is called once at the end to generate |out| hash value
 * of |len| bytes.
 */
__device__ void dev_SHA3_squeeze(unsigned long A[5][5], unsigned char *out, size_t len, size_t r)
{
	unsigned long *A_flat = (unsigned long *)A;
	int i, w = r / 8;

	// assert(r < (25 * sizeof(A[0][0])) && (r % 8) == 0);
	// printf("??\n");

	while (len != 0) {
		for (i = 0; i < w && len != 0; i++) {
			#ifdef USING_PTX
			uint64_t Ai = BitDeinterleave(A_flat[i]);
			#else // ifdef USING_PTX
			unsigned long Ai = A_flat[i];
			#endif // ifdef USING_PTX

			if (len < 8) {
// #ifdef USING_INTEGER
// 				memcpy(out, &Ai, len); Make a compilation exception
// #else // ifdef USING_INTEGER
				for (i = 0; i < len; i++) {
					*out++ = Ai;
					Ai >>= 8;
				}
// #endif // ifdef USING_INTEGER
				return;
			}
// #ifdef USING_INTEGER
// 			memcpy(out, &Ai, 8); Make a compilation exception
// #else // ifdef USING_INTEGER
			out[0] = (Ai);
			out[1] = (Ai >> 8);
			out[2] = (Ai >> 16);
			out[3] = (Ai >> 24);
			out[4] = (Ai >> 32);
			out[5] = (Ai >> 40);
			out[6] = (Ai >> 48);
			out[7] = (Ai >> 56);
// #endif // ifdef USING_INTEGER
			out += 8;
			len -= 8;
		}
		if (len)
			dev_KeccakF1600(A);
	}
} /* SHA3_squeeze */

# define SHA3_BLOCKSIZE(bitlen)

__device__ int dev_ossl_sha3_init(KECCAK1600_CTX *ctx, unsigned char pad,
				  size_t bitlen, size_t outbitlen)
{
	memset(ctx->A, 0, sizeof(ctx->A));
	ctx->bufsz = 0;
	ctx->block_size = (KECCAK1600_WIDTH - bitlen * 2) / 8;
	ctx->md_size = outbitlen / 8;
	ctx->pad = pad;
	return 1;
} /* ossl_sha3_init */


__device__ int dev_ossl_sha3_update(KECCAK1600_CTX *ctx, const void *_inp,
				    size_t len)
{
	const unsigned char *inp = (const unsigned char *)_inp;
	size_t bsz = ctx->block_size;
	size_t num, rem;

	if (len == 0) {
		return 1;
	}

	if ((num = ctx->bufsz) != 0) {  /* process intermediate buffer? */
		rem = bsz - num;

		if (len < rem) {
			memcpy(ctx->buf + num, inp, len);
			ctx->bufsz += len;
			return 1;
		}
		/*
		 * We have enough data to fill or overflow the intermediate
		 * buffer. So we append |rem| bytes and process the block,
		 * leaving the rest for later processing...
		 */
		memcpy(ctx->buf + num, inp, rem);
		inp += rem, len -= rem;
		(void)dev_SHA3_absorb(ctx->A, ctx->buf, bsz, bsz);
		ctx->bufsz = 0;
		/* ctx->buf is processed, ctx->num is guaranteed to be zero */
	}

	if (len >= bsz)
		rem = dev_SHA3_absorb(ctx->A, inp, len, bsz);
	else
		rem = len;

	if (rem) {
		memcpy(ctx->buf, inp + len - rem, rem);
		ctx->bufsz = rem;
	}

	return 1;
} /* ossl_sha3_update */

__device__ int dev_ossl_sha3_final(unsigned char *md, KECCAK1600_CTX *ctx)
{
	size_t bsz = ctx->block_size;
	size_t num = ctx->bufsz;

	if (ctx->md_size == 0) {
		return 1;
	}

	/*
	 * Pad the data with 10*1. Note that |num| can be |bsz - 1|
	 * in which case both byte operations below are performed on
	 * same byte...
	 */
	memset(ctx->buf + num, 0, bsz - num);
	ctx->buf[num] = ctx->pad;
	ctx->buf[bsz - 1] |= 0x80;

	(void)dev_SHA3_absorb(ctx->A, ctx->buf, bsz, bsz);

	dev_SHA3_squeeze(ctx->A, md, ctx->md_size, bsz);

	return 1;
} /* ossl_sha3_final */

__device__ void dev_SHA3_sponge(const unsigned char *inp, size_t len,
				unsigned char *out, unsigned char pad,
				size_t bitlen, size_t outbitlen)
{
	KECCAK1600_CTX ctx;

	dev_ossl_sha3_init(&ctx, pad, bitlen, outbitlen);
	dev_ossl_sha3_update(&ctx, inp, len);
	dev_ossl_sha3_final(out, &ctx);
} // dev_SHA3_sponge
