#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#include "gpu_common.h"

# define KECCAK1600_WIDTH 1600

typedef struct keccak_st KECCAK1600_CTX;

struct keccak_st {
	unsigned long A[5][5];
	size_t block_size;      /* cached ctx->digest->block_size */
	size_t md_size;         /* output length, variable in XOF */
	size_t bufsz;           /* used bytes in below buffer */
	unsigned char buf[KECCAK1600_WIDTH / 8 - 32];
	unsigned char pad;
};

// __device__ void dev_shake128(unsigned char *out, unsigned long long outlen,
// 			     const unsigned char *in, unsigned long long inlen);
// __global__ void global_shake128(unsigned char *out, unsigned long long outlen,
// 				const unsigned char *in, unsigned long long inlen);

// __device__ int dev_ossl_sha3_init(KECCAK1600_CTX *ctx, unsigned char pad,
// 				  size_t bitlen, size_t outbitlen);
// __device__ int dev_ossl_sha3_update(KECCAK1600_CTX *ctx, const void *_inp,
// 	size_t len);
// __device__ int dev_ossl_sha3_final(unsigned char *md, KECCAK1600_CTX *ctx);

__device__ void dev_shake256(unsigned char *output, int outputByteLen,
			     const unsigned char *input, unsigned int inputByteLen);
__global__ void global_shake256(unsigned char *output, int outputByteLen,
				const unsigned char *input, unsigned int inputByteLen);

__global__ void global_parallel_shake256(const u8 *msg, u8 *md,
					 size_t in_size, size_t out_size, size_t n);
