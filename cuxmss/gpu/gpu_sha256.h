#include "gpu_common.h"

# define SHA256_DIGEST_LENGTH       32
# define SHA_LBLOCK                 16
#define HASH_CBLOCK                 (SHA_LBLOCK * 4)

typedef struct self_SHA256state_st {
	u32 h[8];
	u32 Nl, Nh;
	u32 data[SHA_LBLOCK];
	u32 num, md_len;
} self_SHA256_CTX;

__device__ int dev_SHA256_Init(self_SHA256_CTX *c);
__device__ int dev_SHA256_Update(self_SHA256_CTX *c,
				 const void *data_, size_t len);
__device__ int dev_SHA256_Final(u8 *md, self_SHA256_CTX *c);

__device__ void dev_SHA256(const u8 *d, size_t n, u8 *md);

__global__ void global_SHA256(const u8 *d, size_t n, u8 *md);

__global__ void global_parallel_SHA256(const u8 *msg, u8 *md,
				       size_t si, size_t n);
