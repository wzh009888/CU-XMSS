#define SHAKE128_RATE 168
#define SHAKE256_RATE 136

// __device__ void dev_shake128(unsigned char *out, unsigned long long outlen,
// 		  const unsigned char *in, unsigned long long inlen);
// __global__ void global_shake128(unsigned char *out, unsigned long long outlen,
// 		     const unsigned char *in, unsigned long long inlen);

__device__ void dev_shake256_origin(unsigned char *out, unsigned long long outlen,
		  const unsigned char *in, unsigned long long inlen);
__global__ void global_shake256_origin(unsigned char *out, unsigned long long outlen,
		     const unsigned char *in, unsigned long long inlen);
