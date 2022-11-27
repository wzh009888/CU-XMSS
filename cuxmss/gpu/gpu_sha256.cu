#include "gpu_sha256.h"

// #define FASTER // right but slower

u32 this_K256[64] = {
	0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL,
	0x3956c25bUL, 0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL,
	0xd807aa98UL, 0x12835b01UL, 0x243185beUL, 0x550c7dc3UL,
	0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL, 0xc19bf174UL,
	0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL,
	0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL,
	0x983e5152UL, 0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL,
	0xc6e00bf3UL, 0xd5a79147UL, 0x06ca6351UL, 0x14292967UL,
	0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL, 0x53380d13UL,
	0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
	0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL,
	0xd192e819UL, 0xd6990624UL, 0xf40e3585UL, 0x106aa070UL,
	0x19a4c116UL, 0x1e376c08UL, 0x2748774cUL, 0x34b0bcb5UL,
	0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL, 0x682e6ff3UL,
	0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL,
	0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL
};

__constant__ u32 K256[64];

__device__ int dev_SHA256_Init(self_SHA256_CTX *c)
{
	memset(c, 0, sizeof(*c));
	c->h[0] = 0x6a09e667UL;
	c->h[1] = 0xbb67ae85UL;
	c->h[2] = 0x3c6ef372UL;
	c->h[3] = 0xa54ff53aUL;
	c->h[4] = 0x510e527fUL;
	c->h[5] = 0x9b05688cUL;
	c->h[6] = 0x1f83d9abUL;
	c->h[7] = 0x5be0cd19UL;
	c->md_len = SHA256_DIGEST_LENGTH;
	return 1;
} // dev_SHA256_Init

#ifdef USING_INTEGER
#define HOST_c2l(c, l) (l = __byte_perm(*(c++), 0, 0x0123))
#else // ifdef USING_INTEGER
# define HOST_c2l(c, l)  (l = (((unsigned long)(*((c)++))) << 24),          \
			  l |= (((unsigned long)(*((c)++))) << 16),          \
			  l |= (((unsigned long)(*((c)++))) << 8),          \
			  l |= (((unsigned long)(*((c)++)))))
#endif // ifdef USING_INTEGER

# define HOST_l2c(l, c)  (*((c)++) = (u8)(((l) >> 24) & 0xff),      \
			  *((c)++) = (u8)(((l) >> 16) & 0xff),      \
			  *((c)++) = (u8)(((l) >> 8) & 0xff),      \
			  *((c)++) = (u8)(((l)) & 0xff),      \
			  l)

#ifdef USING_PTX
__forceinline__ __device__ static u32 ROTATE(u32 v, u32 n)
{
	asm ("shf.l.clamp.b32 %0, %0, %0, %1;\n\t" \
	     : "+r" (v) \
	     : "r" (n));
	return v;
} // ROTATE
#else // ifdef USING_PTX
# define ROTATE(v, n) (((v) << (n)) | ((v) >> (32 - (n))))
#endif // ifdef USING_PTX

# define Sigma0(x)       (ROTATE((x), 30) ^ ROTATE((x), 19) ^ ROTATE((x), 10))
# define Sigma1(x)       (ROTATE((x), 26) ^ ROTATE((x), 21) ^ ROTATE((x), 7))
# define sigma0(x)       (ROTATE((x), 25) ^ ROTATE((x), 14) ^ ((x) >> 3))
# define sigma1(x)       (ROTATE((x), 15) ^ ROTATE((x), 13) ^ ((x) >> 10))

// # define Ch(x, y, z)       (((x) & (y)) ^ ((~(x)) & (z)))
// # define Maj(x, y, z)      (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
# define Ch(x, y, z)     ((z) ^ ((x) & ((y) ^ (z))))
# define Maj(x, y, z)    (((y) & ((x) | (z))) | ((x) & (z)))

#ifdef FASTER

#  define ROUND_00_15(i, a, b, c, d, e, f, g, h)          do {    \
		T1 += h + Sigma1(e) + Ch(e, f, g) + K256[i];      \
		h = Sigma0(a) + Maj(a, b, c);                     \
		d += T1;        h += T1; } while (0)

#  define ROUND_16_63(i, a, b, c, d, e, f, g, h, X)        do {    \
		s0 = X[(i + 1) & 0x0f];     s0 = sigma0(s0);        \
		s1 = X[(i + 14) & 0x0f];    s1 = sigma1(s1);        \
		T1 = X[(i) & 0x0f] += s0 + s1 + X[(i + 9) & 0x0f];    \
		ROUND_00_15(i, a, b, c, d, e, f, g, h); } while (0)

__device__ void sha256_block_data_order(self_SHA256_CTX *ctx, const void *in,
					size_t num)
{
	u32 a, b, c, d, e, f, g, h, s0, s1, T1;
	u32 X[16];
	int i;
	const u8 *data = (const u8 *)in;

	// const u32 *data = (const u32 *)in;

	while (num--) {
		a = ctx->h[0];
		b = ctx->h[1];
		c = ctx->h[2];
		d = ctx->h[3];
		e = ctx->h[4];
		f = ctx->h[5];
		g = ctx->h[6];
		h = ctx->h[7];

		u32 l;

		(void)HOST_c2l(data, l);
		T1 = X[0] = l;
		ROUND_00_15(0, a, b, c, d, e, f, g, h);
		(void)HOST_c2l(data, l);
		T1 = X[1] = l;
		ROUND_00_15(1, h, a, b, c, d, e, f, g);
		(void)HOST_c2l(data, l);
		T1 = X[2] = l;
		ROUND_00_15(2, g, h, a, b, c, d, e, f);
		(void)HOST_c2l(data, l);
		T1 = X[3] = l;
		ROUND_00_15(3, f, g, h, a, b, c, d, e);
		(void)HOST_c2l(data, l);
		T1 = X[4] = l;
		ROUND_00_15(4, e, f, g, h, a, b, c, d);
		(void)HOST_c2l(data, l);
		T1 = X[5] = l;
		ROUND_00_15(5, d, e, f, g, h, a, b, c);
		(void)HOST_c2l(data, l);
		T1 = X[6] = l;
		ROUND_00_15(6, c, d, e, f, g, h, a, b);
		(void)HOST_c2l(data, l);
		T1 = X[7] = l;
		ROUND_00_15(7, b, c, d, e, f, g, h, a);
		(void)HOST_c2l(data, l);
		T1 = X[8] = l;
		ROUND_00_15(8, a, b, c, d, e, f, g, h);
		(void)HOST_c2l(data, l);
		T1 = X[9] = l;
		ROUND_00_15(9, h, a, b, c, d, e, f, g);
		(void)HOST_c2l(data, l);
		T1 = X[10] = l;
		ROUND_00_15(10, g, h, a, b, c, d, e, f);
		(void)HOST_c2l(data, l);
		T1 = X[11] = l;
		ROUND_00_15(11, f, g, h, a, b, c, d, e);
		(void)HOST_c2l(data, l);
		T1 = X[12] = l;
		ROUND_00_15(12, e, f, g, h, a, b, c, d);
		(void)HOST_c2l(data, l);
		T1 = X[13] = l;
		ROUND_00_15(13, d, e, f, g, h, a, b, c);
		(void)HOST_c2l(data, l);
		T1 = X[14] = l;
		ROUND_00_15(14, c, d, e, f, g, h, a, b);
		(void)HOST_c2l(data, l);
		T1 = X[15] = l;
		ROUND_00_15(15, b, c, d, e, f, g, h, a);

		#pragma unroll 6
		for (i = 16; i < 64; i += 8) {
			ROUND_16_63(i + 0, a, b, c, d, e, f, g, h, X);
			ROUND_16_63(i + 1, h, a, b, c, d, e, f, g, X);
			ROUND_16_63(i + 2, g, h, a, b, c, d, e, f, X);
			ROUND_16_63(i + 3, f, g, h, a, b, c, d, e, X);
			ROUND_16_63(i + 4, e, f, g, h, a, b, c, d, X);
			ROUND_16_63(i + 5, d, e, f, g, h, a, b, c, X);
			ROUND_16_63(i + 6, c, d, e, f, g, h, a, b, X);
			ROUND_16_63(i + 7, b, c, d, e, f, g, h, a, X);
		}

		ctx->h[0] += a;
		ctx->h[1] += b;
		ctx->h[2] += c;
		ctx->h[3] += d;
		ctx->h[4] += e;
		ctx->h[5] += f;
		ctx->h[6] += g;
		ctx->h[7] += h;
	}
} // sha256_block_data_order

#else   // ifdef FASTER

__device__ void sha256_block_data_order(self_SHA256_CTX *ctx, const void *in,
					size_t num)
{
	u32 a, b, c, d, e, f, g, h, s0, s1, T1, T2;
	u32 X[16], l;
	int i;

#ifdef USING_INTEGER
	const u32 *data = (const u32 *)in;
#else // ifdef USING_INTEGER
	// u8 dd[4];
	// u8 *ddd = (u8 *)dd;
	const u8 *data = (const u8 *)in;
#endif // ifdef USING_INTEGER

	while (num--) {
		a = ctx->h[0];
		b = ctx->h[1];
		c = ctx->h[2];
		d = ctx->h[3];
		e = ctx->h[4];
		f = ctx->h[5];
		g = ctx->h[6];
		h = ctx->h[7];

#ifdef USING_UNROLL
		#pragma unroll
#endif // ifdef USING_UNROLL
		for (i = 0; i < 16; i++) {
			// memcpy(dd, data, 4);
			// l = (u32)dd[0] << 24 | (u32)dd[1] << 16 |
			// 	(u32)dd[2] << 8  | (u32)dd[3];
			// data += 4;
			(void)HOST_c2l(data, l);
			T1 = X[i] = l;
			T1 += h + Sigma1(e) + Ch(e, f, g) + K256[i];
			T2 = Sigma0(a) + Maj(a, b, c);
			h = g;
			g = f;
			f = e;
			e = d + T1;
			d = c;
			c = b;
			b = a;
			a = T1 + T2;
		}

#ifdef USING_UNROLL
		#pragma unroll
#endif // ifdef USING_UNROLL
		for (i = 16; i < 64; i++) {
			s0 = X[(i + 1) & 0x0f];
			s0 = sigma0(s0);
			s1 = X[(i + 14) & 0x0f];
			s1 = sigma1(s1);

			T1 = X[i & 0xf] += s0 + s1 + X[(i + 9) & 0xf];
			T1 += h + Sigma1(e) + Ch(e, f, g) + K256[i];
			T2 = Sigma0(a) + Maj(a, b, c);
			h = g;
			g = f;
			f = e;
			e = d + T1;
			d = c;
			c = b;
			b = a;
			a = T1 + T2;
		}

		ctx->h[0] += a;
		ctx->h[1] += b;
		ctx->h[2] += c;
		ctx->h[3] += d;
		ctx->h[4] += e;
		ctx->h[5] += f;
		ctx->h[6] += g;
		ctx->h[7] += h;
	}
} // sha256_block_data_order
#endif // ifdef FASTER

__device__ int dev_SHA256_Update(self_SHA256_CTX *c, const void *data_, size_t len)
{
	const u8 *data = (const u8 *)data_;
	u8 *p;
	u32 l;
	size_t n;

	if (len == 0)
		return 1;

	l = (c->Nl + (((u32)len) << 3)) & 0xffffffffUL;
	if (l < c->Nl)                          /* overflow */
		c->Nh++;
	c->Nh += (u32)(len >> 29);              /* might cause compiler warning on
	                                         * 16-bit */
	c->Nl = l;

	n = c->num;
	if (n != 0) {
		p = (u8 *)c->data;

		if (len >= HASH_CBLOCK || len + n >= HASH_CBLOCK) {
			memcpy(p + n, data, HASH_CBLOCK - n);
			sha256_block_data_order(c, p, 1);
			n = HASH_CBLOCK - n;
			data += n;
			len -= n;
			c->num = 0;
			/*
			 * We use memset rather than OPENSSL_cleanse() here deliberately.
			 * Using OPENSSL_cleanse() here could be a performance issue. It
			 * will get properly cleansed on finalisation so this isn't a
			 * security problem.
			 */
			memset(p, 0, HASH_CBLOCK); /* keep it zeroed */
		} else {
			memcpy(p + n, data, len);
			c->num += (u32)len;
			return 1;
		}
	}

	n = len / HASH_CBLOCK;
	if (n > 0) {
		sha256_block_data_order(c, data, n);
		n *= HASH_CBLOCK;
		data += n;
		len -= n;
	}

	if (len != 0) {
		p = (u8 *)c->data;
		c->num = (u32)len;
		memcpy(p, data, len);
	}
	return 1;
} // dev_SHA256_Update

#define HASH_MAKE_STRING(c, s)   do {    \
		unsigned long ll;               \
		u32 nn;               \
		for (nn = 0; nn < SHA256_DIGEST_LENGTH / 4; nn++)       \
		{ ll = (c)->h[nn]; (void)HOST_l2c(ll, (s)); }  \
} while (0)

__device__ int dev_SHA256_Final(u8 *md, self_SHA256_CTX *c)
{
	u8 *p = (u8 *)c->data;
	size_t n = c->num;

	p[n] = 0x80;            /* there is always room for one */
	n++;

	if (n > (HASH_CBLOCK - 8)) {
		memset(p + n, 0, HASH_CBLOCK - n);
		n = 0;
		sha256_block_data_order(c, p, 1);
	}
	memset(p + n, 0, HASH_CBLOCK - 8 - n);

	p += HASH_CBLOCK - 8;
	(void)HOST_l2c(c->Nh, p);
	(void)HOST_l2c(c->Nl, p);
	p -= HASH_CBLOCK;
	sha256_block_data_order(c, p, 1);
	c->num = 0;
	HASH_MAKE_STRING(c, md);

	return 1;
} // dev_SHA256_Final

__device__ void dev_SHA256(const u8 *d, size_t n, u8 *md)
{
	self_SHA256_CTX c;
	static u8 m[SHA256_DIGEST_LENGTH];

	if (md == NULL)
		md = m;
	dev_SHA256_Init(&c);
	dev_SHA256_Update(&c, d, n);
	dev_SHA256_Final(md, &c);
} // dev_SHA256

__global__ void global_SHA256(const u8 *d, size_t n, u8 *md)
{
	dev_SHA256(d, n, md);
} // global_SHA256

__global__ void global_parallel_SHA256(const u8 *msg, u8 *md,
				       size_t size, size_t n)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	self_SHA256_CTX c;
	u8 m[SHA256_DIGEST_LENGTH]; // static can't be used

	if (tid < n) {
		dev_SHA256_Init(&c);
		dev_SHA256_Update(&c, msg + tid * size, size);
		dev_SHA256_Final(m, &c);

		memcpy(md + tid * SHA256_DIGEST_LENGTH, m, SHA256_DIGEST_LENGTH);
	}
} // global_parallel_SHA256
