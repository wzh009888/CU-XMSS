#include "gpu_hash.h"
#include "gpu_sha256.h"
#include "gpu_sha512.h"
#include "gpu_shake.h"
#include "gpu_shake_origin.h"
#include "gpu_utils.h"
#include "gpu_hash_address.h"

#include <iostream>
using namespace std;

#define XMSS_HASH_PADDING_F 0
#define XMSS_HASH_PADDING_H 1
#define XMSS_HASH_PADDING_HASH 2
#define XMSS_HASH_PADDING_PRF 3
#define XMSS_HASH_PADDING_PRF_KEYGEN 4

__device__ void dev_addr_to_bytes(u8 *bytes, const u32 addr[8])
{
	int i;

	for (i = 0; i < 8; i++)
		dev_ull_to_bytes(bytes + i * 4, 4, addr[i]);
} // dev_addr_to_bytes

__device__ int dev_core_hash(u8 *out, const u8 *in, u64 inlen)
{
	u8 buf[64];

	if (N == 24 && FUNC == XMSS_SHA2) {
		dev_SHA256(in, inlen, buf);
		memcpy(out, buf, 24);
	} else if (N == 24 && FUNC == XMSS_SHAKE256) {
		dev_shake256(out, 24, in, inlen);
	} else if (N == 32 && FUNC == XMSS_SHA2) {
		dev_SHA256(in, inlen, out);
	} else if (N == 32 && FUNC == XMSS_SHAKE256) {
		dev_shake256(out, 32, in, inlen);
		// dev_shake256_origin(out, 32, in, inlen);
	} else if (N == 64 && FUNC == XMSS_SHA2) {
		dev_SHA512(in, inlen, out);
	} else if (N == 64 && FUNC == XMSS_SHAKE256) {
		dev_shake256(out, 64, in, inlen);
	} else {
		return -1;
	}

	return 0;
} // dev_core_hash

/* to test the correct of sha algorithm */
void face_sha256(const u8 *d, u64 n, u8 *md)
{
	CHECK(cudaSetDevice(DEVICE_USED));
	u8 *dev_d = NULL, *dev_md = NULL;

	CHECK(cudaMalloc((void **)&dev_d, n * sizeof(u8)));
	CHECK(cudaMemcpy(dev_d, d, n * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_md, 32 * sizeof(u8)));

	CHECK(cudaDeviceSynchronize());

	global_SHA256 << < 1, 1 >> > (dev_d, n, dev_md);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(md, dev_md, 32 * sizeof(u8), DEVICE_2_HOST));

	cudaFree(dev_d); cudaFree(dev_md);
} // face_sha_test

// 所有消息一样的长度，提供单个消息的长度和消息数
// msg 和 md 表示消息和摘要的起始地址
// size表示单个消息的长度，n表示消息数
void face_dp_sha256(const u8 *msg, u8 *md, u64 s_msg_B,
		    u64 msg_num, u64 blocks, u64 threads)
{
	struct timespec b1, e1, b2, e2;
	double res1, res2;
	u64 t0, t1;
	u8 *dev_msg = NULL, *dev_md = NULL;
	int device = DEVICE_USED;
	cudaDeviceProp deviceProp;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);

	CHECK(cudaMalloc((void **)&dev_msg, s_msg_B * msg_num * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_md, 32 * msg_num * sizeof(u8)));

	// 最快情况 32: 120654 64: 121967 128: 83347
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &b2);

	CHECK(cudaMemcpy(dev_msg, msg, s_msg_B * msg_num * sizeof(u8), HOST_2_DEVICE));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &b1);
	t0 = cpucycles();

	CHECK(cudaDeviceSynchronize());
	global_parallel_SHA256 << < blocks, threads >> >
		(dev_msg, dev_md, s_msg_B, msg_num);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &e1);
	t1 = cpucycles();

	CHECK(cudaMemcpy(md, dev_md, 32 * msg_num * sizeof(u8), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &e2);
	res1 = (e1.tv_sec - b1.tv_sec) * 1e6 + (e1.tv_nsec - b1.tv_nsec) / 1e3;
	res2 = (e2.tv_sec - b2.tv_sec) * 1e6 + (e2.tv_nsec - b2.tv_nsec) / 1e3;
	printf("core sha256\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       res1, msg_num * s_msg_B / res1,
	       t1 - t0, (t1 - t0) * 1.0 / (msg_num * s_msg_B));
	printf("core+cpy sha256\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       res2, msg_num * s_msg_B / res2,
	       t1 - t0, (t1 - t0) * 1.0 / (msg_num * s_msg_B));


	cudaFree(dev_msg); cudaFree(dev_md);
} // face_dp_sha256

void face_msdp_sha256(const u8 *msg, u8 *md, u64 size,
		      u64 msg_N, u64 blocks, u64 threads)
{
	struct timespec start, stop;
	double result;
	u64 t0, t1;
	int device = DEVICE_USED;
	cudaDeviceProp deviceProp;

	int stream_N = 16;
	cudaStream_t stream[stream_N];
	unsigned char *data_in[stream_N];
	unsigned char *data_out[stream_N];

	for (int i = 0; i < stream_N; ++i) {
		CHECK(cudaStreamCreate(&stream[i]));
		CHECK(cudaMalloc((void **)&data_in[i], size * msg_N / stream_N * sizeof(u8)));
		CHECK(cudaMalloc((void **)&data_out[i], 32 * msg_N / stream_N * sizeof(u8)));
	}

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();

	CHECK(cudaDeviceSynchronize());
	for (size_t i = 0; i < stream_N; i++) {
		u64 in_len = size * msg_N * sizeof(u8);
		u64 out_len = 32 * msg_N * sizeof(u8);
		CHECK(cudaMemcpyAsync(data_in[i], &msg[in_len / stream_N * i],
				      in_len / stream_N,  HOST_2_DEVICE, stream[i]));
		global_parallel_SHA256 << < blocks, threads / stream_N, 0, stream[i] >> >
			(data_in[i], data_out[i], size, msg_N / stream_N);
		CHECK(cudaMemcpyAsync(&md[out_len / stream_N * i], data_out[i],
				      out_len / stream_N, DEVICE_2_HOST, stream[i]));
	}
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("core sha256\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       result, msg_N * size / result,
	       t1 - t0, (t1 - t0) * 1.0 / (msg_N * size));

	for (int i = 0; i < stream_N; ++i) {
		CHECK(cudaFree(data_in[i]));
		CHECK(cudaFree(data_out[i]));
	}

} // face_msdp_sha256

void face_sha512(const u8 *d, u64 n, u8 *md)
{
	CHECK(cudaSetDevice(DEVICE_USED));
	u8 *dev_d = NULL, *dev_md = NULL;

	CHECK(cudaMalloc((void **)&dev_d, n * sizeof(u8)));
	CHECK(cudaMemcpy(dev_d, d, n * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_md, 64 * sizeof(u8)));

	CHECK(cudaDeviceSynchronize());

	global_SHA512 << < 1, 1 >> > (dev_d, n, dev_md);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(md, dev_md, 64 * sizeof(u8), DEVICE_2_HOST));

	cudaFree(dev_d); cudaFree(dev_md);
} // face_sha512

// 所有消息一样的长度，提供单个消息的长度和消息数
// msg 和 md 表示消息和摘要的起始地址
// size表示单个消息的长度，n表示消息数
void face_parallel_sha512(const u8 *msg, u8 *md, u64 size, u64 n)
{
	u64 t0, t1;
	struct timespec start, stop;
	double result;
	u8 *dev_msg = NULL, *dev_md = NULL;

	CHECK(cudaSetDevice(DEVICE_USED));

	CHECK(cudaMalloc((void **)&dev_msg, size * n * sizeof(u8)));
	CHECK(cudaMemcpy(dev_msg, msg, size * n * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_md, 64 * n * sizeof(u8)));

	#define SHA5_N 128 // 最快情况 32: 151095 64: 151537 128: 151568
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();

	CHECK(cudaDeviceSynchronize());
	global_parallel_SHA512 << < n / SHA5_N, SHA5_N >> >
		(dev_msg, dev_md, size, n);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("core sha512\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       result, n * size / result, t1 - t0, (t1 - t0) * 1.0 / (n * size));

	CHECK(cudaMemcpy(md, dev_md, 64 * n * sizeof(u8), DEVICE_2_HOST));

	cudaFree(dev_msg); cudaFree(dev_md);
} // face_parallel_sha256

void face_shake256(u8 *md, u64 outlen, const u8 *d, u64 inlen)
{
	CHECK(cudaSetDevice(DEVICE_USED));
	u8 *dev_d = NULL, *dev_md = NULL;

	CHECK(cudaMalloc((void **)&dev_d, inlen * sizeof(u8)));
	CHECK(cudaMemcpy(dev_d, d, inlen * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_md, outlen * sizeof(u8)));

	CHECK(cudaDeviceSynchronize());

	global_shake256 << < 1, 1 >> > (dev_md, outlen, dev_d, inlen);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(md, dev_md, outlen * sizeof(u8), DEVICE_2_HOST));

	cudaFree(dev_d); cudaFree(dev_md);
} // face_shake128

// 所有消息一样的长度，提供单个消息的长度和消息数
// msg 和 md 表示消息和摘要的起始地址
// in_size表示单个消息的长度，out_size表示单个消息摘要的长度
// n表示消息数
void face_parallel_shake256(const u8 *msg, u8 *md, u64 in_size, u64 out_size,
			    u64 msg_num, u64 blocks, u64 threads)
{
	u64 t0, t1;
	struct timespec start, stop;
	double result;
	u8 *dev_msg = NULL, *dev_md = NULL;
	int device = DEVICE_USED;

	CHECK(cudaSetDevice(device));

	CHECK(cudaMalloc((void **)&dev_msg, in_size * msg_num * sizeof(u8)));
	CHECK(cudaMemcpy(dev_msg, msg, in_size * msg_num * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_md, out_size * msg_num * sizeof(u8)));

	// 32:129172 64:124959 128:122831
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();

	CHECK(cudaDeviceSynchronize());
	global_parallel_shake256 << < blocks, threads >> >
		(dev_msg, dev_md, in_size, out_size, msg_num);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("core shake\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       result, (double)msg_num * in_size / result, t1 - t0,
	       (t1 - t0) * 1.0 / msg_num / in_size);

	CHECK(cudaMemcpy(md, dev_md, out_size * msg_num * sizeof(u8), DEVICE_2_HOST));

	cudaFree(dev_msg); cudaFree(dev_md);
} // face_parallel_SHAKE256

//Assume that the thread size is less than 65536
__device__ int init_a[65536];
__device__ self_SHA256_CTX ss_1[65536];

#if N == 32 && FUNC == XMSS_SHA2 && defined(USING_PRE_COMP)
__device__ int dev_prf(u8 *out, const u8 in[32], const u8 *key)
{
	unsigned char buf[2 * 32 + 32];
	self_SHA256_CTX state2;

	// const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (init_a[blockIdx.x]) {
		dev_SHA256_Init(&state2);
		memcpy(state2.h, ss_1[blockIdx.x].h, 40);
		memcpy(state2.data, in, 32);
		state2.num = 32;
	} else {
		dev_SHA256_Init(&ss_1[blockIdx.x]);

		dev_ull_to_bytes(buf, PADDING_LEN, XMSS_HASH_PADDING_PRF);
		memcpy(buf + PADDING_LEN, key, N);
		memcpy(buf + PADDING_LEN + N, in, 32);

		dev_SHA256_Update(&ss_1[blockIdx.x], buf, 2 * N + 32);

		dev_SHA256_Init(&state2);
		memcpy(state2.h, ss_1[blockIdx.x].h, 40);
		memcpy(state2.data, buf + 2 * N, 32);
		state2.num = 32;

		init_a[blockIdx.x] = 1;
	}
	dev_SHA256_Final(out, &state2);
	return 1;
} // dev_prf
#else   // if N == 32 && FUNC == XMSS_SHA2
__device__ int dev_prf(u8 *out, const u8 in[32], const u8 *key)
{
	u8 buf[PADDING_LEN + N + 32];

	dev_ull_to_bytes(buf, PADDING_LEN, XMSS_HASH_PADDING_PRF);
	memcpy(buf + PADDING_LEN, key, N);
	memcpy(buf + PADDING_LEN + N, in, 32);

	return dev_core_hash(out, buf, PADDING_LEN + N + 32);
} // dev_prf
#endif // if N == 32 && FUNC == XMSS_SHA2

__device__ int dev_prf_keygen(u8 *out, const u8 *in, const u8 *key)
{
	u8 buf[PADDING_LEN + 2 * N + 32];

	dev_ull_to_bytes(buf, PADDING_LEN, XMSS_HASH_PADDING_PRF_KEYGEN);
	memcpy(buf + PADDING_LEN, key, N);
	memcpy(buf + PADDING_LEN + N, in, N + 32);

	return dev_core_hash(out, buf, PADDING_LEN + 2 * N + 32);
} // prf_keygen

__device__ int dev_hash_message(u8 *out, const u8 *R, const u8 *root,
				u64 idx, u8 *m_with_prefix, u64 mlen)
{
	/* We're creating a hash using input of the form:
	 * toByte(X, 32) || R || root || index || M */

	// dev_ull_to_bytes(m_with_prefix, PADDING_LEN, XMSS_HASH_PADDING_HASH);
	// memcpy(m_with_prefix + PADDING_LEN, R, N);
	// memcpy(m_with_prefix + PADDING_LEN + N, root, N);
	// dev_ull_to_bytes(m_with_prefix + PADDING_LEN + 2 * N, N, idx);
	// return dev_core_hash(out, m_with_prefix, mlen + PADDING_LEN + 3 * N);

	u8 msg[PADDING_LEN + 3 * N + XMSS_MLEN];
	u8 *mm = (u8 *)msg;

	dev_ull_to_bytes(mm, PADDING_LEN, XMSS_HASH_PADDING_HASH);
	memcpy(mm + PADDING_LEN, R, N);
	memcpy(mm + PADDING_LEN + N, root, N);
	dev_ull_to_bytes(mm + PADDING_LEN + 2 * N, N, idx);
	memcpy(mm + PADDING_LEN + 3 * N,
	       m_with_prefix + PADDING_LEN + 3 * N, XMSS_MLEN);
	return dev_core_hash(out, mm, mlen + PADDING_LEN + 3 * N);
} // dev_hash_message

__device__ int dev_hash_message_modefied(u8 *out, const u8 *R, const u8 *root,
					 u64 idx, const u8 *m, u64 mlen)
{
	u8 msg[PADDING_LEN + 3 * N + XMSS_MLEN];
	u8 *mm = (u8 *)msg;

	dev_ull_to_bytes(mm, PADDING_LEN, XMSS_HASH_PADDING_HASH);
	memcpy(mm + PADDING_LEN, R, N);
	memcpy(mm + PADDING_LEN + N, root, N);
	dev_ull_to_bytes(mm + PADDING_LEN + 2 * N, N, idx);
	memcpy(mm + PADDING_LEN + 3 * N, m, XMSS_MLEN);
	return dev_core_hash(out, mm, mlen + PADDING_LEN + 3 * N);
} // dev_hash_message

__device__ int dev_thash_h(u8 *out, const u8 *in,
			   const u8 *pub_seed, u32 addr[8])
{
	u8 buf[PADDING_LEN + 3 * N];
	u8 bitmask[2 * N];
	u8 addr_as_bytes[32];
	u32 i;

	/* Set the function padding. */
	dev_ull_to_bytes(buf, PADDING_LEN, XMSS_HASH_PADDING_H);

	/* Generate the n-byte key. */
	dev_set_key_and_mask(addr, 0);
	dev_addr_to_bytes(addr_as_bytes, addr);
	dev_prf(buf + PADDING_LEN, addr_as_bytes, pub_seed);

	/* Generate the 2n-byte mask. */
	dev_set_key_and_mask(addr, 1);
	dev_addr_to_bytes(addr_as_bytes, addr);
	dev_prf(bitmask, addr_as_bytes, pub_seed);

	dev_set_key_and_mask(addr, 2);
	dev_addr_to_bytes(addr_as_bytes, addr);
	dev_prf(bitmask + N, addr_as_bytes, pub_seed);

	for (i = 0; i < 2 * N; i++)
		buf[PADDING_LEN + N + i] = in[i] ^ bitmask[i];
	return dev_core_hash(out, buf, PADDING_LEN + 3 * N);
} // dev_thash_h

__device__ int dev_thash_f(u8 *out, const u8 *in, const u8 *pub_seed, u32 addr[8])
{
	u8 buf[PADDING_LEN + 2 * N];
	u8 bitmask[N];
	u8 addr_as_bytes[32];
	u32 i;

	/* Set the function padding. */
	dev_ull_to_bytes(buf, PADDING_LEN, XMSS_HASH_PADDING_F);

	/* Generate the n-byte key. */
	dev_set_key_and_mask(addr, 0);
	dev_addr_to_bytes(addr_as_bytes, addr);
	dev_prf(buf + PADDING_LEN, addr_as_bytes, pub_seed);

	/* Generate the n-byte mask. */
	dev_set_key_and_mask(addr, 1);
	dev_addr_to_bytes(addr_as_bytes, addr);
	dev_prf(bitmask, addr_as_bytes, pub_seed);

	for (i = 0; i < N; i++)
		buf[PADDING_LEN + N + i] = in[i] ^ bitmask[i];
	return dev_core_hash(out, buf, PADDING_LEN + 2 * N);
} // dev_thash_f
