#include <stdio.h>
#include <string.h>
#include <time.h>
#include <openssl/sha.h>
#include <iostream>
using namespace std;

#include "../sha256.h"
#include "../sha512.h"
#include "../fips202.h"

#include "gpu_common.h"
#include "gpu_randombytes.h"
#include "gpu_hash.h"

#include "gpu_keypair_origin.h"
#include "gpu_sign_origin.h"
#include "gpu_keypair_fast.h"
#include "gpu_sign_fast.h"
#include "gpu_verify.h"

#include "sha256_key.h"
#include "sha256_192_key.h"
#include "shake256_key.h"
#include "shake256_192_key.h"

#ifdef XMSSMT

#ifdef USING_BDS
#define XMSS_KEYPAIR gpu_xmssmt_keypair_fast
#define XMSS_SIGN gpu_xmssmt_sign_fast
#define XMSS_DP_KEYPAIR gpu_xmssmt_dp_keypair_fast
#define XMSS_DP_SIGN gpu_xmssmt_dp_sign_fast
#define XMSS_MSDP_SIGN gpu_xmssmt_msdp_sign_fast
#define XMSS_IP_KEYPAIR gpu_xmssmt_ip_keypair_fast
#define XMSS_IP_SIGN gpu_xmssmt_ip_sign_fast
#else // ifdef USING_BDS
#define XMSS_KEYPAIR gpu_xmssmt_keypair_origin
#define XMSS_SIGN gpu_xmssmt_sign_origin
#define XMSS_DP_KEYPAIR gpu_xmssmt_dp_keypair_origin
#define XMSS_DP_SIGN gpu_xmssmt_dp_sign_origin
#define XMSS_MSDP_SIGN gpu_xmssmt_msdp_sign_origin
#define XMSS_IP_KEYPAIR gpu_xmssmt_ip_keypair_origin
#define XMSS_IP_SIGN gpu_xmssmt_ip_sign_origin

#define XMSS_OPK_SIGN gpu_xmssmt_opk_sign_origin
#endif // ifdef USING_BDS

#define XMSS_SIGN_OPEN gpu_xmssmt_sign_open
#define XMSS_IP_SIGN_OPEN gpu_xmssmt_ip_sign_open
#define XMSS_DP_SIGN_OPEN gpu_xmssmt_dp_sign_open
#define XMSS_MSDP_SIGN_OPEN gpu_xmssmt_msdp_sign_open
#define XMSS_OPK_SIGN_OPEN gpu_xmssmt_opk_sign_open
#define XMSS_MSOPK_SIGN_OPEN gpu_xmssmt_msopk_sign_open

#else // ifdef XMSSMT

#ifdef USING_BDS
#define XMSS_KEYPAIR gpu_xmss_keypair_fast
#define XMSS_SIGN gpu_xmss_sign_fast
#define XMSS_DP_KEYPAIR gpu_xmss_dp_keypair_fast
#define XMSS_DP_SIGN gpu_xmss_dp_sign_fast
#define XMSS_MSDP_SIGN gpu_xmss_msdp_sign_fast
#define XMSS_IP_KEYPAIR gpu_xmss_ip_keypair_fast
#define XMSS_IP_SIGN gpu_xmss_ip_sign_fast
#else // ifdef USING_BDS
#define XMSS_KEYPAIR gpu_xmss_keypair_origin
#define XMSS_SIGN gpu_xmss_sign_origin
#define XMSS_DP_KEYPAIR gpu_xmss_dp_keypair_origin
#define XMSS_DP_SIGN gpu_xmss_dp_sign_origin
#define XMSS_MSDP_SIGN gpu_xmss_msdp_sign_origin
#define XMSS_IP_KEYPAIR gpu_xmss_ip_keypair_origin
#define XMSS_IP_SIGN gpu_xmss_ip_sign_origin

#define XMSS_OPK_SIGN gpu_xmss_opk_sign_origin
#endif // ifdef USING_BDS

#define XMSS_SIGN_OPEN gpu_xmss_sign_open
#define XMSS_IP_SIGN_OPEN gpu_xmss_ip_sign_open
#define XMSS_DP_SIGN_OPEN gpu_xmss_dp_sign_open
#define XMSS_MSDP_SIGN_OPEN gpu_xmss_msdp_sign_open
#define XMSS_OPK_SIGN_OPEN gpu_xmss_opk_sign_open
#define XMSS_MSOPK_SIGN_OPEN gpu_xmss_msopk_sign_open
#endif // ifdef XMSSMT

// #define XMSS_TEST_INVALIDSIG

// load constant to gpu
void gpu_Constant_Init()
{
	// int gpu_n;
	//
	// CHECK(cudaGetDeviceCount(&gpu_n));

	// for all device
	// for (int i = 0; i < gpu_n; i++) {
	int device = DEVICE_USED;

	CHECK(cudaSetDevice(device));
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpyToSymbol(K256, this_K256,
				 64 * sizeof(u32), 0, HOST_2_DEVICE));
	CHECK(cudaMemcpyToSymbol(K512, this_K512,
				 80 * sizeof(u64), 0, HOST_2_DEVICE));
	CHECK(cudaMemcpyToSymbol(KeccakF_RoundConstants, this_KeccakF_RoundConstants,
				 24 * sizeof(u64), 0, HOST_2_DEVICE));
	CHECK(cudaMemcpyToSymbol(rhotates, this_rhotates,
				 25 * sizeof(u8), 0, HOST_2_DEVICE));
	CHECK(cudaMemcpyToSymbol(iotas, this_iotas,
				 24 * sizeof(u64), 0, HOST_2_DEVICE));

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	// }

} // gpu_Constant_Init

void sha2_speed_test();
void sha2_validity_test();
void shake256_speed_test();
void shake256_validity_test();

void serial_speed_test();
int serial_validity_test();

void data_parallel_speed_test(u32 dp_num);
int data_parallel_validity_test();

void inner_parallel_speed_test();
int inner_parallel_validity_test();

void opk_parallel_speed_test(int num);
int opk_parallel_validity_test(int opk_num);

/*******************************************/
/*******************************************/
/*******************************************/
/*******************************************/
/*******************************************/
/*******************************************/

int main(int argc, char **argv)
{
	gpu_Constant_Init();

	// sha2_speed_test();
	// sha2_validity_test();

	// shake256_speed_test();
	shake256_validity_test();

	// serial_speed_test();
	// serial_validity_test();

	// inner_parallel_speed_test();
	// inner_parallel_validity_test();

	int num;

	if (argv[1] == NULL) {
		num = 32;
	} else {
		num = atoi(argv[1]);
	}

	// data_parallel_speed_test(num);
	// data_parallel_validity_test();

	opk_parallel_speed_test(num);
	// opk_parallel_validity_test(num);

	return 0;
} // main

/*******************************************/
/*******************************************/
/*******************************************/
/*******************************************/
/*******************************************/
/*******************************************/

void sha2_speed_test()
{
	u64 t0, t1;
	struct timespec start, stop;
	double result;

	u64 hash_msg_bytes = 1024 * 1024 * 1024;        // whole data

	hash_msg_bytes *= 16;                           // 16 GB
	u32 msg_num = 1024 * 1024;
	u8 *d, *md, *gpu_md, *gpu_para_md;

	CHECK(cudaMallocHost(&d, hash_msg_bytes));
	CHECK(cudaMallocHost(&md, 32));
	CHECK(cudaMallocHost(&gpu_md, 32));
	CHECK(cudaMallocHost(&gpu_para_md, 32 * msg_num));
	for (u64 i = 0; i < hash_msg_bytes; i++) d[i] = 2;

	printf("\nsha256 speed test\n");

	printf("-------------------CPU test--------------------\n");
	self_SHA256((const u8 *)d, 1024, md);
	for (int i = 1; i < 20; i++) {
		int msg_size = (2 << i);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		t0 = cpucycles();
		self_SHA256((const u8 *)d, msg_size, md);
		t1 = cpucycles();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		result = (stop.tv_sec - start.tv_sec) * 1e6
			 + (stop.tv_nsec - start.tv_nsec) / 1e3;
		printf("cpu: %dB \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
		       msg_size, result, msg_size / result,
		       t1 - t0, (t1 - t0) * 1.0 / msg_size);
	}

	printf("\n");
	printf("---------------gpu one core test----------------\n");
	face_sha256((const u8 *)d, 1024, gpu_md);
	for (int i = 1; i < 20; i++) {
		int msg_size = (2 << i);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		t0 = cpucycles();
		face_sha256((const u8 *)d, msg_size, gpu_md);
		t1 = cpucycles();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		result = (stop.tv_sec - start.tv_sec) * 1e6
			 + (stop.tv_nsec - start.tv_nsec) / 1e3;
		printf("gpu: %dB \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
		       msg_size, result, msg_size / result,
		       t1 - t0, (t1 - t0) * 1.0 / msg_size);
	}

	printf("\n");
	printf("---------------gpu dp test (82 * 512)----------------\n");
	msg_num = 82 * 512;
	for (int i = 1; ; i++) {
		int msg_size = (2 << i);
		if ((u64)msg_size * msg_num > hash_msg_bytes) break;
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		t0 = cpucycles();
		face_dp_sha256((const u8 *)d, gpu_para_md, msg_size,
			       msg_num, 82, 512);
		t1 = cpucycles();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		result = (stop.tv_sec - start.tv_sec) * 1e6
			 + (stop.tv_nsec - start.tv_nsec) / 1e3;
		printf("pra %d B, \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
		       msg_size, result, (double)msg_size * msg_num / result,
		       t1 - t0, (t1 - t0) * 1.0 / msg_size / msg_num);
	}

	printf("\n");
	printf("---------------gpu msdp test (82 * 512)----------------\n");
	msg_num = 82 * 512;
	for (int i = 10; ; i++) {
		int msg_size = (2 << i);
		if ((u64)msg_size * msg_num > hash_msg_bytes) break;
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		t0 = cpucycles();
		face_msdp_sha256((const u8 *)d, gpu_para_md, msg_size,
				 msg_num, 82, 512);
		t1 = cpucycles();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		result = (stop.tv_sec - start.tv_sec) * 1e6
			 + (stop.tv_nsec - start.tv_nsec) / 1e3;
		printf("pra %d B, \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
		       msg_size, result, (double)msg_size * msg_num / result,
		       t1 - t0, (t1 - t0) * 1.0 / msg_size / msg_num);
	}

/*
        printf("\n");
        printf("---------------gpu dp test (82 * 1024 * 8)----------------\n");
        msg_num = 82 * 1024 * 8; // should < 1024 * 1024
        for (int i = 1; ; i++) {
                int msg_size = (2 << i);
                if ((u64)msg_size * msg_num > hash_msg_bytes) break;
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
                t0 = cpucycles();
                face_dp_sha256((const u8 *)d, gpu_para_md, msg_size,
                               msg_num, msg_num / 32, 32);
                t1 = cpucycles();
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
                result = (stop.tv_sec - start.tv_sec) * 1e6
 + (stop.tv_nsec - start.tv_nsec) / 1e3;
                printf("pra %d B, \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
                       msg_size, result, (double)msg_size * msg_num / result,
                       t1 - t0, (t1 - t0) * 1.0 / msg_size / msg_num);
        }

        printf("\n");
        printf("---------------gpu dp test (128 * 1024 * 8)----------------\n");
        msg_num = 128 * 1024 * 8; // should < 1024 * 1024
        for (int i = 1; ; i++) {
                int msg_size = (2 << i);
                if ((u64)msg_size * msg_num > hash_msg_bytes) break;
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
                t0 = cpucycles();
                face_dp_sha256((const u8 *)d, gpu_para_md, msg_size,
                               msg_num, msg_num / 32, 32);
                t1 = cpucycles();
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
                result = (stop.tv_sec - start.tv_sec) * 1e6
 + (stop.tv_nsec - start.tv_nsec) / 1e3;
                printf("pra %d B, \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
                       msg_size, result, (double)msg_size * msg_num / result,
                       t1 - t0, (t1 - t0) * 1.0 / msg_size / msg_num);
        }
 */
} // sha2_speed_test

void sha2_validity_test()
{
	u64 t0, t1;
	struct timespec start, stop;
	double result;

	u32 se_msg_B = 1024 * 1024;
	u32 s_msg_B = 32;                               // single message size
	u32 p_msg_B = 82 * 512 * s_msg_B;               // for parallel test
	u32 msg_N = p_msg_B / s_msg_B;

	printf("msg_N = %d\n", msg_N);
	u8 *d, *cpu_md, *gpu_md, *cpu_para_md, *gpu_para_md;

	int right;

	CHECK(cudaMallocHost(&d, p_msg_B));
	CHECK(cudaMallocHost(&cpu_md, 32));
	CHECK(cudaMallocHost(&gpu_md, 32));
	CHECK(cudaMallocHost(&cpu_para_md, 32 * msg_N));
	CHECK(cudaMallocHost(&gpu_para_md, 32 * msg_N));
	for (int i = 0; i < p_msg_B; i++) d[i] = i;
	for (int i = 0; i < p_msg_B; i += 7) d[i] += i;

	printf("\nsha256 test\n");
	cout << flush;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();
	self_SHA256((const u8 *)d, se_msg_B, cpu_md);
	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("cpu sha256\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       result, se_msg_B / result, t1 - t0, (t1 - t0) * 1.0 / se_msg_B);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();
	face_sha256((const u8 *)d, se_msg_B, gpu_md);
	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("gpu sha256\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       result, se_msg_B / result, t1 - t0, (t1 - t0) * 1.0 / se_msg_B);

	right = 1;
	for (int j = 0; j < 32; j++) {
		if (cpu_md[j] != gpu_md[j]) {
			right = 0;
			break;
		}
	}
	if (right == 1) printf("single core check right!\n");
	else printf("single core check wrong!\n");

	/* parallel test */
	for (int j = 0; j < msg_N; j++) {
		self_SHA256((const u8 *)(d + j * s_msg_B), s_msg_B, cpu_para_md + j * 32);
	}

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();
	face_dp_sha256((const u8 *)d, gpu_para_md, s_msg_B,
		       msg_N, 82, 512);
	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("ex pra sha256\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       result, p_msg_B / result, t1 - t0, (t1 - t0) * 1.0 / p_msg_B);

	right = 1;
	for (int j = 0; j < 32; j++) {
		if (cpu_para_md[j] != gpu_para_md[j]) {
			right = 0;
			break;
		}
	}
	if (right == 1) printf("parallel check right!\n");
	else printf("parallel check wrong!\n");

	/* multi stream test
	 * When s_msg_B is small, the effect is obvious
	 */
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();
	face_msdp_sha256((const u8 *)d, gpu_para_md, s_msg_B,
			 msg_N, 82, 512);
	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("ex pra sha256\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       result, p_msg_B / result, t1 - t0, (t1 - t0) * 1.0 / p_msg_B);

	right = 1;
	for (int j = 0; j < 32; j++) {
		if (cpu_para_md[j] != gpu_para_md[j]) {
			right = 0;
			break;
		}
	}
	if (right == 1) printf("multi stream check right!\n");
	else printf("multi stream check wrong!\n");

} // sha2_validity_test

void shake256_speed_test()
{
	u64 t0, t1;
	struct timespec start, stop;
	double result;

	u64 hash_msg_bytes = 1024 * 1024 * 1024;        // whole data

	hash_msg_bytes *= 16;                           // 16 GB
	u32 msg_num = 1024 * 1024;
	u8 *d, *md, *gpu_md, *gpu_para_md;

	CHECK(cudaMallocHost(&d, hash_msg_bytes));
	CHECK(cudaMallocHost(&md, 32));
	CHECK(cudaMallocHost(&gpu_md, 32));
	CHECK(cudaMallocHost(&gpu_para_md, 32 * msg_num));
	for (u64 i = 0; i < hash_msg_bytes; i++) d[i] = 2;

	printf("\nshake256 speed test\n");

	for (int i = 1; i < 20; i++) {
		int msg_size = (2 << i);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		t0 = cpucycles();
		shake256(md, 32, (const u8 *)d, msg_size);
		t1 = cpucycles();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		result = (stop.tv_sec - start.tv_sec) * 1e6
			 + (stop.tv_nsec - start.tv_nsec) / 1e3;
		printf("cpu: %dB \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
		       msg_size, result, msg_size / result,
		       t1 - t0, (t1 - t0) * 1.0 / msg_size);
	}

	printf("\n");
	for (int i = 1; i < 20; i++) {
		int msg_size = (2 << i);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		t0 = cpucycles();
		face_shake256(gpu_md, 32, (const u8 *)d, msg_size);
		t1 = cpucycles();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		result = (stop.tv_sec - start.tv_sec) * 1e6
			 + (stop.tv_nsec - start.tv_nsec) / 1e3;
		printf("gpu: %dB \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
		       msg_size, result, msg_size / result,
		       t1 - t0, (t1 - t0) * 1.0 / msg_size);
	}

	printf("\n");
	msg_num = 82 * 512;
	for (int i = 1; ; i++) {
		int msg_size = (2 << i);
		if ((u64)msg_size * msg_num > hash_msg_bytes) break;
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		t0 = cpucycles();
		face_parallel_shake256((const u8 *)d, gpu_para_md, msg_size, 32,
				       msg_num, 82, 512);
		t1 = cpucycles();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		result = (stop.tv_sec - start.tv_sec) * 1e6
			 + (stop.tv_nsec - start.tv_nsec) / 1e3;
		printf("pra %d B, \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
		       msg_size, result, (double)msg_size * msg_num / result,
		       t1 - t0, (t1 - t0) * 1.0 / msg_size / msg_num);
	}

	printf("\n");
	msg_num = 82 * 1024 * 8; // should < 1024 * 1024
	for (int i = 1; ; i++) {
		int msg_size = (2 << i);
		if ((u64)msg_size * msg_num > hash_msg_bytes) break;
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		t0 = cpucycles();
		face_parallel_shake256((const u8 *)d, gpu_para_md, msg_size, 32,
				       msg_num, msg_num / 32, 32);
		t1 = cpucycles();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		result = (stop.tv_sec - start.tv_sec) * 1e6
			 + (stop.tv_nsec - start.tv_nsec) / 1e3;
		printf("pra %d B, \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
		       msg_size, result, (double)msg_size * msg_num / result,
		       t1 - t0, (t1 - t0) * 1.0 / msg_size / msg_num);
	}

	printf("\n");
	msg_num = 128 * 1024 * 8; // should < 1024 * 1024
	for (int i = 1; ; i++) {
		int msg_size = (2 << i);
		if ((u64)msg_size * msg_num > hash_msg_bytes) break;
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		t0 = cpucycles();
		face_parallel_shake256((const u8 *)d, gpu_para_md, msg_size, 32,
				       msg_num, msg_num / 32, 32);
		t1 = cpucycles();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		result = (stop.tv_sec - start.tv_sec) * 1e6
			 + (stop.tv_nsec - start.tv_nsec) / 1e3;
		printf("pra %d B, \t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
		       msg_size, result, (double)msg_size * msg_num / result,
		       t1 - t0, (t1 - t0) * 1.0 / msg_size / msg_num);
	}
} // shake256_speed_test

void shake256_validity_test()
{
	u64 t0, t1;
	struct timespec start, stop;
	double result;

	u32 se_msg_bytes = 64;
	u32 hash_msg_bytes = 1024 * 1024 * 1024;
	u32 s_msg_size = 1024;
	u32 msg_num = hash_msg_bytes / s_msg_size;
	u8 *d, *md, *md_single, *gpu_md, *gpu_para_md;

	CHECK(cudaMallocHost(&d, hash_msg_bytes));
	CHECK(cudaMallocHost(&md, 32));
	CHECK(cudaMallocHost(&md_single, 32));
	CHECK(cudaMallocHost(&gpu_md, 32));
	CHECK(cudaMallocHost(&gpu_para_md, 32 * msg_num));
	for (int i = 0; i < hash_msg_bytes; i++) d[i] = 2;

	printf("\nshake256 validity test\n");

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();
	shake256(md, 32, (const u8 *)d, se_msg_bytes);
	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("cpu shake\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       result, se_msg_bytes / result, t1 - t0,
	       (t1 - t0) * 1.0 / se_msg_bytes);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();
	face_shake256(gpu_md, 32, (const u8 *)d, se_msg_bytes);
	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("gpu shake\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       result, se_msg_bytes / result, t1 - t0,
	       (t1 - t0) * 1.0 / se_msg_bytes);

	shake256(md_single, 32, (const u8 *)d, s_msg_size);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();
	face_parallel_shake256((const u8 *)d, gpu_para_md, s_msg_size, 32,
			       msg_num, msg_num / 32, 32);
	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("pra shake\t%.2lf us\t%.2lfMB/s\tcycles: %lld\tC/B: %.2lf\n",
	       result, hash_msg_bytes / result, t1 - t0,
	       (t1 - t0) * 1.0 / hash_msg_bytes);

	for (int j = 0; j < 32; j++)
		printf("%02x%s", md[j], ((j % 4) == 3) ? " " : "");
	printf("\n");
	for (int j = 0; j < 32; j++)
		printf("%02x%s", gpu_md[j], ((j % 4) == 3) ? " " : "");
	printf("\n");
	for (int j = 0; j < 32; j++)
		printf("%02x%s", md_single[j], ((j % 4) == 3) ? " " : "");
	printf("\n");
	for (int j = 0; j < 32; j++)
		printf("%02x%s", gpu_para_md[j], ((j % 4) == 3) ? " " : "");
	printf("\n");
	for (int j = 32; j < 64; j++)
		printf("%02x%s", gpu_para_md[j], ((j % 4) == 3) ? " " : "");
	printf("\n");
} // shake256_validity_test

void serial_speed_test()
{
	u64 t0, t1;
	u64 t[XMSS_SIGNATURES];
	struct timespec start, stop;
	double result;
	u8 *pk, *sk, *m, *sm, *mout;
	u64 smlen, mlen;

	printf("\n");
	printf("****************************\n");
	printf("*     serial test start    *\n");
	printf("****************************\n");
	printf("\n");

	CHECK(cudaMallocHost(&pk, PK_BYTES));
	CHECK(cudaMallocHost(&sk, SK_BYTES));
	CHECK(cudaMallocHost(&m, XMSS_MLEN));
	CHECK(cudaMallocHost(&sm, SIG_BYTES + XMSS_MLEN));
	CHECK(cudaMallocHost(&mout, SIG_BYTES + XMSS_MLEN));

	randombytes(m, XMSS_MLEN);
	printf("input m =\t");
	for (int i = 0; i < XMSS_MLEN; i++)
		printf("%02x", m[i]);
	printf("\n");

	printf("Benchmarking variant %s\n", XMSS_VARIANT);

	printf("Generating keypair.. ");

	int numTest = 1;

	for (int i = 0; i < numTest; i++) {
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		t0 = cpucycles();
		XMSS_KEYPAIR(pk, sk);
		t1 = cpucycles();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
		printf("total took %lf us (%.2lf msec), %llu cycles\n",
		       result, result / 1e3, t1 - t0);
	}

	printf("Testing %d %s signatures.. \n", XMSS_SIGNATURES, XMSS_VARIANT);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
		XMSS_SIGN(sk, sm, &smlen, m, XMSS_MLEN);
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("total took %lf us (%.2lf msec)\n",
	       result / XMSS_SIGNATURES, result / 1e3 / XMSS_SIGNATURES);

	print_results(t, XMSS_SIGNATURES);

	printf("Verifying %d signatures..\n", XMSS_SIGNATURES);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
		XMSS_SIGN_OPEN(mout, &mlen, sm, smlen, pk);
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("total took %lf us (%.2lf msec)\n",
	       result / XMSS_SIGNATURES, result / 1e3 / XMSS_SIGNATURES);

	printf("mout =\t");
	for (int i = 0; i < XMSS_MLEN; i++)
		printf("%02x", mout[i + SIG_BYTES]);
	printf("\n");

	print_results(t, XMSS_SIGNATURES);

	printf("Signature size: %d (%.2f KiB)\n", SIG_BYTES, SIG_BYTES / 1024.0);
	printf("Public key size: %d (%.2f KiB)\n", PK_BYTES, PK_BYTES / 1024.0);
	printf("Secret key size: %d (%.2f KiB)\n", SK_BYTES, SK_BYTES / 1024.0);

	cudaFreeHost(pk); cudaFreeHost(sk);
	cudaFreeHost(m); cudaFreeHost(sm); cudaFreeHost(mout);

	printf("\n");
	printf("****************************\n");
	printf("*      serial test end     *\n");
	printf("****************************\n");

} // serial_speed_test

#define FILE_WRITE_DATA

void data_parallel_speed_test(u32 dp_num)
{
	u64 t0, t1;
	u64 t[XMSS_SIGNATURES];
	struct timespec start, stop;
	double result;

#ifdef FILE_WRITE_DATA
	FILE *fp;
	fp = fopen("dp.txt", "a+");
#endif // ifdef FILE_WRITE_DATA
	u8 *pk, *sk, *m, *sm, *mout;
	u64 smlen, mlen;

	printf("\n");
	printf("****************************\n");
	printf("* data parallel test start *\n");
	printf("****************************\n");
	printf("\n");

	CHECK(cudaMallocHost(&pk,       dp_num * PK_BYTES));
	CHECK(cudaMallocHost(&sk,       dp_num * SK_BYTES));
	CHECK(cudaMallocHost(&m,        dp_num * XMSS_MLEN));
	CHECK(cudaMallocHost(&sm,       dp_num * SM_BYTES));
	CHECK(cudaMallocHost(&mout,     dp_num * SM_BYTES));

	randombytes(m, XMSS_MLEN * dp_num);

	printf("Generating keypair..\n");

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();

#if (defined(USING_20_KEY)) && TREE_HEIGHT == 20
	for (int i = 0; i < dp_num; i++) {
		memcpy(sk + i * SK_BYTES, public_sk, SK_BYTES);
		memcpy(pk + i * PK_BYTES, public_pk, PK_BYTES);
	}
#else // if (defined(USING_20_KEY)) && TREE_HEIGHT == 20
	XMSS_DP_KEYPAIR(pk, sk, dp_num);
#endif // ifndef USING_20_KEY

	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external keygen %lf ms, %llu cycles, %.2lf keygen/s\n",
	       result / 1e3, t1 - t0, dp_num / result * 1e6);
	printf("internal keygen %lf ms, %llu cycles, %.2lf keygen/s\n",
	       g_result / 1e3, t1 - t0, dp_num / g_result * 1e6);
#ifdef FILE_WRITE_DATA
	double d = g_result / 1e3;
	char dd[20];
	sprintf(dd, "%d ", dp_num);
	fputs(dd, fp);
	sprintf(dd, "%.2lf ", d);
	fputs(dd, fp);
#endif // ifdef FILE_WRITE_DATA

	printf("Testing %d %s signatures.. \n", XMSS_SIGNATURES, XMSS_VARIANT);

#ifndef TEST_STREAM

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
		XMSS_DP_SIGN(sk, sm, &smlen, m, XMSS_MLEN, dp_num);
		cout << "." << flush;
	}
	cout << endl;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external average sign %.2lf msec, %.2lf sig / s\n",
	       result / XMSS_SIGNATURES / 1e3,
	       dp_num / (result / XMSS_SIGNATURES / 1e6));
	printf("internal average sign %.2lf msec, %.2lf sig / s\n",
	       g_result / XMSS_SIGNATURES / 1e3,
	       dp_num / (g_result / XMSS_SIGNATURES / 1e6));
#ifdef FILE_WRITE_DATA
	d = g_result / XMSS_SIGNATURES / 1e3;
	sprintf(dd, "%.2lf ", d);
	fputs(dd, fp);
#endif // ifdef FILE_WRITE_DATA

	print_results(t, XMSS_SIGNATURES);

	printf("Verifying %d signatures..\n", XMSS_SIGNATURES);

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
		XMSS_DP_SIGN_OPEN(mout, &mlen, sm, smlen, pk, dp_num);
		cout << "." << flush;
	}
	cout << endl;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external average open %.2lf msec, %.2lf ver / s\n",
	       result / XMSS_SIGNATURES / 1e3,
	       dp_num / (result / XMSS_SIGNATURES / 1e6));
	printf("internal average open %.2lf msec, %.2lf ver / s\n",
	       g_result / XMSS_SIGNATURES / 1e3,
	       dp_num / (g_result / XMSS_SIGNATURES / 1e6));
#ifdef FILE_WRITE_DATA
	d = g_result / XMSS_SIGNATURES / 1e3;
	sprintf(dd, "%.2lf\n", d);
	fputs(dd, fp);
#endif // ifdef FILE_WRITE_DATA

	print_results(t, XMSS_SIGNATURES);
#else // ifndef TEST_STREAM

	/* multi stream test start */

	printf("Multi stream signing\n");

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
		XMSS_MSDP_SIGN(sk, sm, &smlen, m, XMSS_MLEN, dp_num);
		// XMSS_DP_SIGN(sk, sm, &smlen, m, XMSS_MLEN, dp_num);
		cout << "." << flush;
	}
	cout << endl;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external average sign %.2lf msec, %.2lf sig / s\n",
	       result / XMSS_SIGNATURES / 1e3,
	       dp_num / (result / XMSS_SIGNATURES / 1e6));
	printf("internal average sign %.2lf msec, %.2lf sig / s\n",
	       g_result / XMSS_SIGNATURES / 1e3,
	       dp_num / (g_result / XMSS_SIGNATURES / 1e6));
#ifdef FILE_WRITE_DATA
	d = g_result / XMSS_SIGNATURES / 1e3;
	sprintf(dd, "%.2lf ", d);
	fputs(dd, fp);
#endif // ifdef FILE_WRITE_DATA

	print_results(t, XMSS_SIGNATURES);

	printf("Multi stream Verifying\n");

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
		XMSS_MSDP_SIGN_OPEN(mout, &mlen, sm, smlen, pk, dp_num);
		cout << "." << flush;
	}
	cout << endl;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external average open %.2lf msec, %.2lf ver / s\n",
	       result / XMSS_SIGNATURES / 1e3,
	       dp_num / (result / XMSS_SIGNATURES / 1e6));
	printf("internal average open %.2lf msec, %.2lf ver / s\n",
	       g_result / XMSS_SIGNATURES / 1e3,
	       dp_num / (g_result / XMSS_SIGNATURES / 1e6));
#ifdef FILE_WRITE_DATA
	d = g_result / XMSS_SIGNATURES / 1e3;
	sprintf(dd, "%.2lf\n", d);
	fputs(dd, fp);
#endif // ifdef FILE_WRITE_DATA

	print_results(t, XMSS_SIGNATURES);
	/* multi stream test finished */
#endif // ifndef TEST_STREAM

	printf("Signature size: %d (%.2f KiB)\n", SIG_BYTES, SIG_BYTES / 1024.0);
	printf("Public key size: %d (%.2f KiB)\n", PK_BYTES, PK_BYTES / 1024.0);
	printf("Secret key size: %d (%.2f KiB)\n", SK_BYTES, SK_BYTES / 1024.0);

	cudaFreeHost(pk); cudaFreeHost(sk);
	cudaFreeHost(m); cudaFreeHost(sm); cudaFreeHost(mout);

	fclose(fp);

	printf("\n");
	printf("**************************\n");
	printf("* data parallel test end *\n");
	printf("**************************\n");
} // data_parallel_speed_test

void inner_parallel_speed_test()
{
	u64 t0, t1;
	u64 t[XMSS_SIGNATURES];
	struct timespec start, stop;
	double result;

	u8 *pk, *sk, *m, *sm, *mout;
	u64 smlen, mlen;

	printf("\n");
	printf("*****************************\n");
	printf("* inner parallel test start *\n");
	printf("*****************************\n\n");

	CHECK(cudaMallocHost(&pk, PK_BYTES));
	CHECK(cudaMallocHost(&sk, SK_BYTES));
	CHECK(cudaMallocHost(&m, XMSS_MLEN));
	CHECK(cudaMallocHost(&sm, SIG_BYTES + XMSS_MLEN));
	CHECK(cudaMallocHost(&mout, SIG_BYTES + XMSS_MLEN));

	randombytes(m, XMSS_MLEN);

	printf("Benchmarking variant %s\n", XMSS_VARIANT);

	printf("Generating keypair..\n");

	XMSS_IP_KEYPAIR(pk, sk);

	for (int i = 0; i < PK_BYTES; i++) {
		printf("0x%02x, ", pk[i]);
	}
	printf("\n");
	for (int i = 0; i < SK_BYTES; i++) {
		printf("0x%02x, ", sk[i]);
	}
	printf("\n");

	int numTest = 4;

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();
	for (int i = 0; i < numTest; i++) {
		XMSS_IP_KEYPAIR(pk, sk);
		cout << "." << flush;
	}
	cout << endl;
	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external keygen %lf us (%.2lf msec), %llu cycles\n",
	       result / numTest, result / 1e3 / numTest, (t1 - t0) / numTest);
	printf("internal keygen %lf us (%.2lf msec)\n",
	       g_result / numTest, g_result / 1e3 / numTest);

	printf("Testing %d %s signatures.. \n", XMSS_SIGNATURES, XMSS_VARIANT);

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
		XMSS_IP_SIGN(sk, sm, &smlen, m, XMSS_MLEN);
		cout << "." << flush;
	}
	cout << endl;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external sign %lf us (%.2lf msec)\n",
	       result / XMSS_SIGNATURES, result / 1e3 / XMSS_SIGNATURES);
	printf("internal sign %lf us (%.2lf msec)\n",
	       g_result / XMSS_SIGNATURES, g_result / 1e3 / XMSS_SIGNATURES);

	print_results(t, XMSS_SIGNATURES);

	printf("Verifying %d signatures..\n", XMSS_SIGNATURES);

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
		XMSS_IP_SIGN_OPEN(mout, &mlen, sm, smlen, pk);
		cout << "." << flush;
	}
	cout << endl;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external open %lf us (%.2lf msec)\n",
	       result / XMSS_SIGNATURES, result / 1e3 / XMSS_SIGNATURES);
	printf("internal open %lf us (%.2lf msec)\n",
	       g_result / XMSS_SIGNATURES, g_result / 1e3 / XMSS_SIGNATURES);

	print_results(t, XMSS_SIGNATURES);

	printf("Signature size: %d (%.2f KiB)\n", SIG_BYTES, SIG_BYTES / 1024.0);
	printf("Public key size: %d (%.2f KiB)\n", PK_BYTES, PK_BYTES / 1024.0);
	printf("Secret key size: %d (%.2f KiB)\n", SK_BYTES, SK_BYTES / 1024.0);

	cudaFreeHost(pk); cudaFreeHost(sk);
	cudaFreeHost(m); cudaFreeHost(sm); cudaFreeHost(mout);

	printf("\n");
	printf("*****************************\n");
	printf("*  inner parallel test end  *\n");
	printf("*****************************\n");
} // inner_parallel_speed_test

void opk_parallel_speed_test(int opk_num)
{
	u64 t0, t1;
	u64 t[XMSS_SIGNATURES];
	struct timespec start, stop;
	double result;

	u8 *pk, *sk, *m, *sm, *mout;
	u64 smlen, mlen; // sm and m have same length

	printf("\n");
	printf("******************************\n");
	printf("* one pk parallel test start *\n");
	printf("******************************\n");

	CHECK(cudaMallocHost(&pk, PK_BYTES));
	CHECK(cudaMallocHost(&sk, SK_BYTES));
	CHECK(cudaMallocHost(&m, opk_num * XMSS_MLEN));
	CHECK(cudaMallocHost(&sm, opk_num * (SIG_BYTES + XMSS_MLEN)));
	CHECK(cudaMallocHost(&mout, opk_num * XMSS_MLEN));

	randombytes(m, XMSS_MLEN * opk_num);

	printf("Testing %s.. \n", XMSS_VARIANT);
	printf("Generating keypair..\n");

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	t0 = cpucycles();
	XMSS_IP_KEYPAIR(pk, sk);

	t1 = cpucycles();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external keygen %lf ms, %llu cycles, %.2lf keygen/s\n",
	       result / 1e3, t1 - t0, 1 / result * 1e6);
	printf("internal keygen %lf ms, %llu cycles, %.2lf keygen/s\n",
	       g_result / 1e3, t1 - t0, 1 / g_result * 1e6);

	printf("Testing %d signatures.. \n", XMSS_SIGNATURES);

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
#ifdef USING_BDS
		for (int j = 0; j < opk_num; j++) {
			XMSS_IP_SIGN(sk, sm + j * (SIG_BYTES + XMSS_MLEN), &smlen,
				     m + j * XMSS_MLEN, XMSS_MLEN);
		}
#else // ifdef USING_BDS
		XMSS_OPK_SIGN(sk, sm, &smlen, m, XMSS_MLEN, opk_num);
#endif // ifdef USING_BDS

		cout << "." << flush;
	}
	cout << endl;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external average sign %.2lf msec, %.2lf sig / s\n",
	       result / XMSS_SIGNATURES / 1e3,
	       opk_num / (result / XMSS_SIGNATURES / 1e6));
	printf("internal average sign %.2lf msec, %.2lf sig / s\n",
	       g_result / XMSS_SIGNATURES / 1e3,
	       opk_num / (g_result / XMSS_SIGNATURES / 1e6));

	print_results(t, XMSS_SIGNATURES);

	printf("Verifying %d signatures..\n", XMSS_SIGNATURES);

	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
		XMSS_OPK_SIGN_OPEN(mout, &mlen, sm, smlen, pk, opk_num);
		cout << "." << flush;
	}
	cout << endl;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external average open %.2lf msec, %.2lf ver / s\n",
	       result / XMSS_SIGNATURES / 1e3,
	       opk_num / (result / XMSS_SIGNATURES / 1e6));
	printf("internal average open %.2lf msec, %.2lf ver / s\n",
	       g_result / XMSS_SIGNATURES / 1e3,
	       opk_num / (g_result / XMSS_SIGNATURES / 1e6));

	print_results(t, XMSS_SIGNATURES);

	/* multi stream test start */
	printf("multi stream Verifying %d signatures..\n", XMSS_SIGNATURES);
	g_result = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		t[i] = cpucycles();
		XMSS_MSOPK_SIGN_OPEN(mout, &mlen, sm, smlen, pk, opk_num);
		cout << "." << flush;
	}
	cout << endl;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("external average open %.2lf msec, %.2lf ver / s\n",
	       result / XMSS_SIGNATURES / 1e3,
	       opk_num / (result / XMSS_SIGNATURES / 1e6));
	printf("internal average open %.2lf msec, %.2lf ver / s\n",
	       g_result / XMSS_SIGNATURES / 1e3,
	       opk_num / (g_result / XMSS_SIGNATURES / 1e6));

	print_results(t, XMSS_SIGNATURES);
	/* multi stream test finished */

	printf("Signature size: %d (%.2f KiB)\n", SIG_BYTES, SIG_BYTES / 1024.0);
	printf("Public key size: %d (%.2f KiB)\n", PK_BYTES, PK_BYTES / 1024.0);
	printf("Secret key size: %d (%.2f KiB)\n", SK_BYTES, SK_BYTES / 1024.0);

	cudaFreeHost(pk); cudaFreeHost(sk);
	cudaFreeHost(m); cudaFreeHost(sm); cudaFreeHost(mout);

	printf("\n");
	printf("****************************\n");
	printf("* one pk parallel test end *\n");
	printf("****************************\n");
} // opk_parallel_speed_test

int serial_validity_test()
{
	u8 *pk, *sk, *m, *sm, *mout;
	u64 smlen, mlen;
	int ret;

	printf("\n****************************\n");
	printf("*  serial valid test start *\n");
	printf("****************************\n\n");

	CHECK(cudaMallocHost(&pk, PK_BYTES));
	CHECK(cudaMallocHost(&sk, SK_BYTES));
	CHECK(cudaMallocHost(&m, XMSS_MLEN));
	CHECK(cudaMallocHost(&sm, SIG_BYTES + XMSS_MLEN));
	CHECK(cudaMallocHost(&mout, SIG_BYTES + XMSS_MLEN));

	randombytes(m, XMSS_MLEN);
	printf("m =\t");
	for (int i = 0; i < XMSS_MLEN; i++)
		printf("%02x", m[i]);
	printf("\n");

	XMSS_KEYPAIR(pk, sk);
	for (int i = 0; i < 32; i++)
		printf("%02x%s", pk[i], ((i % 4) == 3) ? " " : "");
	printf("\n");

	printf("Testing %d %s signatures.. \n", XMSS_SIGNATURES, XMSS_VARIANT);

	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		printf("  - iteration #%d:\n", i);

		XMSS_SIGN(sk, sm, &smlen, m, XMSS_MLEN);

		// for (int j = 0; j < SK_BYTES; j++)
		// 	printf("%02x%s", sk[j], ((j % 4) == 3) ? " " : "");
		// printf("\n");

		if (smlen != SIG_BYTES + XMSS_MLEN) {
			printf("  X smlen incorrect [%llu != %u]!\n",
			       smlen, SIG_BYTES);
			ret = -1;
		}else {
			printf("    smlen as expected [%llu].\n", smlen);
		}

		/* Test if signature is valid. */
		if (XMSS_SIGN_OPEN(mout, &mlen, sm, smlen, pk)) {
			printf("  X verification failed!\n");
			ret = -1;
		}else {
			printf("    verification succeeded.\n");
		}

		/* Test if the correct message was recovered. */
		if (mlen != XMSS_MLEN) {
			printf("  X mlen incorrect [%llu != %u]!\n", mlen, XMSS_MLEN);
			ret = -1;
		}else {
			printf("    mlen as expected [%llu].\n", mlen);
		}
		if (memcmp(m, mout, XMSS_MLEN)) {
			printf("  X output message incorrect!\n");
			ret = -1;
		}else {
			printf("    output message as expected.\n");
		}

		/* Test if flipping bits invalidates the signature (it should). */

		/* Flip the first bit of the message. Should invalidate. */
		sm[smlen - 1] ^= 1;
		if (!XMSS_SIGN_OPEN(mout, &mlen, sm, smlen, pk)) {
			printf("  X flipping a bit of m DID NOT invalidate signature!\n");
			ret = -1;
		}else {
			printf("    flipping a bit of m invalidates signature.\n");
		}
		sm[smlen - 1] ^= 1;

		#ifdef XMSS_TEST_INVALIDSIG
		int j;
		/* Flip one bit per hash; the signature is almost entirely hashes.
		   This also flips a bit in the index, which is also a useful test. */
		for (j = 0; j < (int)(smlen - XMSS_MLEN); j += N) {
			sm[j] ^= 1;
			if (!XMSS_SIGN_OPEN(mout, &mlen, sm, smlen, pk)) {
				printf("  X flipping bit %d DID NOT invalidate sig + m!\n", j);
				sm[j] ^= 1;
				ret = -1;
				break;
			}
			sm[j] ^= 1;
		}
		if (j >= (int)(smlen - XMSS_MLEN)) {
			printf("    changing any signature hash invalidates signature.\n");
		}
		#endif // ifdef XMSS_TEST_INVALIDSIG
	}

	printf("Signature size: %d (%.2f KiB)\n", SIG_BYTES, SIG_BYTES / 1024.0);
	printf("Public key size: %d (%.2f KiB)\n", PK_BYTES, PK_BYTES / 1024.0);
	printf("Secret key size: %d (%.2f KiB)\n", SK_BYTES, SK_BYTES / 1024.0);

	cudaFreeHost(pk); cudaFreeHost(sk);
	cudaFreeHost(m); cudaFreeHost(sm); cudaFreeHost(mout);

	printf("\n");
	printf("****************************\n");
	printf("*  serial valid test end   *\n");
	printf("****************************\n");

	return ret;
} // serial_validity_test

int data_parallel_validity_test()
{
	u8 *pk, *sk, *m, *sm, *mout;
	u64 smlen, mlen;
	int ret;
	int wrong_sum;

	int dp_num = 16384;

	printf("\n");
	printf("********************************\n");
	printf("*data parallel valid test start*\n");
	printf("********************************\n");
	printf("\n");
	printf("dp_num = %d\n", dp_num);

	CHECK(cudaMallocHost(&pk, dp_num * (PK_BYTES)));
	CHECK(cudaMallocHost(&sk, dp_num * (SK_BYTES)));
	CHECK(cudaMallocHost(&m, dp_num * XMSS_MLEN));
	CHECK(cudaMallocHost(&sm, dp_num * (SIG_BYTES + XMSS_MLEN)));
	CHECK(cudaMallocHost(&mout, dp_num * XMSS_MLEN));

	randombytes(m, XMSS_MLEN * dp_num);

#if (defined(USING_20_KEY)) && TREE_HEIGHT == 20
	for (int i = 0; i < dp_num; i++) {
		memcpy(sk + i * SK_BYTES, public_sk, SK_BYTES);
		memcpy(pk + i * PK_BYTES, public_pk, PK_BYTES);
	}
#else // if (defined(USING_20_KEY)) && TREE_HEIGHT == 20
	XMSS_DP_KEYPAIR(pk, sk, dp_num);
#endif // ifndef USING_20_KEY
	// Pk/SK should be different if random numbers are used
	printf("pk1: ");
	for (int i = 0; i < 32; i++)
		printf("%02x%s", pk[i], ((i % 4) == 3) ? " " : "");
	printf("\n");
	printf("pk2: ");
	for (int i = 0; i < 32; i++)
		printf("%02x%s", pk[i + PK_BYTES], ((i % 4) == 3) ? " " : "");
	printf("\n");

	printf("sk1: ");
	for (int i = 0; i < 32; i++)
		printf("%02x%s", sk[i], ((i % 4) == 3) ? " " : "");
	printf("\n");
	printf("sk2: ");
	for (int i = 0; i < 32; i++)
		printf("%02x%s", sk[i + SK_BYTES], ((i % 4) == 3) ? " " : "");
	printf("\n");

	printf("Testing %d %s signatures.. \n", XMSS_SIGNATURES, XMSS_VARIANT);

	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		printf("  - iteration #%d:\n", i);

		XMSS_DP_SIGN(sk, sm, &smlen, m, XMSS_MLEN, dp_num);

		if (smlen != SIG_BYTES + XMSS_MLEN) {
			printf("  X smlen incorrect [%llu != %u]!\n",
			       smlen, SIG_BYTES);
			ret = -1;
		}else {
			printf("    smlen as expected [%llu].\n", smlen);
		}

		/* Test if signature is valid. */
		ret = 0;
		wrong_sum = 0;
		for (int i = 0; i < dp_num; i++) {
			if (XMSS_IP_SIGN_OPEN(mout, &mlen, sm + i * smlen, smlen, pk + i * PK_BYTES)) {
				printf("  X verification failed!\n");
				ret = -1;
				wrong_sum++;
			}
			/* Test if the correct message was recovered. */
			if (mlen != XMSS_MLEN) {
				printf("  X mlen incorrect [%llu != %u]!\n", mlen, XMSS_MLEN);
				ret = -2;
				wrong_sum++;
			}
			if (memcmp(m + i * XMSS_MLEN, mout, XMSS_MLEN)) {
				printf("  X output message incorrect!\n");
				ret = -3;
				wrong_sum++;
			}
		}
		if (wrong_sum == 0) {
			printf("    verification succeeded.\n");
			printf("    mlen as expected [%llu].\n", mlen);
			printf("    output message as expected.\n");
		}

		/* Test if flipping bits invalidates the signature (it should). */

		/* Flip the first bit of the message. Should invalidate. */
		ret = 0;
		wrong_sum = 0;
		for (int i = 0; i < dp_num; i++) {
			sm[smlen + i * smlen - 1] ^= 1;
			if (!XMSS_IP_SIGN_OPEN(mout, &mlen, sm + i * smlen, smlen, pk + i * PK_BYTES)) {
				printf("  X flipping a bit of m DID NOT invalidate signature!\n");
				ret = -1;
				wrong_sum++;
			}
			sm[smlen + i * smlen - 1] ^= 1;
		}
		if (wrong_sum == 0) {
			printf("    flipping a bit of m invalidates signature.\n");
		}

		// test parallel verification, only test mout and mlen
		ret = 0;
		wrong_sum = 0;
		XMSS_DP_SIGN_OPEN(mout, &mlen, sm, smlen, pk, dp_num);
		/* Test if the correct message was recovered. */
		for (int i = 0; i < dp_num; i++) {
			if (mlen != XMSS_MLEN) {
				printf("  X mlen incorrect [%llu != %u]!\n", mlen, XMSS_MLEN);
				ret = -2;
				wrong_sum++;
			}
			if (memcmp(m + i * XMSS_MLEN, mout + i * XMSS_MLEN, XMSS_MLEN)) {
				printf("  X output message incorrect!\n");
				ret = -3;
				wrong_sum++;
			}
		}
		if (wrong_sum == 0) {
			printf("    dp: mlen as expected [%llu].\n", mlen);
			printf("    dp: output message as expected.\n");
		}
	}

	printf("Signature size: %d (%.2f KiB)\n", SIG_BYTES, SIG_BYTES / 1024.0);
	printf("Public key size: %d (%.2f KiB)\n", PK_BYTES, PK_BYTES / 1024.0);
	printf("Secret key size: %d (%.2f KiB)\n", SK_BYTES, SK_BYTES / 1024.0);

	cudaFreeHost(pk); cudaFreeHost(sk);
	cudaFreeHost(m); cudaFreeHost(sm); cudaFreeHost(mout);

	printf("\n");
	printf("********************************\n");
	printf("* data parallel valid test end *\n");
	printf("********************************\n");
	printf("\n");

	return ret;
} // data_parallel_validity_test

int inner_parallel_validity_test()
{
	u8 *pk, *sk, *m, *sm, *mout;
	u64 smlen, mlen;
	int ret;

	printf("\n");
	printf("*********************************\n");
	printf("*inner parallel valid test start*\n");
	printf("*********************************\n");
	printf("\n");

	CHECK(cudaMallocHost(&pk, PK_BYTES));
	CHECK(cudaMallocHost(&sk, SK_BYTES));
	CHECK(cudaMallocHost(&m, XMSS_MLEN));
	CHECK(cudaMallocHost(&sm, SIG_BYTES + XMSS_MLEN));
	CHECK(cudaMallocHost(&mout, SIG_BYTES + XMSS_MLEN));

	randombytes(m, XMSS_MLEN);

	XMSS_IP_KEYPAIR(pk, sk);
	printf("pk-root: \n");
	for (int j = 0; j < N; j++)
		printf("%02x%s", pk[j], ((j % 4) == 3) ? " " : "");
	printf("\n");

	printf("Testing %d %s signatures.. \n", XMSS_SIGNATURES, XMSS_VARIANT);

	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		printf("  - iteration #%d:\n", i);

		XMSS_IP_SIGN(sk, sm, &smlen, m, XMSS_MLEN);

		if (smlen != SIG_BYTES + XMSS_MLEN) {
			printf("  X smlen incorrect [%llu != %u]!\n",
			       smlen, SIG_BYTES);
			ret = -1;
		}else {
			printf("    smlen as expected [%llu].\n", smlen);
		}

		/* Test if signature is valid. */
		if (XMSS_IP_SIGN_OPEN(mout, &mlen, sm, smlen, pk)) {
			printf("  X verification failed!\n");
			ret = -1;
		}else {
			printf("    verification succeeded.\n");
		}

		/* Test if the correct message was recovered. */
		if (mlen != XMSS_MLEN) {
			printf("  X mlen incorrect [%llu != %u]!\n", mlen, XMSS_MLEN);
			ret = -1;
		}else {
			printf("    mlen as expected [%llu].\n", mlen);
		}
		if (memcmp(m, mout, XMSS_MLEN)) {
			printf("  X output message incorrect!\n");
			ret = -1;
		}else {
			printf("    output message as expected.\n");
		}

		/* Test if flipping bits invalidates the signature (it should). */

		/* Flip the first bit of the message. Should invalidate. */
		sm[smlen - 1] ^= 1;
		if (!XMSS_IP_SIGN_OPEN(mout, &mlen, sm, smlen, pk)) {
			printf("  X flipping a bit of m DID NOT invalidate signature!\n");
			ret = -1;
		}else {
			printf("    flipping a bit of m invalidates signature.\n");
		}
		sm[smlen - 1] ^= 1;

		#ifdef XMSS_TEST_INVALIDSIG
		int j;
		/* Flip one bit per hash; the signature is almost entirely hashes.
		   This also flips a bit in the index, which is also a useful test. */
		for (j = 0; j < (int)(smlen - XMSS_MLEN); j += N) {
			sm[j] ^= 1;
			if (!XMSS_IP_SIGN_OPEN(mout, &mlen, sm, smlen, pk)) {
				printf("  X flipping bit %d DID NOT invalidate sig + m!\n", j);
				sm[j] ^= 1;
				ret = -1;
				break;
			}
			sm[j] ^= 1;
		}
		if (j >= (int)(smlen - XMSS_MLEN)) {
			printf("    changing any signature hash invalidates signature.\n");
		}
		#endif // ifdef XMSS_TEST_INVALIDSIG

	}

	printf("Signature size: %d (%.2f KiB)\n", SIG_BYTES, SIG_BYTES / 1024.0);
	printf("Public key size: %d (%.2f KiB)\n", PK_BYTES, PK_BYTES / 1024.0);
	printf("Secret key size: %d (%.2f KiB)\n", SK_BYTES, SK_BYTES / 1024.0);

	cudaFreeHost(pk); cudaFreeHost(sk);
	cudaFreeHost(m); cudaFreeHost(sm); cudaFreeHost(mout);

	printf("\n");
	printf("*******************************\n");
	printf("*inner parallel valid test end*\n");
	printf("*******************************\n");

	return ret;
} // inner_parallel_validity_test

int opk_parallel_validity_test(int opk_num)
{
	u8 *pk, *sk, *m, *sm, *mout;
	u64 smlen, mlen;
	int ret;
	int wrong_sum;

	printf("\n");
	printf("**********************************\n");
	printf("*one pk parallel valid test start*\n");
	printf("**********************************\n");
	printf("\n");

	CHECK(cudaMallocHost(&pk, PK_BYTES));
	CHECK(cudaMallocHost(&sk, SK_BYTES));
	CHECK(cudaMallocHost(&m, opk_num * XMSS_MLEN));
	CHECK(cudaMallocHost(&sm, opk_num * SM_BYTES));
	CHECK(cudaMallocHost(&mout, opk_num * SM_BYTES));

	randombytes(m, opk_num * XMSS_MLEN);

	// Use the same key to see if the same signature is generated
	u8 *pk_opk, *sk_opk, *sm_opk, *sm_temp;

	CHECK(cudaMallocHost(&pk_opk, PK_BYTES));
	CHECK(cudaMallocHost(&sk_opk, SK_BYTES));
	CHECK(cudaMallocHost(&sm_temp, SM_BYTES));
	CHECK(cudaMallocHost(&sm_opk, opk_num * SM_BYTES));

	XMSS_IP_KEYPAIR(pk, sk);

	printf("Testing %d %s signatures.. \n", XMSS_SIGNATURES, XMSS_VARIANT);

	for (int i = 0; i < XMSS_SIGNATURES; i++) {
		printf("  - iteration #%d:\n", i);

#ifdef USING_BDS
		wrong_sum = 0;
		for (int j = 0; j < opk_num; j++) {
			XMSS_IP_SIGN(sk, sm + j * SM_BYTES,
				     &smlen, m + j * XMSS_MLEN, XMSS_MLEN);

			if (smlen != SIG_BYTES + XMSS_MLEN) {
				wrong_sum++;
				ret = -1;
			}
		}
		if (wrong_sum == 0) {
			printf("    smlen as expected [%llu].\n", smlen);
		} else {
			printf("  X smlen incorrect [%llu != %u]!\n", smlen, SIG_BYTES);
		}

#else // ifdef USING_BDS

		memcpy(pk_opk, pk, PK_BYTES);
		memcpy(sk_opk, sk, SK_BYTES);

		XMSS_OPK_SIGN(sk, sm, &smlen, m, XMSS_MLEN, opk_num);
		if (smlen == SIG_BYTES + XMSS_MLEN) {
			printf("    smlen as expected [%llu].\n", smlen);
		} else {
			printf("  X smlen incorrect [%llu != %u]!\n", smlen, SIG_BYTES);
		}

#ifdef SAME_CHECK
		int wrong = 0;
		for (int j = 0; j < opk_num; j++) {
			XMSS_IP_SIGN(sk_opk, sm_opk, &smlen, m + j * XMSS_MLEN, XMSS_MLEN);

			for (int iter = 0; iter < SIG_BYTES; iter++) {
				if (sm_opk[iter] != sm[iter + j * SM_BYTES]) {
					wrong++;
					printf("  X same_check: %d sm check wrong\n", j);
					printf("%d, %02x %02x\n",
					       iter, sm_opk[iter], sm[iter + j * SM_BYTES]);
					j = opk_num;
					break;
				}
			}
		}
		if (wrong == 0) printf("    same_check: sm check right\n");

		wrong = 0;
		for (int iter = 0; iter < SK_BYTES; iter++) {
			if (sk_opk[iter] != sk[iter]) {
				wrong++;
				// printf("%d %02x, %02x\n", iter, sk_opk[iter], sk[iter]);
			}
		}
		if (wrong == 0) printf("    same_check: sk check right\n");
		else printf("  X same_check: sk check wrong\n");
#endif  // ifdef SAME_CHECK

#endif  // ifdef USING_BDS

		// test parallel verification, only test mout and mlen
		ret = 0;
		if (1) {
			XMSS_OPK_SIGN_OPEN(mout, &mlen, sm, smlen, pk, opk_num);
		} else {
			for (int j = 0; j < opk_num; j++) {
				XMSS_IP_SIGN_OPEN(mout + j * XMSS_MLEN, &mlen,
						  sm + j * SM_BYTES, smlen, pk);
			}
		}
		if (mlen != XMSS_MLEN) {
			printf("  X mlen incorrect [%llu != %u]!\n", mlen, XMSS_MLEN);
			ret = -2;
		} else {
			printf("    opk: mlen as expected [%llu].\n", mlen);
		}
		/* Test if the correct message was recovered. */
		wrong_sum = 0;
		for (int i = 0; i < opk_num; i++) {
			if (memcmp(m + i * XMSS_MLEN, mout + i * XMSS_MLEN, XMSS_MLEN)) {
				printf("  X %d output message incorrect!, break!\n", i);
				ret = -3;
				wrong_sum++;
				break;
			}
		}
		if (wrong_sum == 0) {
			printf("    opk: output message as expected.\n");
		}

		// // test multi stream parallel verification, only test mout and mlen
		// ret = 0;
		// wrong_sum = 0;
		// XMSS_MSOPK_SIGN_OPEN(mout, &mlen, sm, smlen, pk, opk_num);
		// /* Test if the correct message was recovered. */
		// if (mlen != XMSS_MLEN) {
		// 	printf("  X mlen incorrect [%llu != %u]!\n", mlen, XMSS_MLEN);
		// 	ret = -2;
		// 	wrong_sum++;
		// }
		// for (int i = 0; i < opk_num; i++) {
		// 	if (memcmp(m + i * XMSS_MLEN,
		// 		   mout + i * XMSS_MLEN, XMSS_MLEN)) {
		// 		printf("  X %d output message incorrect!\n", i);
		// 		ret = -3;
		// 		wrong_sum++;
		// 	}
		// }
		// if (wrong_sum == 0) {
		// 	printf("    msopk: mlen as expected [%llu].\n", mlen);
		// 	printf("    msopk: output message as expected.\n");
		// }

	}

	printf("Signature size: %d (%.2f KiB)\n", SIG_BYTES, SIG_BYTES / 1024.0);
	printf("Public key size: %d (%.2f KiB)\n", PK_BYTES, PK_BYTES / 1024.0);
	printf("Secret key size: %d (%.2f KiB)\n", SK_BYTES, SK_BYTES / 1024.0);

	cudaFreeHost(pk); cudaFreeHost(sk);
	cudaFreeHost(m); cudaFreeHost(sm); cudaFreeHost(mout);

	printf("\n");
	printf("**********************************\n");
	printf("* one pk parallel valid test end *\n");
	printf("**********************************\n");
	printf("\n");

	return ret;
} // opk_parallel_validity_test
