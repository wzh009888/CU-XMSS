
#include <math.h>
#include <time.h>
#include <stdio.h>
#include "gpu_common.h"

struct timespec g_start, g_stop;
double g_result;

u64 cpucycles(void)
{
	u64 result;
	__asm volatile (".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
			: "=a" (result) ::  "%rdx");

	return result;
} // cpucycles

int cmp_llu(const void *a, const void *b)
{
	if (*(u64 *)a < *(u64 *)b) return -1;
	if (*(u64 *)a > *(u64 *)b) return 1;
	return 0;
} // cmp_llu

u64 median(u64 *l, u64 llen)
{
	qsort(l, llen, sizeof(u64), cmp_llu);

	if (llen % 2) return l[llen / 2];
	else return (l[llen / 2 - 1] + l[llen / 2]) / 2;
} // median

u64 average(u64 *t, u64 tlen)
{
	u64 acc = 0;
	u64 i;

	for (i = 0; i < tlen; i++)
		acc += t[i];
	return acc / (tlen);
} // average

void print_results(u64 *t, u64 tlen)
{
	u64 i;

	for (i = 0; i < tlen - 1; i++)
		t[i] = t[i + 1] - t[i];
	printf("\tmedian        : %llu cycles\n", median(t, tlen));
	printf("\taverage       : %llu cycles\n", average(t, tlen - 1));
	printf("\n");
} // print_results
