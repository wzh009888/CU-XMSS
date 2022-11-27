#include "gpu_common.h"

int gpu_xmss_sign_open(u8 *m, u64 *mlen, const u8 *sm, u64 smlen,
		       const u8 *pk);
int gpu_xmssmt_sign_open(u8 *m, u64 *mlen, const u8 *sm, u64 smlen,
			 const u8 *pk);

int gpu_xmss_dp_sign_open(u8 *m, u64 *mlen,
			  const u8 *sm, u64 smlen, const u8 *pk, u64 num);
int gpu_xmssmt_dp_sign_open(u8 *m, u64 *mlen,
			    const u8 *sm, u64 smlen, const u8 *pk, u64 num);

int gpu_xmss_msdp_sign_open(u8 *m, u64 *mlen,
			    const u8 *sm, u64 smlen, const u8 *pk, u64 num);
int gpu_xmssmt_msdp_sign_open(u8 *m, u64 *mlen,
			      const u8 *sm, u64 smlen, const u8 *pk, u64 num);

int gpu_xmss_ip_sign_open(u8 *m, u64 *mlen, const u8 *sm, u64 smlen,
			  const u8 *pk);
int gpu_xmssmt_ip_sign_open(u8 *m, u64 *mlen, const u8 *sm, u64 smlen,
			    const u8 *pk);

int gpu_xmss_opk_sign_open(u8 *m, u64 *mlen,
			   const u8 *sm, u64 smlen, const u8 *pk, u64 num);
int gpu_xmssmt_opk_sign_open(u8 *m, u64 *mlen,
			     const u8 *sm, u64 smlen, const u8 *pk, u64 num);

int gpu_xmss_msopk_sign_open(u8 *m, u64 *mlen,
			     const u8 *sm, u64 smlen, const u8 *pk, u64 num);
int gpu_xmssmt_msopk_sign_open(u8 *m, u64 *mlen,
			       const u8 *sm, u64 smlen, const u8 *pk, u64 num);
