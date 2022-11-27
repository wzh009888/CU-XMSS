
NVCC = nvcc -ccbin g++
CC = gcc

DIR_GPU_SRC = gpu
DIR_CPU_SRC = .
DIR_CPU_OBJ = obj
DIR_GPU_OBJ = gpu/obj
DIR_BIN = bin

WHOLE_LINK = -dlto -arch=sm_86
# WHOLE_LINK = -dlto -arch=sm_75
CFLAGS = -Wall -g -O3 -Wextra -Wpedantic
GPUFLAGS = -O3 -m64 -dlink -rdc=true $(WHOLE_LINK)
LDLIBS = -lcrypto
GPULDLIBS = -L/home/wzh/cuda/cuda-11.2/lib64 -lcudart \
			-lstdc++ -lm \
			$(WHOLE_LINK)

SOURCES = params.c hash.c fips202.c hash_address.c randombytes.c wots.c \
		xmss.c xmss_core.c xmss_commons.c utils.c sha256.c sha512.c
HEADERS = params.h hash.h fips202.h hash_address.h randombytes.h wots.h \
		xmss.h xmss_core.h xmss_commons.h utils.h sha256.h sha512.h

SOURCES_FAST = $(subst xmss_core.c, xmss_core_fast.c, $(SOURCES))
HEADERS_FAST = $(subst xmss_core.c, xmss_core_fast.c, $(HEADERS))

TESTS = test/wots \
		test/oid \
		test/speed \
		test/speed_fast \
		test/xmss_determinism \
		test/xmss \
		test/xmss_fast \
		test/xmssmt \
		test/xmssmt_fast \
		test/maxsigsxmss \
		test/maxsigsxmssmt \

UI = ui/xmss_keypair \
	 ui/xmss_sign \
	 ui/xmss_open \
	 ui/xmssmt_keypair \
	 ui/xmssmt_sign \
	 ui/xmssmt_open \
	 ui/xmss_keypair_fast \
	 ui/xmss_sign_fast \
	 ui/xmss_open_fast \
	 ui/xmssmt_keypair_fast \
	 ui/xmssmt_sign_fast \
	 ui/xmssmt_open_fast \

# all: tests ui gpu
all: gpu tests
	cp $(target) ./bin

target = gpu/xmss_fast gpu/xmssmt_fast gpu/xmss gpu/xmssmt

gpu: $(target)

TEST_DATA_FEW = 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384

TEST_DATA_LARGE = 16384 17408 18432 19456 20480 21504 22528 23552 24576 25600 \
26624 27648 28672 29696 30720 31744 32768 33792 34816 35840 36864 37888 38912 \
39936 40960 41984 43008 44032 45056 46080 47104 48128 49152 50176 51200 52224 \
53248 54272 55296 56320 57344 58368 59392 60416 61440 62464 63488 64512 65536

VAR_ALL = \
		gpu/xmss_fast01 gpu/xmss_fast02 gpu/xmss_fast03 \
		gpu/xmss_fast11 gpu/xmss_fast12 gpu/xmss_fast13 \
		gpu/xmss_fast21 gpu/xmss_fast22 gpu/xmss_fast23 \
		gpu/xmss_fast31 gpu/xmss_fast32 gpu/xmss_fast33 \
		gpu/xmssmt_fast01 gpu/xmssmt_fast11 gpu/xmssmt_fast21 gpu/xmssmt_fast31\
		gpu/xmssmt_fast02 gpu/xmssmt_fast12 gpu/xmssmt_fast22 gpu/xmssmt_fast32\
		gpu/xmssmt_fast03 gpu/xmssmt_fast13 gpu/xmssmt_fast23 gpu/xmssmt_fast33\
		gpu/xmssmt_fast04 gpu/xmssmt_fast14 gpu/xmssmt_fast24 gpu/xmssmt_fast34\
		gpu/xmssmt_fast05 gpu/xmssmt_fast15 gpu/xmssmt_fast25 gpu/xmssmt_fast35\
		gpu/xmssmt_fast06 gpu/xmssmt_fast16 gpu/xmssmt_fast26 gpu/xmssmt_fast36\
		gpu/xmssmt_fast07 gpu/xmssmt_fast17 gpu/xmssmt_fast27 gpu/xmssmt_fast37\
		gpu/xmssmt_fast08 gpu/xmssmt_fast18 gpu/xmssmt_fast28 gpu/xmssmt_fast38\
		gpu/xmss01 gpu/xmss02 gpu/xmss03 \
		gpu/xmss11 gpu/xmss12 gpu/xmss13 \
		gpu/xmss21 gpu/xmss22 gpu/xmss23 \
		gpu/xmss31 gpu/xmss32 gpu/xmss33 \
		gpu/xmssmt01 gpu/xmssmt11 gpu/xmssmt21 gpu/xmssmt31\
		gpu/xmssmt02 gpu/xmssmt12 gpu/xmssmt22 gpu/xmssmt32\
		gpu/xmssmt03 gpu/xmssmt13 gpu/xmssmt23 gpu/xmssmt33\
		gpu/xmssmt04 gpu/xmssmt14 gpu/xmssmt24 gpu/xmssmt34\
		gpu/xmssmt05 gpu/xmssmt15 gpu/xmssmt25 gpu/xmssmt35\
		gpu/xmssmt06 gpu/xmssmt16 gpu/xmssmt26 gpu/xmssmt36\
		gpu/xmssmt07 gpu/xmssmt17 gpu/xmssmt27 gpu/xmssmt37\
		gpu/xmssmt08 gpu/xmssmt18 gpu/xmssmt28 gpu/xmssmt38\

VAR_FAST_ALL = \
		gpu/xmss_fast01 gpu/xmss_fast02 gpu/xmss_fast03 \
		gpu/xmss_fast11 gpu/xmss_fast12 gpu/xmss_fast13 \
		gpu/xmss_fast21 gpu/xmss_fast22 gpu/xmss_fast23 \
		gpu/xmss_fast31 gpu/xmss_fast32 gpu/xmss_fast33 \
		gpu/xmssmt_fast01 gpu/xmssmt_fast11 gpu/xmssmt_fast21 gpu/xmssmt_fast31\
		gpu/xmssmt_fast02 gpu/xmssmt_fast12 gpu/xmssmt_fast22 gpu/xmssmt_fast32\
		gpu/xmssmt_fast03 gpu/xmssmt_fast13 gpu/xmssmt_fast23 gpu/xmssmt_fast33\
		gpu/xmssmt_fast04 gpu/xmssmt_fast14 gpu/xmssmt_fast24 gpu/xmssmt_fast34\
		gpu/xmssmt_fast05 gpu/xmssmt_fast15 gpu/xmssmt_fast25 gpu/xmssmt_fast35\
		gpu/xmssmt_fast06 gpu/xmssmt_fast16 gpu/xmssmt_fast26 gpu/xmssmt_fast36\
		gpu/xmssmt_fast07 gpu/xmssmt_fast17 gpu/xmssmt_fast27 gpu/xmssmt_fast37\
		gpu/xmssmt_fast08 gpu/xmssmt_fast18 gpu/xmssmt_fast28 gpu/xmssmt_fast38\

VAR_ORIGIN_ALL = \
		gpu/xmss01 gpu/xmss02 gpu/xmss03 \
		gpu/xmss11 gpu/xmss12 gpu/xmss13 \
		gpu/xmss21 gpu/xmss22 gpu/xmss23 \
		gpu/xmss31 gpu/xmss32 gpu/xmss33 \
		gpu/xmssmt01 gpu/xmssmt11 gpu/xmssmt21 gpu/xmssmt31\
		gpu/xmssmt02 gpu/xmssmt12 gpu/xmssmt22 gpu/xmssmt32\
		gpu/xmssmt03 gpu/xmssmt13 gpu/xmssmt23 gpu/xmssmt33\
		gpu/xmssmt04 gpu/xmssmt14 gpu/xmssmt24 gpu/xmssmt34\
		gpu/xmssmt05 gpu/xmssmt15 gpu/xmssmt25 gpu/xmssmt35\
		gpu/xmssmt06 gpu/xmssmt16 gpu/xmssmt26 gpu/xmssmt36\
		gpu/xmssmt07 gpu/xmssmt17 gpu/xmssmt27 gpu/xmssmt37\
		gpu/xmssmt08 gpu/xmssmt18 gpu/xmssmt28 gpu/xmssmt38\


VAR_SHA256 = 	gpu/xmss01 gpu/xmss02 gpu/xmss03 \
		gpu/xmss_fast01 gpu/xmss_fast02 gpu/xmss_fast03 \
		gpu/xmssmt01 gpu/xmssmt_fast01 \
		gpu/xmssmt02 gpu/xmssmt_fast02 \
		gpu/xmssmt03 gpu/xmssmt_fast03 \
		gpu/xmssmt04 gpu/xmssmt_fast04 \
		gpu/xmssmt05 gpu/xmssmt_fast05 \
		gpu/xmssmt06 gpu/xmssmt_fast06 \
		gpu/xmssmt07 gpu/xmssmt_fast07 \
		gpu/xmssmt08 gpu/xmssmt_fast08

VAR_SHAKE256 = 	gpu/xmss11 gpu/xmss12 gpu/xmss13 \
		gpu/xmss_fast11 gpu/xmss_fast12 gpu/xmss_fast13 \
		gpu/xmssmt11 gpu/xmssmt_fast11 \
		gpu/xmssmt12 gpu/xmssmt_fast12 \
		gpu/xmssmt13 gpu/xmssmt_fast13 \
		gpu/xmssmt14 gpu/xmssmt_fast14 \
		gpu/xmssmt15 gpu/xmssmt_fast15 \
		gpu/xmssmt16 gpu/xmssmt_fast16 \
		gpu/xmssmt17 gpu/xmssmt_fast17 \
		gpu/xmssmt18 gpu/xmssmt_fast18
# sha256/192
VAR2_SHA256_192 = 	gpu/xmss21 gpu/xmss22 gpu/xmss23 \
		gpu/xmss_fast21 gpu/xmss_fast22 gpu/xmss_fast23 \
		gpu/xmssmt21 gpu/xmssmt_fast21 \
		gpu/xmssmt22 gpu/xmssmt_fast22 \
		gpu/xmssmt23 gpu/xmssmt_fast23 \
		gpu/xmssmt24 gpu/xmssmt_fast24 \
		gpu/xmssmt25 gpu/xmssmt_fast25 \
		gpu/xmssmt26 gpu/xmssmt_fast26 \
		gpu/xmssmt27 gpu/xmssmt_fast27 \
		gpu/xmssmt28 gpu/xmssmt_fast28
# shake256/192
VAR_SHAKE256_192 = 	gpu/xmss31 gpu/xmss32 gpu/xmss33 \
		gpu/xmss_fast31 gpu/xmss_fast32 gpu/xmss_fast33 \
		gpu/xmssmt31 gpu/xmssmt_fast31 \
		gpu/xmssmt32 gpu/xmssmt_fast32 \
		gpu/xmssmt33 gpu/xmssmt_fast33 \
		gpu/xmssmt34 gpu/xmssmt_fast34 \
		gpu/xmssmt35 gpu/xmssmt_fast35 \
		gpu/xmssmt36 gpu/xmssmt_fast36 \
		gpu/xmssmt37 gpu/xmssmt_fast37 \
		gpu/xmssmt38 gpu/xmssmt_fast38

VAR_LARGE_FAST = \
			gpu/xmss_fast03 gpu/xmss_fast13 gpu/xmss_fast23 gpu/xmss_fast33 \
			gpu/xmssmt_fast03 gpu/xmssmt_fast13 \
			gpu/xmssmt_fast23 gpu/xmssmt_fast33 \
			gpu/xmssmt_fast06 gpu/xmssmt_fast16 \
			gpu/xmssmt_fast26 gpu/xmssmt_fast36 \

var_sha256: $(VAR_SHA256)
	cp $(VAR_SHA256) ./bin

var1: $(VAR1)
	cp $(VAR1) ./bin

var2: $(VAR2)
	cp $(VAR2) ./bin

var3: $(VAR3)
	cp $(VAR3) ./bin

var_fast: $(VAR_FAST_ALL)
	cp $(VAR_FAST_ALL) ./bin

var_origin: $(VAR_ORIGIN_ALL)
	cp $(VAR_ORIGIN_ALL) ./bin
	VAR_ORIGIN_ALL

var_large_fast: $(VAR_LARGE_FAST)
	cp $(VAR_LARGE_FAST) ./bin

var_all:
	make -j var0
	make -j var1
	make -j var2
	make -j var3

run_fast_few:
	make bash00
	make bash10
	make bash20
	make bash30

run_origin:
	make bash01
	make bash11
	make bash21
	make bash31

# ./gpu/xmss_fast01 or ./gpu/xmss_fast

run_dp_few:
	for x in $(TEST_DATA_FEW); \
	do \
	./gpu/xmss_fast $$x; \
	done

run_dp_large:
	for x in $(TEST_DATA_LARGE); \
	do \
	./gpu/xmss_fast $$x; \
	done

# h/d = 20
bash_large:
	./gpu/xmss_fast03 262144 >> ./bash_txt/xmss_fast03.txt
	./gpu/xmssmt_fast03 262144 >> ./bash_txt/xmssmt_fast03.txt
	./gpu/xmssmt_fast06 262144 >> ./bash_txt/xmssmt_fast06.txt
	./gpu/xmss_fast13 262144 >> ./bash_txt/xmss_fast13.txt
	./gpu/xmssmt_fast13 262144 >> ./bash_txt/xmssmt_fast13.txt
	./gpu/xmssmt_fast16 262144 >> ./bash_txt/xmssmt_fast16.txt
	./gpu/xmss_fast23 262144 >> ./bash_txt/xmss_fast23.txt
	./gpu/xmssmt_fast23 262144 >> ./bash_txt/xmssmt_fast23.txt
	./gpu/xmssmt_fast26 262144 >> ./bash_txt/xmssmt_fast26.txt
	./gpu/xmss_fast33 262144 >> ./bash_txt/xmss_fast33.txt
	./gpu/xmssmt_fast33 262144 >> ./bash_txt/xmssmt_fast33.txt
	./gpu/xmssmt_fast36 262144 >> ./bash_txt/xmssmt_fast36.txt

bash00:
	./gpu/xmss_fast01 262144 >> ./bash_txt/xmss_fast01.txt
	./gpu/xmss_fast02 262144 >> ./bash_txt/xmss_fast02.txt
	./gpu/xmssmt_fast01 262144 >> ./bash_txt/xmssmt_fast01.txt
	./gpu/xmssmt_fast02 262144 >> ./bash_txt/xmssmt_fast02.txt
	./gpu/xmssmt_fast04 262144 >> ./bash_txt/xmssmt_fast04.txt
	./gpu/xmssmt_fast05 262144 >> ./bash_txt/xmssmt_fast05.txt
	./gpu/xmssmt_fast07 262144 >> ./bash_txt/xmssmt_fast07.txt
	./gpu/xmssmt_fast08 262144 >> ./bash_txt/xmssmt_fast08.txt

bash10:
	./gpu/xmss_fast11 262144 >> ./bash_txt/xmss_fast11.txt
	./gpu/xmss_fast12 262144 >> ./bash_txt/xmss_fast12.txt
	./gpu/xmssmt_fast11 262144 >> ./bash_txt/xmssmt_fast11.txt
	./gpu/xmssmt_fast12 262144 >> ./bash_txt/xmssmt_fast12.txt
	./gpu/xmssmt_fast14 262144 >> ./bash_txt/xmssmt_fast14.txt
	./gpu/xmssmt_fast15 262144 >> ./bash_txt/xmssmt_fast15.txt
	./gpu/xmssmt_fast17 262144 >> ./bash_txt/xmssmt_fast17.txt
	./gpu/xmssmt_fast18 262144 >> ./bash_txt/xmssmt_fast18.txt

bash20:
	./gpu/xmss_fast21 262144 >> ./bash_txt/xmss_fast21.txt
	./gpu/xmss_fast22 262144 >> ./bash_txt/xmss_fast22.txt
	./gpu/xmssmt_fast21 262144 >> ./bash_txt/xmssmt_fast21.txt
	./gpu/xmssmt_fast22 262144 >> ./bash_txt/xmssmt_fast22.txt
	./gpu/xmssmt_fast24 262144 >> ./bash_txt/xmssmt_fast24.txt
	./gpu/xmssmt_fast25 262144 >> ./bash_txt/xmssmt_fast25.txt
	./gpu/xmssmt_fast27 262144 >> ./bash_txt/xmssmt_fast27.txt
	./gpu/xmssmt_fast28 262144 >> ./bash_txt/xmssmt_fast28.txt

bash30:
	./gpu/xmss_fast31 262144 >> ./bash_txt/xmss_fast31.txt
	./gpu/xmss_fast32 262144 >> ./bash_txt/xmss_fast32.txt
	./gpu/xmssmt_fast31 262144 >> ./bash_txt/xmssmt_fast31.txt
	./gpu/xmssmt_fast32 262144 >> ./bash_txt/xmssmt_fast32.txt
	./gpu/xmssmt_fast34 262144 >> ./bash_txt/xmssmt_fast34.txt
	./gpu/xmssmt_fast35 262144 >> ./bash_txt/xmssmt_fast35.txt
	./gpu/xmssmt_fast37 262144 >> ./bash_txt/xmssmt_fast37.txt
	./gpu/xmssmt_fast38 262144 >> ./bash_txt/xmssmt_fast38.txt

bash01:
	./gpu/xmss01 16384 >> ./bash_txt/xmss01.txt
	./gpu/xmss02 16384 >> ./bash_txt/xmss02.txt
	./gpu/xmss03 16384 >> ./bash_txt/xmss03.txt
	./gpu/xmssmt01 16384 >> ./bash_txt/xmssmt01.txt
	./gpu/xmssmt02 16384 >> ./bash_txt/xmssmt02.txt
	./gpu/xmssmt03 16384 >> ./bash_txt/xmssmt03.txt
	./gpu/xmssmt04 16384 >> ./bash_txt/xmssmt04.txt
	./gpu/xmssmt05 16384 >> ./bash_txt/xmssmt05.txt
	./gpu/xmssmt06 16384 >> ./bash_txt/xmssmt06.txt
	./gpu/xmssmt07 16384 >> ./bash_txt/xmssmt07.txt
	./gpu/xmssmt08 16384 >> ./bash_txt/xmssmt08.txt

bash11:
	./gpu/xmss11 16384 >> ./bash_txt/xmss11.txt
	./gpu/xmss12 16384 >> ./bash_txt/xmss12.txt
	./gpu/xmss13 16384 >> ./bash_txt/xmss13.txt
	./gpu/xmssmt11 16384 >> ./bash_txt/xmssmt11.txt
	./gpu/xmssmt12 16384 >> ./bash_txt/xmssmt12.txt
	./gpu/xmssmt13 16384 >> ./bash_txt/xmssmt13.txt
	./gpu/xmssmt14 16384 >> ./bash_txt/xmssmt14.txt
	./gpu/xmssmt15 16384 >> ./bash_txt/xmssmt15.txt
	./gpu/xmssmt16 16384 >> ./bash_txt/xmssmt16.txt
	./gpu/xmssmt17 16384 >> ./bash_txt/xmssmt17.txt
	./gpu/xmssmt18 16384 >> ./bash_txt/xmssmt18.txt

bash21:
	./gpu/xmss21 16384 >> ./bash_txt/xmss21.txt
	./gpu/xmss22 16384 >> ./bash_txt/xmss22.txt
	./gpu/xmss23 16384 >> ./bash_txt/xmss23.txt
	./gpu/xmssmt21 16384 >> ./bash_txt/xmssmt21.txt
	./gpu/xmssmt22 16384 >> ./bash_txt/xmssmt22.txt
	./gpu/xmssmt23 16384 >> ./bash_txt/xmssmt23.txt
	./gpu/xmssmt24 16384 >> ./bash_txt/xmssmt24.txt
	./gpu/xmssmt25 16384 >> ./bash_txt/xmssmt25.txt
	./gpu/xmssmt26 16384 >> ./bash_txt/xmssmt26.txt
	./gpu/xmssmt27 16384 >> ./bash_txt/xmssmt27.txt
	./gpu/xmssmt28 16384 >> ./bash_txt/xmssmt28.txt

bash31:
	./gpu/xmss31 16384 >> ./bash_txt/xmss31.txt
	./gpu/xmss32 16384 >> ./bash_txt/xmss32.txt
	./gpu/xmss33 16384 >> ./bash_txt/xmss33.txt
	./gpu/xmssmt31 16384 >> ./bash_txt/xmssmt31.txt
	./gpu/xmssmt32 16384 >> ./bash_txt/xmssmt32.txt
	./gpu/xmssmt33 16384 >> ./bash_txt/xmssmt33.txt
	./gpu/xmssmt34 16384 >> ./bash_txt/xmssmt34.txt
	./gpu/xmssmt35 16384 >> ./bash_txt/xmssmt35.txt
	./gpu/xmssmt36 16384 >> ./bash_txt/xmssmt36.txt
	./gpu/xmssmt37 16384 >> ./bash_txt/xmssmt37.txt
	./gpu/xmssmt38 16384 >> ./bash_txt/xmssmt38.txt

tests: $(TESTS)
ui: $(UI)

test: $(TESTS:=.exec)

.PHONY: clean test

test/%.exec: test/%
	@$<

test/xmss_fast: test/xmss.c $(SOURCES_FAST) $(OBJS) $(HEADERS_FAST)
	$(CC) $(CFLAGS) -o $@ $(SOURCES_FAST) $< $(LDLIBS)

test/xmss: test/xmss.c $(SOURCES) $(OBJS) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $< $(LDLIBS)

test/xmssmt_fast: test/xmss.c $(SOURCES_FAST) $(OBJS) $(HEADERS_FAST)
	$(CC) -DXMSSMT $(CFLAGS) -o $@ $(SOURCES_FAST) $< $(LDLIBS)

test/xmssmt: test/xmss.c $(SOURCES) $(OBJS) $(HEADERS)
	$(CC) -DXMSSMT $(CFLAGS) -o $@ $(SOURCES) $< $(LDLIBS)

test/speed: test/speed.c $(SOURCES) $(OBJS) $(HEADERS)
	$(CC) -DXMSS_VARIANT=\"XMSS-SHA2_20_256\" $(CFLAGS) -o $@ $(SOURCES) $< $(LDLIBS)
# $(CC) -DXMSSMT -DXMSS_VARIANT=\"XMSSMT-SHA2_60/3_256\" $(CFLAGS) -o $@ $(SOURCES) $< $(LDLIBS)
# $(CC) -DXMSS_VARIANT=\"XMSS-SHAKE256_10_256\" $(CFLAGS) -o $@ $(SOURCES) $< $(LDLIBS)

test/speed_fast: test/speed.c $(SOURCES_FAST) $(OBJS) $(HEADERS_FAST)
	$(CC) -DXMSS_VARIANT=\"XMSS-SHA2_20_192\" $(CFLAGS) -o $@ $(SOURCES_FAST) $< $(LDLIBS)
# $(CC) -DXMSSMT -DXMSS_VARIANT=\"XMSSMT-SHA2_40/2_256\" $(CFLAGS) -o $@ $(SOURCES_FAST) $< $(LDLIBS)
# $(CC) -DXMSS_VARIANT=\"XMSS-SHAKE256_10_256\" $(CFLAGS) -o $@ $(SOURCES_FAST) $< $(LDLIBS)

test/vectors: test/vectors.c $(SOURCES) $(OBJS) $(HEADERS)
	$(CC) -DXMSSMT $(CFLAGS) -o $@ $(SOURCES) $< $(LDLIBS)

test/maxsigsxmss: test/xmss_max_signatures.c $(SOURCES_FAST) $(OBJS) $(HEADERS_FAST)
	$(CC) $(CFLAGS) -o $@ $(SOURCES_FAST) $< $(LDLIBS)

test/maxsigsxmssmt: test/xmss_max_signatures.c $(SOURCES_FAST) $(OBJS) $(HEADERS_FAST)
	$(CC) -DXMSSMT $(CFLAGS) -o $@ $(SOURCES_FAST) $< $(LDLIBS)

test/%: test/%.c $(SOURCES) $(OBJS) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $< $(LDLIBS)

ui/xmss_%_fast: ui/%.c $(SOURCES_FAST) $(OBJS) $(HEADERS_FAST)
	$(CC) $(CFLAGS) -o $@ $(SOURCES_FAST) $< $(LDLIBS)

ui/xmssmt_%_fast: ui/%.c $(SOURCES_FAST) $(OBJS) $(HEADERS_FAST)
	$(CC) -DXMSSMT $(CFLAGS) -o $@ $(SOURCES_FAST) $< $(LDLIBS)

ui/xmss_%: ui/%.c $(SOURCES) $(OBJS) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $< $(LDLIBS)

ui/xmssmt_%: ui/%.c $(SOURCES) $(OBJS) $(HEADERS)
	$(CC) -DXMSSMT $(CFLAGS) -o $@ $(SOURCES) $< $(LDLIBS)

$(DIR_CPU_OBJ)/%.o: %.c $(HEADERS)
	@mkdir -p $(DIR_CPU_OBJ) $(DIR_BIN)
	$(CC) $(CFLAGS) -c -o "$@" "$<"

#**************** cpu hash object start ****************

HASH_SOURCES = fips202.c sha256.c sha512.c

OBJECTS = $(patsubst %.c, $(DIR_CPU_OBJ)/%.o, $(HASH_SOURCES))

#***************** cpu hash object end *****************

GPU_SOURCES = 	gpu_common.cu \
				gpu_sha256.cu \
				gpu_sha512.cu \
				gpu_shake.cu \
				gpu_shake_origin.cu \
				gpu_hash_address.cu \
				gpu_utils.cu \
				gpu_wots.cu \
				gpu_hash.cu \
				gpu_xmss_commons.cu \
				gpu_randombytes.cu \
				gpu_xmss_core_origin.cu \
				gpu_xmss_core_fast.cu \
				gpu_keypair_origin.cu \
				gpu_keypair_fast.cu \
				gpu_sign_origin.cu \
				gpu_sign_fast.cu \
				gpu_verify.cu \
				xmss.cu

GPU_HEADERS = $(DIR_GPU_SRC)/gpu_common.h

GPU_BIN = gpu/xmss gpu/xmssmt gpu/xmss_fast gpu/xmssmt_fast

#***************** xmss origin code start *****************
$(DIR_GPU_OBJ)/%.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) $(GPUFLAGS) -c -o "$@" "$<"

GPU_XMSS_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%.o, $(GPU_SOURCES))

gpu/xmss: $(GPU_XMSS_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
#***************** xmss origin code end *****************

#***************** xmss fast code start *****************
$(DIR_GPU_OBJ)/%_f.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS $(GPUFLAGS) -c -o "$@" "$<"

GPU_XMSS_F_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f.o, $(GPU_SOURCES))

gpu/xmss_fast: $(GPU_XMSS_F_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
#***************** xmss fast code end *****************

#***************** xmssmt origin code start *****************
$(DIR_GPU_OBJ)/%_mt.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT $(GPUFLAGS) -c -o "$@" "$<"

GPU_XMSSMT_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt.o, $(GPU_SOURCES))

gpu/xmssmt: $(GPU_XMSSMT_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
#***************** xmssmt origin code end *****************

#***************** xmssmt fast code start *****************
$(DIR_GPU_OBJ)/%_mt_f.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS $(GPUFLAGS) -c -o "$@" "$<"

GPU_XMSSMT_F_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f.o, $(GPU_SOURCES))

gpu/xmssmt_fast: $(GPU_XMSSMT_F_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
#***************** xmssmt fast code end *****************

#***************** xmss origin variant start *****************
$(DIR_GPU_OBJ)/%01.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHA2_10_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%02.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHA2_16_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%03.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHA2_20_256 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%11.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHAKE256_10_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%12.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHAKE256_16_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%13.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHAKE256_20_256 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%21.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHA2_10_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%22.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHA2_16_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%23.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHA2_20_192 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%31.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHAKE256_10_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%32.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHAKE256_16_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%33.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DVARIANT -DXMSS_SHAKE256_20_192 $(GPUFLAGS) -c -o "$@" "$<"

GPU_XMSS01_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%01.o, $(GPU_SOURCES))
GPU_XMSS02_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%02.o, $(GPU_SOURCES))
GPU_XMSS03_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%03.o, $(GPU_SOURCES))

GPU_XMSS11_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%11.o, $(GPU_SOURCES))
GPU_XMSS12_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%12.o, $(GPU_SOURCES))
GPU_XMSS13_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%13.o, $(GPU_SOURCES))

GPU_XMSS21_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%21.o, $(GPU_SOURCES))
GPU_XMSS22_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%22.o, $(GPU_SOURCES))
GPU_XMSS23_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%23.o, $(GPU_SOURCES))

GPU_XMSS31_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%31.o, $(GPU_SOURCES))
GPU_XMSS32_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%32.o, $(GPU_SOURCES))
GPU_XMSS33_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%33.o, $(GPU_SOURCES))

gpu/xmss01: $(GPU_XMSS01_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss02: $(GPU_XMSS02_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss03: $(GPU_XMSS03_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmss11: $(GPU_XMSS11_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss12: $(GPU_XMSS12_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss13: $(GPU_XMSS13_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmss21: $(GPU_XMSS21_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss22: $(GPU_XMSS22_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss23: $(GPU_XMSS23_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmss31: $(GPU_XMSS31_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss32: $(GPU_XMSS32_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss33: $(GPU_XMSS33_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
#****************** xmss origin variant end ******************

#***************** xmss fast variant start *****************
$(DIR_GPU_OBJ)/%_f01.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHA2_10_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_f02.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHA2_16_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_f03.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHA2_20_256 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%_f11.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHAKE256_10_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_f12.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHAKE256_16_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_f13.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHAKE256_20_256 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%_f21.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHA2_10_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_f22.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHA2_16_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_f23.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHA2_20_192 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%_f31.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHAKE256_10_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_f32.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHAKE256_16_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_f33.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DUSING_BDS -DVARIANT -DXMSS_SHAKE256_20_192 $(GPUFLAGS) -c -o "$@" "$<"

GPU_XMSS_F01_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f01.o, $(GPU_SOURCES))
GPU_XMSS_F02_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f02.o, $(GPU_SOURCES))
GPU_XMSS_F03_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f03.o, $(GPU_SOURCES))

GPU_XMSS_F11_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f11.o, $(GPU_SOURCES))
GPU_XMSS_F12_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f12.o, $(GPU_SOURCES))
GPU_XMSS_F13_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f13.o, $(GPU_SOURCES))

GPU_XMSS_F21_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f21.o, $(GPU_SOURCES))
GPU_XMSS_F22_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f22.o, $(GPU_SOURCES))
GPU_XMSS_F23_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f23.o, $(GPU_SOURCES))

GPU_XMSS_F31_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f31.o, $(GPU_SOURCES))
GPU_XMSS_F32_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f32.o, $(GPU_SOURCES))
GPU_XMSS_F33_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_f33.o, $(GPU_SOURCES))

gpu/xmss_fast01: $(GPU_XMSS_F01_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss_fast02: $(GPU_XMSS_F02_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss_fast03: $(GPU_XMSS_F03_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmss_fast11: $(GPU_XMSS_F11_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss_fast12: $(GPU_XMSS_F12_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss_fast13: $(GPU_XMSS_F13_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmss_fast21: $(GPU_XMSS_F21_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss_fast22: $(GPU_XMSS_F22_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss_fast23: $(GPU_XMSS_F23_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmss_fast31: $(GPU_XMSS_F31_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss_fast32: $(GPU_XMSS_F32_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmss_fast33: $(GPU_XMSS_F33_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
#****************** xmss fast variant end ******************

#***************** xmssmt origin variant *****************
$(DIR_GPU_OBJ)/%_mt01.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_20_2_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt02.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_20_4_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt03.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_40_2_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt04.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_40_4_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt05.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_40_8_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt06.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_60_3_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt07.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_60_6_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt08.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_60_12_256 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%_mt11.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_20_2_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt12.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_20_4_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt13.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_40_2_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt14.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_40_4_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt15.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_40_8_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt16.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_60_3_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt17.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_60_6_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt18.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_60_12_256 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%_mt21.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_20_2_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt22.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_20_4_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt23.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_40_2_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt24.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_40_4_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt25.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_40_8_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt26.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_60_3_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt27.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_60_6_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt28.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHA2_60_12_192 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%_mt31.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_20_2_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt32.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_20_4_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt33.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_40_2_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt34.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_40_4_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt35.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_40_8_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt36.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_60_3_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt37.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_60_6_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt38.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DVARIANT -DXMSSMT_SHAKE256_60_12_192 $(GPUFLAGS) -c -o "$@" "$<"

GPU_XMSSMT01_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt01.o, $(GPU_SOURCES))
GPU_XMSSMT02_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt02.o, $(GPU_SOURCES))
GPU_XMSSMT03_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt03.o, $(GPU_SOURCES))
GPU_XMSSMT04_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt04.o, $(GPU_SOURCES))
GPU_XMSSMT05_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt05.o, $(GPU_SOURCES))
GPU_XMSSMT06_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt06.o, $(GPU_SOURCES))
GPU_XMSSMT07_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt07.o, $(GPU_SOURCES))
GPU_XMSSMT08_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt08.o, $(GPU_SOURCES))

GPU_XMSSMT11_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt11.o, $(GPU_SOURCES))
GPU_XMSSMT12_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt12.o, $(GPU_SOURCES))
GPU_XMSSMT13_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt13.o, $(GPU_SOURCES))
GPU_XMSSMT14_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt14.o, $(GPU_SOURCES))
GPU_XMSSMT15_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt15.o, $(GPU_SOURCES))
GPU_XMSSMT16_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt16.o, $(GPU_SOURCES))
GPU_XMSSMT17_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt17.o, $(GPU_SOURCES))
GPU_XMSSMT18_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt18.o, $(GPU_SOURCES))

GPU_XMSSMT21_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt21.o, $(GPU_SOURCES))
GPU_XMSSMT22_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt22.o, $(GPU_SOURCES))
GPU_XMSSMT23_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt23.o, $(GPU_SOURCES))
GPU_XMSSMT24_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt24.o, $(GPU_SOURCES))
GPU_XMSSMT25_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt25.o, $(GPU_SOURCES))
GPU_XMSSMT26_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt26.o, $(GPU_SOURCES))
GPU_XMSSMT27_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt27.o, $(GPU_SOURCES))
GPU_XMSSMT28_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt28.o, $(GPU_SOURCES))

GPU_XMSSMT31_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt31.o, $(GPU_SOURCES))
GPU_XMSSMT32_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt32.o, $(GPU_SOURCES))
GPU_XMSSMT33_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt33.o, $(GPU_SOURCES))
GPU_XMSSMT34_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt34.o, $(GPU_SOURCES))
GPU_XMSSMT35_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt35.o, $(GPU_SOURCES))
GPU_XMSSMT36_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt36.o, $(GPU_SOURCES))
GPU_XMSSMT37_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt37.o, $(GPU_SOURCES))
GPU_XMSSMT38_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt38.o, $(GPU_SOURCES))

gpu/xmssmt01: $(GPU_XMSSMT01_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt02: $(GPU_XMSSMT02_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt03: $(GPU_XMSSMT03_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt04: $(GPU_XMSSMT04_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt05: $(GPU_XMSSMT05_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt06: $(GPU_XMSSMT06_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt07: $(GPU_XMSSMT07_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt08: $(GPU_XMSSMT08_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmssmt11: $(GPU_XMSSMT11_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt12: $(GPU_XMSSMT12_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt13: $(GPU_XMSSMT13_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt14: $(GPU_XMSSMT14_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt15: $(GPU_XMSSMT15_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt16: $(GPU_XMSSMT16_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt17: $(GPU_XMSSMT17_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt18: $(GPU_XMSSMT18_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmssmt21: $(GPU_XMSSMT21_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt22: $(GPU_XMSSMT22_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt23: $(GPU_XMSSMT23_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt24: $(GPU_XMSSMT24_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt25: $(GPU_XMSSMT25_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt26: $(GPU_XMSSMT26_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt27: $(GPU_XMSSMT27_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt28: $(GPU_XMSSMT28_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmssmt31: $(GPU_XMSSMT31_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt32: $(GPU_XMSSMT32_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt33: $(GPU_XMSSMT33_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt34: $(GPU_XMSSMT34_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt35: $(GPU_XMSSMT35_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt36: $(GPU_XMSSMT36_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt37: $(GPU_XMSSMT37_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt38: $(GPU_XMSSMT38_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
#***************** xmssmt origin varient end *****************

#***************** xmssmt fast variant *****************
$(DIR_GPU_OBJ)/%_mt_f01.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_20_2_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f02.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_20_4_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f03.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_40_2_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f04.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_40_4_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f05.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_40_8_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f06.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_60_3_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f07.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_60_6_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f08.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_60_12_256 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%_mt_f11.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_20_2_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f12.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_20_4_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f13.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_40_2_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f14.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_40_4_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f15.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_40_8_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f16.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_60_3_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f17.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_60_6_256 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f18.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_60_12_256 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%_mt_f21.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_20_2_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f22.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_20_4_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f23.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_40_2_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f24.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_40_4_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f25.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_40_8_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f26.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_60_3_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f27.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_60_6_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f28.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHA2_60_12_192 $(GPUFLAGS) -c -o "$@" "$<"

$(DIR_GPU_OBJ)/%_mt_f31.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_20_2_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f32.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_20_4_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f33.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_40_2_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f34.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_40_4_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f35.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_40_8_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f36.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_60_3_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f37.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_60_6_192 $(GPUFLAGS) -c -o "$@" "$<"
$(DIR_GPU_OBJ)/%_mt_f38.o: $(DIR_GPU_SRC)/%.cu $(HEADERS) $(GPU_HEADERS)
	@mkdir -p $(DIR_GPU_OBJ)
	$(NVCC) -DXMSSMT -DUSING_BDS -DVARIANT -DXMSSMT_SHAKE256_60_12_192 $(GPUFLAGS) -c -o "$@" "$<"

GPU_XMSSMT_F01_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f01.o, $(GPU_SOURCES))
GPU_XMSSMT_F02_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f02.o, $(GPU_SOURCES))
GPU_XMSSMT_F03_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f03.o, $(GPU_SOURCES))
GPU_XMSSMT_F04_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f04.o, $(GPU_SOURCES))
GPU_XMSSMT_F05_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f05.o, $(GPU_SOURCES))
GPU_XMSSMT_F06_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f06.o, $(GPU_SOURCES))
GPU_XMSSMT_F07_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f07.o, $(GPU_SOURCES))
GPU_XMSSMT_F08_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f08.o, $(GPU_SOURCES))

GPU_XMSSMT_F11_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f11.o, $(GPU_SOURCES))
GPU_XMSSMT_F12_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f12.o, $(GPU_SOURCES))
GPU_XMSSMT_F13_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f13.o, $(GPU_SOURCES))
GPU_XMSSMT_F14_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f14.o, $(GPU_SOURCES))
GPU_XMSSMT_F15_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f15.o, $(GPU_SOURCES))
GPU_XMSSMT_F16_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f16.o, $(GPU_SOURCES))
GPU_XMSSMT_F17_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f17.o, $(GPU_SOURCES))
GPU_XMSSMT_F18_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f18.o, $(GPU_SOURCES))

GPU_XMSSMT_F21_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f21.o, $(GPU_SOURCES))
GPU_XMSSMT_F22_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f22.o, $(GPU_SOURCES))
GPU_XMSSMT_F23_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f23.o, $(GPU_SOURCES))
GPU_XMSSMT_F24_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f24.o, $(GPU_SOURCES))
GPU_XMSSMT_F25_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f25.o, $(GPU_SOURCES))
GPU_XMSSMT_F26_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f26.o, $(GPU_SOURCES))
GPU_XMSSMT_F27_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f27.o, $(GPU_SOURCES))
GPU_XMSSMT_F28_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f28.o, $(GPU_SOURCES))

GPU_XMSSMT_F31_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f31.o, $(GPU_SOURCES))
GPU_XMSSMT_F32_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f32.o, $(GPU_SOURCES))
GPU_XMSSMT_F33_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f33.o, $(GPU_SOURCES))
GPU_XMSSMT_F34_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f34.o, $(GPU_SOURCES))
GPU_XMSSMT_F35_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f35.o, $(GPU_SOURCES))
GPU_XMSSMT_F36_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f36.o, $(GPU_SOURCES))
GPU_XMSSMT_F37_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f37.o, $(GPU_SOURCES))
GPU_XMSSMT_F38_OBJECTS = $(patsubst %.cu, $(DIR_GPU_OBJ)/%_mt_f38.o, $(GPU_SOURCES))

gpu/xmssmt_fast01: $(GPU_XMSSMT_F01_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast02: $(GPU_XMSSMT_F02_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast03: $(GPU_XMSSMT_F03_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast04: $(GPU_XMSSMT_F04_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast05: $(GPU_XMSSMT_F05_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast06: $(GPU_XMSSMT_F06_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast07: $(GPU_XMSSMT_F07_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast08: $(GPU_XMSSMT_F08_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmssmt_fast11: $(GPU_XMSSMT_F11_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast12: $(GPU_XMSSMT_F12_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast13: $(GPU_XMSSMT_F13_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast14: $(GPU_XMSSMT_F14_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast15: $(GPU_XMSSMT_F15_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast16: $(GPU_XMSSMT_F16_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast17: $(GPU_XMSSMT_F17_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast18: $(GPU_XMSSMT_F18_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmssmt_fast21: $(GPU_XMSSMT_F21_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast22: $(GPU_XMSSMT_F22_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast23: $(GPU_XMSSMT_F23_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast24: $(GPU_XMSSMT_F24_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast25: $(GPU_XMSSMT_F25_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast26: $(GPU_XMSSMT_F26_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast27: $(GPU_XMSSMT_F27_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast28: $(GPU_XMSSMT_F28_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)

gpu/xmssmt_fast31: $(GPU_XMSSMT_F31_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast32: $(GPU_XMSSMT_F32_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast33: $(GPU_XMSSMT_F33_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast34: $(GPU_XMSSMT_F34_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast35: $(GPU_XMSSMT_F35_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast36: $(GPU_XMSSMT_F36_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast37: $(GPU_XMSSMT_F37_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
gpu/xmssmt_fast38: $(GPU_XMSSMT_F38_OBJECTS) $(OBJECTS)
	$(NVCC) -O3 -o $@ $^ $(GPULDLIBS)
#***************** xmssmt fast varient end *****************

# Higher priority
# $(DIR_GPU_OBJ)/gpu_sha256.o: $(DIR_GPU_SRC)/gpu_sha256.cu $(HEADERS)
# 	$(NVCC) $(GPUFLAGS) -c -o "$@" "$<" #-maxrregcount 24 slower
# $(DIR_GPU_OBJ)/gpu_shake_mt_f.o: $(DIR_GPU_SRC)/gpu_shake.cu $(HEADERS)
# 	$(NVCC) -DXMSSMT -DUSING_BDS $(GPUFLAGS) -c -o "$@" "$<" -maxrregcount 128

#***************** xmssmt gpu code end *****************

clean:
	-$(RM) $(TESTS)
	-$(RM) test/vectors
	-$(RM) $(UI)
	-$(RM) $(DIR_CPU_OBJ)/*.o
	-$(RM) $(DIR_GPU_OBJ)/*.o
	-$(RM) $(GPU_BIN)
	-$(RM) $(VAR_ALL)
	-$(RM) $(DIR_BIN)/*
