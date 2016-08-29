CC=g++
CFLAGS=-g3 -O2 -Wshadow -Wall -Werror -Wunused -I/usr/local/torch/install/include/ -I/usr/local/torch/install/include/TH -I/usr/local/torch/install/include/THC -I/usr/local/cuda/include/ -I. -I../nn-gpu/ -I../matrix-gpu/ -fPIC
LIBS=

LIB_SRCS=\
	nnio_lua.C \
	normalize_lua.C \
	simpletest_lua.C \
	tfidf_lua.C \
	timehires_lua.C \
	x21profile_lua.C \

LIB_OBJS=$(LIB_SRCS:C=o)

LEX_FILES=\

LEX_SRCS=$(LEX_FILES:l=C)
LEX_OBJS=$(LEX_SRCS:C=o)


PROTOS=\

PROTOS_DIRS=$(PROTOS:.protoc=)
PROTOS_CSOURCES=$(PROTOS:.protoc=.protoc.pb.C)
PROTOS_HEADERS=$(PROTOS:.protoc=.protoc.pb.h)
PROTOS_GENERATED=$(PROTOS_CSOURCES) $(PROTOS_HEADERS)
PROTOS_OBJS=$(PROTOS_CSOURCES:C=o)

MAIN_SRCS=\

MAIN_OBJS=$(MAIN_SRCS:C=o)

OBJS=$(LIB_OBJS) $(MAIN_OBJS) $(LEX_OBJS) $(PROTOS_OBJS)
CANDIDATES=$(filter %.d,$(patsubst %.C,%.d,$(patsubst %.c,%.d,$(LIB_SRCS) $(LEX_SRCS) $(MAIN_SRCS))))
DEPS=$(join $(dir $(CANDIDATES)),$(addprefix .,$(notdir $(CANDIDATES))))


BINS=\
	nnio.so \
	normalize.so \
	simpletest.so \
	tfidf.so \
	timehires.so \
	x21profile.so \


all: $(PROTOS_GENERATED) $(LEX_SRCS) $(OBJS) $(BINS)

test: all
	/usr/local/torch/install/bin/th nnio_test.lua

push: checkin
	/usr/bin/rsync -avz -e ssh --exclude=.*.sw? --exclude=.*.d --exclude=*.txt --exclude=*.txt.bz2 /home/binesh/src/torch7-libv2/ gpu.home.hex21.com:/home/binesh/src/torch7-libv2/
	/usr/bin/rsync -avz -e ssh --exclude=.*.sw? --exclude=.*.d --exclude=*.txt --exclude=*.txt.bz2 /home/binesh/src/torch7-libv2/ som.hex21.com:/home/binesh/src/torch7-libv2/

checkin:
	/usr/bin/ci -l -m- -t- Makefile $(MAIN_SRCS) $(LIB_SRCS) $(LEX_SRCS) *.[CH]


clean:
	/bin/rm -f $(OBJS) $(BINS) $(LEX_SRCS) $(UGH_SRCS) $(UGH_HDRS) $(PROTOS_GENERATED) *.o $(DEPS)

nnio.so: nnio_lua.o
	$(CC) -shared $(CFLAGS) -o $(@) $(^)

normalize.so: normalize_lua.o
	$(CC) -shared $(CFLAGS) -o $(@) $(^)

simpletest.so: simpletest_lua.o
	$(CC) -shared $(CFLAGS) -o $(@) $(^)

tfidf.so: tfidf_lua.o
	$(CC) -shared $(CFLAGS) -o $(@) $(^)

timehires.so: timehires_lua.o
	$(CC) -shared $(CFLAGS) -o $(@) $(^)

x21profile.so: x21profile_lua.o
	$(CC) -shared $(CFLAGS) -o $(@) $(^)

%.o: %.C
	$(CC) -c $(CFLAGS) $(filter-out /usr/local/include/%.C,$(filter %.C,$(^))) -o $(@)

%.C: %.l
	/usr/bin/flex -s -o$(@) $(^)

.%.d: %.C
	@$(CC) $(CFLAGS) -MT $(patsubst %.C,%.o,$(patsubst %.c,%.o,$(<))) -M $(<) -o $(@)

.%.d: %.c
	@$(CC) $(CFLAGS) -MT $(patsubst %.C,%.o,$(patsubst %.c,%.o,$(<))) -M $(<) -o $(@)

-include $(DEPS)
