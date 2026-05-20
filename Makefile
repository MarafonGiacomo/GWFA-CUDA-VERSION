CC = gcc
NVCC = nvcc
CFLAGS = -lineinfo -g 
HOST_CFLAGS = -O3
INCLUDES =
OBJS = gwf-ed.o gfa-base.o gfa-io.o gfa-sub.o
PROG = gwf-test
LIBS = -lz -lpthread -lm


ifneq ($(gwf_debug),)
    CFLAGS += -DGWF_ENABLE_DEBUG_LOG
endif

.SUFFIXES: .c .cu .o
.PHONY: all clean depend

all: $(PROG)

$(PROG): $(OBJS) main.o
	$(NVCC) $(CFLAGS) -rdc=true $^ -o $@ $(LIBS)

.cu.o:
	$(NVCC) -rdc=true -c $(CFLAGS) $(INCLUDES) -Xcompiler "$(HOST_CFLAGS)" -Xptxas -v $< -o $@

.c.o:
	$(CC) -c $(CFLAGS) $(INCLUDES) $(HOST_CFLAGS) $< -o $@

clean:
	rm -fr gmon.out *.o a.out $(PROG) *~ *.a *.dSYM


gfa-base.o: gfa-priv.h gfa.h kstring.h khash.h ksort.h
gfa-io.o: kstring.h gfa-priv.h gfa.h kseq.h
gfa-sub.o: gfa-priv.h gfa.h kavl.h khash.h ksort.h
gwf-ed.o: gwfa.h ksort.h khashl.h kdq.h kvec.h
gwfa-lin.o: gwfa.h ksort.h
main.o: gfa.h gfa-priv.h gwfa.h ketopt.h kseq.h
