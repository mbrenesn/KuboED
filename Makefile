default : ED.x

CXXLINKER=icpc
CXX=icpc
CXXFLAGS=-g -O2 -DMKL_ILP64

ED.x : ED.o
	$(CXXLINKER) $^ -o $@ -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

%.o : %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $< -I${MKLROOT}/include

clean :
	rm *.o *.x
