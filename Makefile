CC_FILES=$(shell find ./csrc/ -name "*.cu")
EXE_FILES=$(CC_FILES:.cu=)

all:$(EXE_FILES)

%:%.cu
	nvcc -o $@ $< -O2 -g -G -arch=sm_86 -std=c++17 -Icsrc/3rd/cutlass/include -Icsrc/3rd/cutlass/tools/util/include

clean:
	rm -rf $(EXE_FILES)