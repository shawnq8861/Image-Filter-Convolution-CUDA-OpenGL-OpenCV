NVCC = /usr/local/cuda-8.0/bin/nvcc
NVCC_FLAGS = -g -G
LD_FLAGS = -Wno-deprecated-gpu-targets -lopencv_core -lopencv_highgui
INPUT_FILES = main.cpp kernel.cu
OUTPUT_FILE = modifyFilterImage

all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(INPUT_FILES)
	$(NVCC) $(NVCC_FLAGS) $(INPUT_FILES) $(LD_FLAGS) -o $(OUTPUT_FILE)

clean:
	rm $(OUTPUT_FILE)
