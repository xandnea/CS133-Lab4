# Makefile for running Vitis HLS scripts

# Default target
all: vitis csim

# Run synthesis script
vitis:
	v++ -c --mode hls --config vitis_hls.ini --work_dir hls_cnn cnn.cpp

# Run C simulation script
csim:
	#vitis_hls -f csim.tcl
	# Please run ulimit -s unlimited
	# You can try vitis_hls -f csim.tcl but it is faster to run the csim.cpp directly
	g++ csim.cpp cnn.cpp -I/tools/Xilinx/2025.1/Vitis/include/ -O3 -o csim.out
	./csim.out

# Clean intermediate HLS project directories (optional)
clean:
	rm -rf hls_cnn

zip:
	sudo apt install -y zip
	cp hls_cnn/hls/syn/report/kernel_cnn_csynth.rpt .
	zip -r lab4.zip kernel_cnn_csynth.rpt lab4-report.pdf cnn.cpp

.PHONY: all vitis csim clean
