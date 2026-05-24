#include <ap_int.h>
#include <cstring>
#include <hls_stream.h>
#include <hls_vector.h>

typedef hls::vector<float,16> float16;
typedef hls::vector<float,8> float8;
typedef hls::vector<float,4> float4;
typedef hls::vector<float,2> float2;
typedef hls::vector<float,1> float1;
    void kernel_cnn(float4 vinput[3326976], float1 vweight[1638400], float16 voutput[802816]) ;
