
// Gaussian filter of image

__kernel void gaussian_filter(__read_only image2d_t srcImg,
                              __write_only image2d_t dstImg,
                              sampler_t sampler,
                              int width, int height,
			      __local float* A)
{
    // Gaussian Kernel is:
    // 1  2  1
    // 2  4  2
    // 1  2  1
    float kernelWeights[9] = { 1.0f, 2.0f, 1.0f,
                               2.0f, 4.0f, 2.0f,
                               1.0f, 2.0f, 1.0f };

    int localWorkId = get_local_id(0);
    int localWorkSize = get_local_size(0);
    int globalId = get_global_id(0);
    int offset = ((globalId/localWorkSize) * localWorkSize) - 1;

    for(int count1 = localWorkId; count1 < width + 2; count1 = count1 + localWorkSize)
    {
    	for(int count2 = 0; count2 < localWorkSize + 2; count2++)
	{
		float4 outColor = read_imagef(srcImg, sampler, (int2)(count1, count2 + offset));
		A[((count2*(width+2))+count1)*4] = outColor.x;
		A[((count2*(width+2))+count1)*4 + 1] = outColor.y;
		A[((count2*(width+2))+count1)*4 + 2] = outColor.z;
		A[((count2*(width+2))+count1)*4 + 3] = outColor.w;
	}
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(globalId < height)
    {
	    for(int count = 0; count < width; count++)
	    {
	    	int2 startImageCoord = (int2) (count, get_local_id(0));
	    	int2 endImageCoord   = (int2) (count + 2, get_local_id(0) + 2);
	    	int2 outImageCoord = (int2) (count, get_global_id(0));

	    	if (outImageCoord.x < width)
	    	{
			int weight = 0;
			float4 outColor = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
			for( int y = startImageCoord.y; y <= endImageCoord.y; y++)
			{
		    		for( int x = startImageCoord.x; x <= endImageCoord.x; x++)
		    		{
		    			float a = A[((y*(width+2)) + x)*4];
					float b = A[((y*(width+2)) + x)*4 + 1];
					float c = A[((y*(width+2)) + x)*4 + 2];
					float d = A[((y*(width+2)) + x)*4 + 3];
		        		outColor += ((float4)(a,b,c,d) * (kernelWeights[weight] / 16.0f));
		        		weight += 1;
		    		}
			}

			// Write the output value to image
			write_imagef(dstImg, outImageCoord, outColor);
	    	}	
	     }	
    }

}
