#pragma cl_amd_printf : enable

inline void AtomicAdd(volatile __global float *source, const float operand)
{
	union
	{
		unsigned int intVal;
		float floatVal;
	} newVal;

	union
	{
		unsigned int intVal;
		float floatVal;
	} prevVal;

	do
	{
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline float sigmoidActivate(float sum)
{
	return 1.f / (1.f + (float)exp(-sum));
}

inline float devSigmoidActivate(float sum)
{
	float y = sigmoidActivate(sum);
	return y * (1 - y);
}

inline float UseActivator(uint activId, float sum)
{
	switch (activId)
	{
	case 1:
		return sigmoidActivate(sum);
		break;

	default:
		return 0;
		break;
	}
}

inline float UseDevActivator(uint activId, float sum)
{
	switch (activId)
	{
	case 1:
		return devSigmoidActivate(sum);
		break;

	default:
		return 0;
		break;
	}
}

//================================================================================

__kernel void MulWeight(__global float* input, uint inputOffset,
						__global float* wm, uint wOffset, uint wLocalOffset,
						__global float* out, __global float* lastYBuf, uint lastOffset)
{
	uint i = get_global_id(0);
	uint g = get_global_id(1);

	lastYBuf[lastOffset + i] = input[inputOffset + i];

	float mul = input[inputOffset + i] * wm[wOffset + i * wLocalOffset + g];
	AtomicAdd(&out[g], mul);
}

__kernel void ZeroBuffer(__global float* input)
{
	uint i = get_global_id(0);
	input[i] = 0;
}

__kernel void Activate(__global float* output,
						__global float* tm, uint offset,
						uint activId, __global float* lastSumBuf)
{
	uint i = get_global_id(0);

	output[i] -= tm[offset + i];
	lastSumBuf[offset + i] = output[i];
	
	output[i] = UseActivator(activId, output[i]);
}

//================================================================================

__kernel void Compare(__global float* output, __global float* refs, uint refOffset, 
						__global uint* chgFlag, float targetError)
{
	uint i = get_global_id(0);

	if (fabs((float)(output[i] - refs[refOffset + i])) > targetError)
		*chgFlag = 0;
}

__kernel void OutLayerError(__global float* out, __global float* ref, uint refOffset,
							__global float* errs, uint offset)
{
	uint i = get_global_id(0);
	errs[offset + i] = out[i] - ref[refOffset + i];
}

__kernel void HiddenLayerError(__global float* errs, uint offset, uint nextOffset,
								__global float* wm, uint wOffset, uint wLocalOffset,
								__global float* lastSum, uint activId)
{
	uint i = get_global_id(0);
	uint g = get_global_id(1);

	float addVal = errs[nextOffset + g] * UseDevActivator(activId, lastSum[nextOffset + g]) * wm[wOffset + i * wLocalOffset + g];

	AtomicAdd(&errs[offset + i], addVal);
}

__kernel void ChangeWeight(__global float* wm, uint wOffset, uint wLocalOffset,
							__global float* errs, uint offset, __global float* lastSum,
							__global float* lastY, uint lastYOffset,
							uint activId, float a)
{
	uint i = get_global_id(0);
	uint g = get_global_id(1);

	float addVal = -1 * (a * errs[offset + g] * UseDevActivator(activId, lastSum[offset + g]) * lastY[lastYOffset + i]);

	AtomicAdd(&wm[wOffset + i * wLocalOffset + g], addVal);
}

__kernel void ChangeT(__global float *tm, uint tOffset, __global float* errs, uint activId, __global float* lastSum, float a)
{
	uint i = get_global_id(0);

	tm[tOffset + i] += a * errs[tOffset + i] * UseDevActivator(activId, lastSum[tOffset + i]);
}