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

//================================================================================

__kernel void MulWeight(__global float* input,
	__global float* wm, uint wOffset, uint wLocalOffset,
	__global float* out)
{
	uint i = get_global_id(0);
	uint g = get_global_id(1);

	float mul = input[i] * wm[wOffset + i * wLocalOffset + g];
	AtomicAdd(&out[g], mul);

	//printf("MUL %d %d: %f OUT: %f (INPUT MY: %f THIS WM: %f\n", i, g, mul, out[g], input[i], wm[wOffset + i * wLocalOffset + g]);
}

__kernel void ZeroBuffer(__global float* input)
{
	uint i = get_global_id(0);
	//printf("=== DELETE %d ===\n", i);
	input[i] = 0;
}

__kernel void Activate(__global float* output,
	__global float* tm, uint offset,
	uint activId)
{
	uint i = get_global_id(0);

	//printf("INPUT: %d: %f\n", i, output[i]);

	output[i] -= tm[offset + i];

	switch (activId)
	{
	case 1:
		output[i] = sigmoidActivate(output[i]);
		break;

	default:
		output[i] = sigmoidActivate(output[i]);
		break;
	}

	//printf("ACTIVATION %d: %f\n", i, output[i]);
}