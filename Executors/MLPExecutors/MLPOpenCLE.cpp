#include "MLPOpenCLE.h"

#ifdef _DEBUG
#include <iostream>
#endif

namespace MentalAI
{
	namespace MLPExecutor
	{
		std::vector<cl::Device> MLPOpenCLE::getDeviceList()
		{
			std::vector<cl::Platform> _platforms;
			cl::Platform::get(&_platforms);

			std::vector<std::vector<cl::Device>> _devices(_platforms.size());
			std::vector<cl::Device> deviceList;

			for (int i = 0; i < _platforms.size(); i++)
				_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &_devices[i]);

			for (int i = 0; i < _devices.size(); i++)
			{
				for (int j = 0; j < _devices[i].size(); j++)
					deviceList.push_back(_devices[i][j]);
			}

			return deviceList;
		}

		MLPOpenCLE::MLPOpenCLE()
		{
			wMatrix = nullptr;
			tMatrix = nullptr;
			activatonFunc = nullptr;

			devices = getDeviceList();

			if (devices.empty())
			{
				deviceReady = false;
				return;
			}

			uint maxCuDeivceId = 0;
			uint maxCu = devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

			for (uint i = 0; i < devices.size(); i++)
			{
				cl_uint thisDeviceCu = devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

				if (thisDeviceCu > maxCu && devices[i].getInfo<CL_DEVICE_AVAILABLE>() == CL_TRUE)
				{
					maxCu = thisDeviceCu;
					maxCuDeivceId = i;
				}
			}

			device.push_back(devices[maxCuDeivceId]);
			deviceReady = initPreparation();
		}

		MLPOpenCLE::MLPOpenCLE(cl::Device device)
		{
			wMatrix = nullptr;
			tMatrix = nullptr;
			activatonFunc = nullptr;

			if (device.getInfo<CL_DEVICE_AVAILABLE>() == CL_TRUE)
			{
				this->device.push_back(device);
				kernelWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

				deviceReady = initPreparation();
			}
			else deviceReady = false;
		}

		bool MLPOpenCLE::initPreparation()
		{
			try
			{
				kernelWorkGroupSize = device[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
				context = cl::Context(device);
				commandQueue = cl::CommandQueue(context, device[0]);

				std::string sourceCode("#pragma cl_amd_printf : enable\n\ninline void AtomicAdd(volatile __global float *source, const float operand) \n{\n\tunion \n\t{\n\t\tunsigned int intVal;\n\t\tfloat floatVal;\n\t} newVal;\n\n\tunion \n\t{\n\t\tunsigned int intVal;\n\t\tfloat floatVal;\n\t} prevVal;\n\n\tdo \n\t{\n\t\tprevVal.floatVal = *source;\n\t\tnewVal.floatVal = prevVal.floatVal + operand;\n\t} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);\n}\n\ninline float sigmoidActivate(float sum)\n{\n\treturn 1.f / (1.f + (float)exp(-sum));\n}\n\n//================================================================================\n\n__kernel void MulWeight(__global float* input, \n\t\t\t\t   __global float* wm, uint wOffset, uint wLocalOffset, \n\t\t\t\t   __global float* out)\n{\n\tuint i = get_global_id(0);\n\tuint g = get_global_id(1);\n\t\n\tfloat mul = input[i] * wm[wOffset + i * wLocalOffset + g];\n\tAtomicAdd(&out[g], mul);\n\t\n\t//printf(\"MUL %d %d: %f OUT: %f (INPUT MY: %f THIS WM: %f\", i, g, mul, out[g], input[i], wm[wOffset + i * wLocalOffset + g]);\n\t\n\tinput[i] = 0;\n}\n\n__kernel void ZeroBuffer(__global float* input)\n{\n\tuint i = get_global_id(0);\t\n\tinput[i] = 0;\n}\n\n__kernel void Activate(__global float* output, \n\t\t\t\t\t\t__global float* tm, uint offset,\n\t\t\t\t\t\tuint activId)\n{\n\tuint i = get_global_id(0);\n\n\toutput[i] -= tm[offset + i];\n\n\tswitch (activId)\n\t{\n\tcase 1:\n\t\toutput[i] = sigmoidActivate(output[i]);\n\t\tbreak;\n\n\tdefault: \n\t\toutput[i] = sigmoidActivate(output[i]);\n\t\tbreak;\n\t}\n\t\n\t//printf(\"ACTIVATION %d: %f\", i, output[i]);\n}");

				cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
				program = cl::Program(context, source);
				program.build(device);

				kernelMult = cl::Kernel(program, "MulWeight");
				kernelActivation = cl::Kernel(program, "Activate");
				kernelZero = cl::Kernel(program, "ZeroBuffer");

				return true;
			}
			catch (cl::Error err)
			{
				#ifdef _DEBUG

				cl::STRING_CLASS buildlog;
				program.getBuildInfo(device[0], CL_PROGRAM_BUILD_LOG, &buildlog);

				std::cout << std::endl << buildlog << std::endl;

				#endif

				return false;
			}
			catch (...) // Sorry
			{
				return false;
			}
		}

		//================================================================================

		IMLPExecutor* MLPOpenCLE::createCopy()
		{
			MLPOpenCLE *newExecutor = new MLPOpenCLE();
			newExecutor->prepareExecutor(*wMatrix, *tMatrix, *activatonFunc);

			return newExecutor;
		}

		int MLPOpenCLE::prepareExecutor(std::vector<std::vector<std::vector<float>>> &wMatrix,
			std::vector<std::vector<float>> &tMatrix,
			std::vector<Activation::IActivation*> &activatonFunc)
		{
			if (!deviceReady) return -1;

			this->wMatrix = &wMatrix;
			this->tMatrix = &tMatrix;
			this->activatonFunc = &activatonFunc;

			wmOffsetSizes.resize(wMatrix.size());

			//================================================================================

			uint full_size = 0;

			maxLayerSize = wMatrix[0].size();

			for (uint i = 0; i < wMatrix.size(); i++)
			{
				full_size += wMatrix[i].size() * wMatrix[i][0].size();

				if (wMatrix[i].size() > maxLayerSize)
					maxLayerSize = wMatrix[i].size();

				wmOffsetSizes[i] = wMatrix[i].size() * wMatrix[i][0].size();
			}

			if (wMatrix[wMatrix.size() - 1][0].size() > maxLayerSize)
				maxLayerSize = wMatrix[wMatrix.size() - 1][0].size();

			float *line = new float[full_size];
			uint id = 0;

			for (uint i = 0; i < wMatrix.size(); i++)
			{
				for (uint g = 0; g < wMatrix[i].size(); g++)
				{
					for (uint z = 0; z < wMatrix[i][g].size(); z++)
					{
						line[id++] = wMatrix[i][g][z];
					}
				}
			}

			clWmBuf = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, full_size * sizeof(float), line);

			delete[] line;

			//================================================================================

			full_size = 0;

			for (uint i = 0; i < tMatrix.size(); i++)
			{
				full_size += tMatrix[i].size();
			}

			line = new float[full_size];
			id = 0;

			for (uint i = 0; i < tMatrix.size(); i++)
			{
				for (uint g = 0; g < tMatrix[i].size(); g++)
				{
					line[id++] = tMatrix[i][g];
				}
			}

			clTmBuf = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, full_size * sizeof(float), line);

			delete[] line;

			//================================================================================

			uint *actLine = new uint[activatonFunc.size()];

			for (uint i = 0; i < activatonFunc.size(); i++)
			{
				actLine[i] = activatonFunc[i]->getKernelActivatorId();
			}

			clActBuf = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, activatonFunc.size() * sizeof(uint), actLine);

			delete[] actLine;
			//================================================================================

			float *tmpZeroBuffer = new float[maxLayerSize];

			for (uint i = 0; i < maxLayerSize; i++) tmpZeroBuffer[i] = 0;

			//clInpLayer = new cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, maxLayerSize * sizeof(float), tmpZeroBuffer);
			clOutLayer = new cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, maxLayerSize * sizeof(float), tmpZeroBuffer);

			delete[] tmpZeroBuffer;

			return 0;
		}

		void MLPOpenCLE::quitExecutor()
		{
			wMatrix = nullptr;
			tMatrix = nullptr;
			activatonFunc = nullptr;

			delete clWmBuf;
			delete clTmBuf;
			delete clActBuf;
			delete clOutLayer;
		}

		std::vector<float> MLPOpenCLE::getOutputLayerFromInput(std::vector<float> &input)
		{
			if (!deviceReady) return std::vector<float>(0);

			float *tmpZeroBuffer = new float[maxLayerSize];

			for (uint i = 0; i < maxLayerSize; i++) 
				tmpZeroBuffer[i] = 0;

			for (uint i = 0; i < input.size(); i++) 
				tmpZeroBuffer[i] = input[i];

			clInpLayer = new cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, maxLayerSize * sizeof(float), tmpZeroBuffer);

			delete[] tmpZeroBuffer;

			uint tOffset = 0;

			for (uint i = 0; i < wMatrix->size(); i++)
			{
				kernelMult.setArg(0, (i % 2 == 0 ? *clInpLayer : *clOutLayer));
				kernelMult.setArg(1, *clWmBuf);
				kernelMult.setArg(2, (i != 0 ? wmOffsetSizes[i - 1] : 0));
				kernelMult.setArg(3, wMatrix->operator[](i)[0].size());
				kernelMult.setArg(4, (i % 2 == 0 ? *clOutLayer : *clInpLayer));

				commandQueue.enqueueNDRangeKernel(kernelMult, cl::NullRange, cl::NDRange(input.size(), wMatrix->operator[](i)[0].size()), cl::NDRange());

				kernelZero.setArg(0, (i % 2 == 0 ? *clInpLayer : *clOutLayer));

				commandQueue.enqueueNDRangeKernel(kernelZero, cl::NullRange, cl::NDRange(wMatrix->operator[](i).size()), cl::NDRange());

				kernelActivation.setArg(0, (i % 2 == 0 ? *clOutLayer : *clInpLayer));
				kernelActivation.setArg(1, *clTmBuf);
				kernelActivation.setArg(2, tOffset);
				kernelActivation.setArg(3, activatonFunc->operator[](i)->getKernelActivatorId());

				commandQueue.enqueueNDRangeKernel(kernelActivation, cl::NullRange, cl::NDRange(wMatrix->operator[](i)[0].size()), cl::NDRange());

				tOffset += tMatrix->operator[](i).size();
			}

			commandQueue.finish();

			uint outSize = wMatrix->operator[](wMatrix->size() - 1)[0].size();

			float *outputBuffer = new float[outSize];

			commandQueue.enqueueReadBuffer((wMatrix->size() % 2 == 0 ? *clInpLayer : *clOutLayer), CL_TRUE, 0, outSize * sizeof(float), outputBuffer);

			std::vector<float> outVec(outSize);

			for (uint i = 0; i < outVec.size(); i++) 
				outVec[i] = outputBuffer[i];

			delete[] outputBuffer;
			delete clInpLayer;

			return outVec;
		}
	}
}