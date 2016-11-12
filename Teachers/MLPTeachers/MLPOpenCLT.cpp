#include "MLPOpenCLT.h"

#ifdef _DEBUG
#include <iostream>
#endif /* _DEBUG */

namespace MentalAI
{
	namespace Teacher
	{
		std::vector<cl::Device> MLPOpenCLT::getDeviceList()
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

		bool MLPOpenCLT::initPreparation()
		{
			try
			{
				kernelWorkGroupSize = device[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
				context = cl::Context(device);
				commandQueue = cl::CommandQueue(context, device[0]);

				std::string sourceCode("#pragma cl_amd_printf : enable\n\ninline void AtomicAdd(volatile __global float *source, const float operand)\n{\n\tunion\n\t{\n\t\tunsigned int intVal;\n\t\tfloat floatVal;\n\t} newVal;\n\n\tunion\n\t{\n\t\tunsigned int intVal;\n\t\tfloat floatVal;\n\t} prevVal;\n\n\tdo\n\t{\n\t\tprevVal.floatVal = *source;\n\t\tnewVal.floatVal = prevVal.floatVal + operand;\n\t} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);\n}\n\ninline float sigmoidActivate(float sum)\n{\n\treturn 1.f / (1.f + (float)exp(-sum));\n}\n\ninline float devSigmoidActivate(float sum)\n{\n\tfloat y = sigmoidActivate(sum);\n\treturn y * (1 - y);\n}\n\ninline float UseActivator(uint activId, float sum)\n{\n\tswitch (activId)\n\t{\n\tcase 1:\n\t\treturn sigmoidActivate(sum);\n\t\tbreak;\n\n\tdefault:\n\t\treturn 0;\n\t\tbreak;\n\t}\n}\n\ninline float UseDevActivator(uint activId, float sum)\n{\n\tswitch (activId)\n\t{\n\tcase 1:\n\t\treturn devSigmoidActivate(sum);\n\t\tbreak;\n\n\tdefault:\n\t\treturn 0;\n\t\tbreak;\n\t}\n}\n\n//================================================================================\n\n__kernel void MulWeight(__global float* input, uint inputOffset,\n\t\t\t\t\t\t__global float* wm, uint wOffset, uint wLocalOffset,\n\t\t\t\t\t\t__global float* out, __global float* lastYBuf, uint lastOffset)\n{\n\tuint i = get_global_id(0);\n\tuint g = get_global_id(1);\n\n\tlastYBuf[lastOffset + i] = input[inputOffset + i];\n\n\tfloat mul = input[inputOffset + i] * wm[wOffset + i * wLocalOffset + g];\n\tAtomicAdd(&out[g], mul);\n}\n\n__kernel void ZeroBuffer(__global float* input)\n{\n\tuint i = get_global_id(0);\n\tinput[i] = 0;\n}\n\n__kernel void Activate(__global float* output,\n\t\t\t\t\t\t__global float* tm, uint offset,\n\t\t\t\t\t\tuint activId, __global float* lastSumBuf)\n{\n\tuint i = get_global_id(0);\n\n\toutput[i] -= tm[offset + i];\n\tlastSumBuf[offset + i] = output[i];\n\t\n\toutput[i] = UseActivator(activId, output[i]);\n}\n\n//================================================================================\n\n__kernel void Compare(__global float* output, __global float* refs, uint refOffset, \n\t\t\t\t\t\t__global uint* chgFlag, float targetError)\n{\n\tuint i = get_global_id(0);\n\n\tif (fabs((float)(output[i] - refs[refOffset + i])) > targetError)\n\t\t*chgFlag = 0;\n}\n\n__kernel void OutLayerError(__global float* out, __global float* ref, uint refOffset,\n\t\t\t\t\t\t\t__global float* errs, uint offset)\n{\n\tuint i = get_global_id(0);\n\terrs[offset + i] = out[i] - ref[refOffset + i];\n}\n\n__kernel void HiddenLayerError(__global float* errs, uint offset, uint nextOffset,\n\t\t\t\t\t\t\t\t__global float* wm, uint wOffset, uint wLocalOffset,\n\t\t\t\t\t\t\t\t__global float* lastSum, uint activId)\n{\n\tuint i = get_global_id(0);\n\tuint g = get_global_id(1);\n\n\tfloat addVal = errs[nextOffset + g] * UseDevActivator(activId, lastSum[nextOffset + g]) * wm[wOffset + i * wLocalOffset + g];\n\n\tAtomicAdd(&errs[offset + i], addVal);\n}\n\n__kernel void ChangeWeight(__global float* wm, uint wOffset, uint wLocalOffset,\n\t\t\t\t\t\t\t__global float* errs, uint offset, __global float* lastSum,\n\t\t\t\t\t\t\t__global float* lastY, uint lastYOffset,\n\t\t\t\t\t\t\tuint activId, float a)\n{\n\tuint i = get_global_id(0);\n\tuint g = get_global_id(1);\n\n\tfloat addVal = -1 * (a * errs[offset + g] * UseDevActivator(activId, lastSum[offset + g]) * lastY[lastYOffset + i]);\n\n\tAtomicAdd(&wm[wOffset + i * wLocalOffset + g], addVal);\n}\n\n__kernel void ChangeT(__global float *tm, uint tOffset, __global float* errs, uint activId, __global float* lastSum, float a)\n{\n\tuint i = get_global_id(0);\n\n\ttm[tOffset + i] += a * errs[tOffset + i] * UseDevActivator(activId, lastSum[tOffset + i]);\n}");

				cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
				program = cl::Program(context, source);
				program.build(device);

				//Подставить в будущем норм функции
				kernelMult = cl::Kernel(program, "MulWeight");
				kernelActivation = cl::Kernel(program, "Activate");
				kernelZero = cl::Kernel(program, "ZeroBuffer");
				kernelCompare = cl::Kernel(program, "Compare");
				kernelErrOutput = cl::Kernel(program, "OutLayerError");
				kernelErrHidden = cl::Kernel(program, "HiddenLayerError");
				//kernelFullErr = cl::Kernel(program, "HiddenLayerError");
				kernelChangeW = cl::Kernel(program, "ChangeWeight");
				kernelChangeT = cl::Kernel(program, "ChangeT");

				return true;
			}
			catch (cl::Error err)
			{
				#ifdef _DEBUG

				cl::STRING_CLASS buildlog;
				program.getBuildInfo(device[0], CL_PROGRAM_BUILD_LOG, &buildlog);

				std::cout << std::endl << buildlog << std::endl;

				#endif /* _DEBUG */

				return false;
			}
			catch (...)
			{
				return false;
			}
		}

		cl::Buffer* MLPOpenCLT::getOutputLayerFromInput(uint index)
		{
			if (!deviceReady) return 0;

			float *tmpZeroBuffer = new float[maxLayerSize];

			for (uint i = 0; i < maxLayerSize; i++)
				tmpZeroBuffer[i] = 0;

			clOutLayer = new cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, maxLayerSize * sizeof(float), tmpZeroBuffer);
			clInpLayer = new cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, maxLayerSize * sizeof(float), tmpZeroBuffer);

			for (uint i = 0; i < wMatrix.size(); i++)
			{
				if (i == 0)
				{
					kernelMult.setArg(0, *clTrainSetBuf);
					kernelMult.setArg(1, index * trainingSet[index].size());
				}
				else
				{
					kernelMult.setArg(0, (i % 2 == 0 ? *clInpLayer : *clOutLayer));
					kernelMult.setArg(1, 0);
				}

				kernelMult.setArg(2, *clWmBuf);
				kernelMult.setArg(3, wmOffsetSizes[i]);
				kernelMult.setArg(4, wMatrix[i][0].size());
				kernelMult.setArg(5, (i % 2 == 0 ? *clOutLayer : *clInpLayer));
				kernelMult.setArg(6, *clLasyYBuf);
				kernelMult.setArg(7, lastYOffsetSizes[i]);

				commandQueue.enqueueNDRangeKernel(kernelMult, cl::NullRange, cl::NDRange(wMatrix[i].size(), wMatrix[i][0].size()), cl::NDRange());

				if (i != 0)
				{
					kernelZero.setArg(0, (i % 2 == 0 ? *clInpLayer : *clOutLayer));
					commandQueue.enqueueNDRangeKernel(kernelZero, cl::NullRange, cl::NDRange(wMatrix[i].size()), cl::NDRange());
				}

				kernelActivation.setArg(0, (i % 2 == 0 ? *clOutLayer : *clInpLayer));
				kernelActivation.setArg(1, *clTmBuf);
				kernelActivation.setArg(2, tmOffsetSizes[i]);
				kernelActivation.setArg(3, pActivators[i]->getKernelActivatorId());
				kernelActivation.setArg(4, *clLastSumBuf);

				commandQueue.enqueueNDRangeKernel(kernelActivation, cl::NullRange, cl::NDRange(wMatrix[i][0].size()), cl::NDRange());

				commandQueue.finish();

				if (wMatrix.size() % 2 == 0)
				{
					delete clOutLayer;
					return clInpLayer;
				}
				else
				{
					delete clInpLayer;
					return clOutLayer;
				}
			}

			delete[] tmpZeroBuffer;

			uint tOffset = 0;
			uint wOffset = 0;
		}

		bool MLPOpenCLT::Compare(cl::Buffer *out, uint index)
		{
			float *line = new float[1];

			line[0] = 1;

			cl::Buffer *clChgFlag = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 1 * sizeof(float), line);

			delete[] line;

			kernelCompare.setArg(0, *out);
			kernelCompare.setArg(1, *clRefValueBuf);
			kernelCompare.setArg(2, index * refVal[index].size());
			kernelCompare.setArg(3, *clChgFlag);
			kernelCompare.setArg(4, targetError);

			commandQueue.enqueueNDRangeKernel(kernelCompare, cl::NullRange, cl::NDRange(refVal[index].size()), cl::NDRange());
			commandQueue.finish();

			float *retFlag = new float[1];

			commandQueue.enqueueReadBuffer(*clChgFlag, CL_TRUE, 0, 1 * sizeof(float), retFlag);

			bool retValue = (bool)retFlag[0];

			delete[] retFlag;
			delete clChgFlag;

			return retValue;
		}

		float MLPOpenCLT::calculateError()
		{
			return 1000;
		}

		void MLPOpenCLT::getLayersError(cl::Buffer *out, uint index)
		{
			kernelZero.setArg(0, *clErrBuf);
			commandQueue.enqueueNDRangeKernel(kernelZero, cl::NullRange, cl::NDRange(tmFullSize), cl::NDRange());
			commandQueue.finish();

			kernelErrOutput.setArg(0, *out);
			kernelErrOutput.setArg(1, *clRefValueBuf);
			kernelErrOutput.setArg(2, index * refVal[index].size());
			kernelErrOutput.setArg(3, *clErrBuf);
			kernelErrOutput.setArg(4, tmOffsetSizes[tmOffsetSizes.size() - 1]);

			commandQueue.enqueueNDRangeKernel(kernelErrOutput, cl::NullRange, cl::NDRange(refVal[index].size()), cl::NDRange());
			commandQueue.finish();

			if (tMatrix.size() == 1) return;

			for (uint i = tMatrix.size() - 2;; --i)
			{
				kernelErrHidden.setArg(0, *clErrBuf);
				kernelErrHidden.setArg(1, tmOffsetSizes[i]);
				kernelErrHidden.setArg(2, tmOffsetSizes[i + 1]);
				kernelErrHidden.setArg(3, *clWmBuf);
				kernelErrHidden.setArg(4, wmOffsetSizes[i]);
				kernelErrHidden.setArg(5, wMatrix[i][0].size());
				kernelErrHidden.setArg(6, *clLastSumBuf);
				kernelErrHidden.setArg(7, pActivators[i]->getKernelActivatorId());

				commandQueue.enqueueNDRangeKernel(kernelErrHidden, cl::NullRange, cl::NDRange(wMatrix[i].size(), wMatrix[i][0].size()), cl::NDRange());

				if (i == 0) return;
			}
		}

		//================================================================================

		MLPOpenCLT::MLPOpenCLT()
		{
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

			lock = false;
		}

		MLPOpenCLT::MLPOpenCLT(cl::Device device)
		{
			if (device.getInfo<CL_DEVICE_AVAILABLE>() == CL_TRUE)
			{
				this->device.push_back(device);
				kernelWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

				deviceReady = initPreparation();
			}
			else deviceReady = false;

			lock = false;
		}

		MLPOpenCLT::~MLPOpenCLT()
		{
			/*
			delete clWmBuf;
			delete clInpLayer;
			delete clOutLayer;
			delete clTmBuf;
			delete clActBuf;
			delete clLastSumBuf;
			delete clLasyYBuf;	
			delete clTrainSetBuf;
			delete clRefValueBuf;*/
		}

		bool MLPOpenCLT::loadTrainingSet(std::vector<std::vector<float>> trainSet)
		{
			if (lock) return false;
			if (trainSet.empty()) return false;
			if (trainSet[0].empty()) return false;
			if (wMatrix.empty()) return false;
			if (trainSet[0].size() != wMatrix[0].size()) return false;

			this->trainingSet = trainSet;
			return true;
		}

		bool MLPOpenCLT::loadReferenceValues(std::vector<std::vector<float>> refvalue)
		{
			if (lock) return false;
			if (refvalue.size() != trainingSet.size() || trainingSet.empty()) return false;
			if (wMatrix[wMatrix.size() - 1][0].size() != refvalue[0].size()) return false;

			refVal = refvalue;
			return true;
		}

		bool MLPOpenCLT::setWeightMatrix(std::vector<std::vector<std::vector<float>>> wMatrix)
		{
			if (lock) return false;

			//Size check start

			if (wMatrix.size() < 1) return false;

			try {
				uint _size = 0;

				for (uint i = 0; i < wMatrix.size(); i++)
				{
					if (wMatrix[i].size())
					{
						if (i > 0)
						{
							if (_size != wMatrix[i].size())
								return false;
						}

						_size = wMatrix[i][0].size();

						for (uint g = 0; g < wMatrix[i].size(); g++)
						{
							if (wMatrix[i][g].size() != _size) return false;
						}
					}
					else return false;
				}
			}
			catch (...)
			{
				return false;
			}
			//Size check finish

			this->wMatrix = wMatrix;

			if (tMatrix.size() == wMatrix.size())
			{
				for (uint i = 0; i < wMatrix.size(); i++)
					if (wMatrix[i][0].size() != tMatrix[i].size()) tMatrix.clear();
			}
			else tMatrix.clear();

			if (pActivators.size() != wMatrix.size())
			{
				for (auto i : pActivators)
					if (i != nullptr) delete i;

				pActivators.clear();
			}

			return true;
		}

		bool MLPOpenCLT::setActivators(std::vector<Activation::IActivation*> pActivators)
		{
			if (lock) return false;
			if (pActivators.size() != wMatrix.size()) return false;

			this->pActivators.resize(pActivators.size(), nullptr);

			for (uint i = 0; i < this->pActivators.size(); i++)
				this->pActivators[i] = pActivators[i];

			return true;

		}

		std::vector<std::vector<std::vector<float>>> MLPOpenCLT::getNewWeightMatrix()
		{
			float *wmLine = new float[wmFullSize];

			commandQueue.enqueueReadBuffer(*clWmBuf, CL_TRUE, 0, wmFullSize * sizeof(float), wmLine);

			uint index = 0;

			for (uint i = 0; i < wMatrix.size(); i++)
			{
				for (uint g = 0; g < wMatrix[i].size(); g++)
				{
					for (uint z = 0; z < wMatrix[i][g].size(); z++)
						wMatrix[i][g][z] = wmLine[index++];
				}
			}

			delete[] wmLine;

			return wMatrix;
		}

		std::vector<std::vector<float>> MLPOpenCLT::getNewTMatrix()
		{
			float *tmLine = new float[tmFullSize];

			commandQueue.enqueueReadBuffer(*clWmBuf, CL_TRUE, 0, tmFullSize * sizeof(float), tmLine);

			uint index = 0;

			for (uint i = 0; i < tMatrix.size(); i++)
			{
				for (uint g = 0; g < tMatrix[i].size(); g++)
				{
					tMatrix[i][g] = tmLine[index++];
				}
			}

			delete[] tmLine;

			return tMatrix;
		}

		bool MLPOpenCLT::setTrainStep(float a)
		{
			if (lock) return false;

			this->a = a;
			return true;
		}

		bool MLPOpenCLT::setTargetError(float err)
		{
			if (lock) return false;

			this->targetError = err;
			return true;
		}

		float MLPOpenCLT::getCurrError()
		{
			return currError;
		}

		int MLPOpenCLT::prepareTeacher()
		{
			if (!deviceReady) return -1;

			//================================================================================
			
			uint full_size = 0;

			maxLayerSize = wMatrix[0].size();
			wmOffsetSizes.resize(wMatrix.size(), 0);
			lastYOffsetSizes.resize(wMatrix.size(), 0);

			for (uint i = 0; i < wMatrix.size(); i++)
			{
				full_size += wMatrix[i].size() * wMatrix[i][0].size();

				if (wMatrix[i].size() > maxLayerSize)
					maxLayerSize = wMatrix[i].size();

				if (i > 0)
				{
					wmOffsetSizes[i] = wmOffsetSizes[i - 1] + wMatrix[i - 1].size() * wMatrix[i - 1][0].size();
					lastYOffsetSizes[i] = lastYOffsetSizes[i - 1] + wMatrix[i - 1].size();
				}
			}

			wmFullSize = full_size;

			if (wMatrix[wMatrix.size() - 1][0].size() > maxLayerSize)
				maxLayerSize = wMatrix[wMatrix.size() - 1][0].size();

			float *line = new float[full_size];
			uint id = 0;

			for (uint i = 0; i < wMatrix.size(); i++)
				for (uint g = 0; g < wMatrix[i].size(); g++)
					for (uint z = 0; z < wMatrix[i][g].size(); z++)
						line[id++] = wMatrix[i][g][z];

			clWmBuf = new cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, full_size * sizeof(float), line);

			delete[] line;

			//================================================================================

			full_size = 0;

			tMatrix.resize(wMatrix.size());

			for (uint i = 0; i < wMatrix.size(); i++)
				tMatrix[i].resize(wMatrix[i][0].size(), 0);

			tmOffsetSizes.resize(tMatrix.size(), 0);

			for (uint i = 0; i < tMatrix.size(); i++)
			{
				full_size += tMatrix[i].size();
			
				if (i > 0)
					tmOffsetSizes[i] = tmOffsetSizes[i - 1] + tMatrix[i - 1].size();
			}

			tmFullSize = full_size;

			line = new float[full_size];
			id = 0;

			for (uint i = 0; i < tMatrix.size(); i++)
				for (uint g = 0; g < tMatrix[i].size(); g++)
					line[id++] = tMatrix[i][g];

			clTmBuf = new cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, full_size * sizeof(float), line);
			clErrBuf = new cl::Buffer(context, CL_MEM_READ_WRITE, full_size * sizeof(float));
			clLastSumBuf = new cl::Buffer(context, CL_MEM_READ_WRITE, full_size * sizeof(float));

			full_size -= wMatrix[wMatrix.size() - 1][0].size();
			full_size += wMatrix[0].size();

			clLasyYBuf = new cl::Buffer(context, CL_MEM_READ_WRITE, full_size * sizeof(float));

			delete[] line;

			//================================================================================

			/*uint *actLine = new uint[pActivators.size()];

			for (uint i = 0; i < pActivators.size(); i++)
				actLine[i] = pActivators[i]->getKernelActivatorId();

			clActBuf = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pActivators.size() * sizeof(uint), actLine);

			delete[] actLine;*/

			//================================================================================

			full_size = trainingSet.size() * trainingSet[0].size();

			line = new float[full_size];

			for (uint i = 0; i < trainingSet.size(); i++)
				for (uint g = 0; g < trainingSet[i].size(); g++)
					line[i * trainingSet[i].size() + g] = trainingSet[i][g];

			clTrainSetBuf = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, full_size * sizeof(float), line);

			delete[] line;

			//================================================================================

			full_size = refVal.size() * refVal[0].size();

			line = new float[full_size];

			for (uint i = 0; i < refVal.size(); i++)
				for (uint g = 0; g < refVal[i].size(); g++)
					line[i * refVal[i].size() + g] = refVal[i][g];

			clRefValueBuf = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, full_size * sizeof(float), line);

			delete[] line;

			//================================================================================

			lock = true;

			return 0;
		}

		void MLPOpenCLT::quitTeacher()
		{
			//delete buffers
		}

		int MLPOpenCLT::tryTrain(uint maxIter)
		{
			if (!lock) return -1;
			if (!deviceReady) return -3;

			try
			{
				bool exitFlag = false;
				uint iter_num = 0;
				uint glob_iter = 0;
				float netError = 0;

				while (!exitFlag)
				{
					for (uint i = 0; i < trainingSet.size(); i++)
					{
						cl::Buffer* out = getOutputLayerFromInput(i);

						if (Compare(out, i))
						{
							delete out;
							continue;
						}

						getLayersError(out, i);

						for (uint g = wMatrix.size() - 1;; g--)
						{
							kernelChangeW.setArg(0, *clWmBuf);
							kernelChangeW.setArg(1, wmOffsetSizes[g]);
							kernelChangeW.setArg(2, wMatrix[g][0].size());
							kernelChangeW.setArg(3, *clErrBuf);
							kernelChangeW.setArg(4, tmOffsetSizes[g]);
							kernelChangeW.setArg(5, *clLastSumBuf);
							kernelChangeW.setArg(6, *clLasyYBuf);
							kernelChangeW.setArg(7, lastYOffsetSizes[g]);
							kernelChangeW.setArg(8, pActivators[g]->getKernelActivatorId());
							kernelChangeW.setArg(9, a);

							commandQueue.enqueueNDRangeKernel(kernelChangeW, cl::NullRange, cl::NDRange(wMatrix[g].size(), wMatrix[g][0].size()), cl::NDRange());
							commandQueue.finish();

							kernelChangeT.setArg(0, *clTmBuf);
							kernelChangeT.setArg(1, tmOffsetSizes[g]);
							kernelChangeT.setArg(2, *clErrBuf);
							kernelChangeT.setArg(3, pActivators[g]->getKernelActivatorId());
							kernelChangeT.setArg(4, *clLastSumBuf);
							kernelChangeT.setArg(5, a);

							commandQueue.enqueueNDRangeKernel(kernelChangeT, cl::NullRange, cl::NDRange(wMatrix[g][0].size()), cl::NDRange());
							commandQueue.finish();

							if (g == 0) break;
						}
					}

					iter_num++;
					glob_iter++;

					if (maxIter != 0)
						if (glob_iter >= maxIter)
							return 1;

					if (iter_num % check_step == 0)
					{
						currError = netError;
						netError = calculateError();

						if (netError <= targetError)
							return 0;

						if (netError == currError)
							return 2;

						iter_num = 0;
					}
				}
			}
			catch (...)
			{
				return -2;
			}
		}

	}
}