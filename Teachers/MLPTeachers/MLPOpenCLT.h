#pragma once

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "IMLPTeacher.h"
#include "CL/cl.hpp"

namespace MentalAI
{
	typedef unsigned int uint;
	typedef unsigned long ulong;
	typedef unsigned short ushort;

	namespace Teacher
	{
		class MLPOpenCLT : public IMLPTeacher
		{
		private:

			cl::Context context;                            /* CL context */
			std::vector<cl::Device> devices;                /* CL device list */
			std::vector<cl::Device> device;                 /* CL device to be used */
			std::vector<cl::Platform> platforms;			/* list of platforms */
			cl::CommandQueue commandQueue;                  /* CL command queue */
			cl::Program program;                            /* CL program  */
			cl::Kernel kernelMult;                          /* CL kernel */
			cl::Kernel kernelDevActivation;					/* CL kernel */
			cl::Kernel kernelActivation;                    /* CL kernel */
			cl::Kernel kernelZero;							/* CL kernel */
			cl::Kernel kernelCompare;						/* CL kernel */
			cl::Kernel kernelErrOutput;						/* CL kernel */
			cl::Kernel kernelErrHidden;						/* CL kernel */
			cl::Kernel kernelFullErr;						/* CL kernel */
			cl::Kernel kernelChangeW;						/* CL kernel */
			cl::Kernel kernelChangeT;						/* CL kernel */
			
			cl::Buffer *clWmBuf;							/* Buffer for weight matrix */
			cl::Buffer *clErrBuf;							/* Buffer for errors */
			cl::Buffer *clInpLayer;							/* Buffer for input layer */
			cl::Buffer *clOutLayer;							/* Buffer for output layer */
			cl::Buffer *clTmBuf;							/* Buffer for T matrix */
			cl::Buffer *clActBuf;							/* Buffer for activators id */
			cl::Buffer *clLastSumBuf;						/* Buffer for last weighted sum */
			cl::Buffer *clLasyYBuf;							/* Buffer for last Y values */
			cl::Buffer *clTrainSetBuf;						/* Buffer for train set */
			cl::Buffer *clRefValueBuf;						/* Buffer for reference values */

			size_t kernelWorkGroupSize;                     /* Group Size returned by kernel */
			size_t maxLayerSize;							/* Size for temp layers */

			bool deviceReady;								/* Flag needed for executing */

			//================================================================================

			std::vector<std::vector<float>> trainingSet;
			std::vector<std::vector<float>> refVal;
			std::vector<std::vector<std::vector<float>>> wMatrix;
			std::vector<size_t> wmOffsetSizes;
			std::vector<size_t> tmOffsetSizes; // Also used by clLastSubBuf and clErrBuf
			std::vector<size_t> lastYOffsetSizes;
			std::vector<std::vector<float>> tMatrix;
			std::vector<Activation::IActivation*> pActivators;

			float a; //Training step
			float targetError;
			float currError;

			uint tmFullSize;
			uint wmFullSize;

			//Lock change network params.
			bool lock;

			uint check_step;

			//================================================================================

			float calculateError();
			cl::Buffer* getOutputLayerFromInput(uint index); //Modified for training
			bool Compare(cl::Buffer *out, uint index);
			void getLayersError(cl::Buffer *out, uint index);
			std::vector<cl::Device> getDeviceList();
			bool initPreparation();

		public:

			MLPOpenCLT();
			MLPOpenCLT(cl::Device device);

			~MLPOpenCLT();

			bool loadTrainingSet(std::vector<std::vector<float>> trainSet);
			bool loadReferenceValues(std::vector<std::vector<float>> refvalue);
			int prepareTeacher();
			void quitTeacher();
			bool setWeightMatrix(std::vector<std::vector<std::vector<float>>> wMatrix);
			bool setActivators(std::vector<Activation::IActivation*> pActivators);
			int tryTrain(uint maxIter = 0);
			std::vector<std::vector<std::vector<float>>> getNewWeightMatrix();
			std::vector<std::vector<float>> getNewTMatrix();
			bool setTrainStep(float a);
			bool setTargetError(float err);
			float getCurrError();
		};
	}
}