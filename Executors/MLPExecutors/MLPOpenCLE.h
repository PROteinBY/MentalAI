#pragma once

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "CL/cl.hpp"
#include "IMLPExecutor.h"

namespace MentalAI
{
	namespace MLPExecutor
	{
		class MLPOpenCLE : public IMLPExecutor
		{
		public:

			//=========================== STATIC METHODS ===========================

			static std::vector<cl::Device> getDeviceList();

			//======================================================================

			MLPOpenCLE();						   //Use max CU device for executing
			MLPOpenCLE(cl::Device device);		   //Use 'device' for executing 

			bool initPreparation();

		private:

			//=============================== FIELDS ===============================

			std::vector<std::vector<std::vector<float>>> *wMatrix;
			std::vector<std::vector<float>> *tMatrix;
			std::vector<Activation::IActivation*> *activatonFunc;
			std::vector<size_t> wmOffsetSizes;

			cl::Context context;                            /* CL context */
			std::vector<cl::Device> devices;                /* CL device list */
			std::vector<cl::Device> device;                 /* CL device to be used */
			std::vector<cl::Platform> platforms;			/* list of platforms */
			cl::CommandQueue commandQueue;                  /* CL command queue */
			cl::Program program;                            /* CL program  */
			cl::Kernel kernelMult;                          /* CL kernel */
			cl::Kernel kernelActivation;                    /* CL kernel */
			cl::Kernel kernelZero;							/* CL kernel */

			cl::Buffer *clWmBuf;							/* Buffer for weight matrix */
			cl::Buffer *clInpLayer;							/* Buffer for input layer */
			cl::Buffer *clOutLayer;							/* Buffer for output layer */
			cl::Buffer *clTmBuf;							/* Buffer for T matrix */
			cl::Buffer *clActBuf;							/* Buffer for activators id */

			size_t kernelWorkGroupSize;                     /* Group Size returned by kernel */
			size_t maxLayerSize;							/* Size for temp layers */

			bool deviceReady;								/* Flag needed for executing */

			//======================= IMLPExecutor methods =========================   

			IMLPExecutor* createCopy();

			std::vector<float> getOutputLayerFromInput(std::vector<float> &input);

			int prepareExecutor(std::vector<std::vector<std::vector<float>>> &wMatrix,
				std::vector<std::vector<float>> &tMatrix,
				std::vector<Activation::IActivation*> &activatonFunc);

			void quitExecutor();

			//======================================================================
		};
	}
}