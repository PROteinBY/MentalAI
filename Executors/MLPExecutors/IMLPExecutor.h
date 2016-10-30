#pragma once

#include <vector>

#include "../../Activation/IActivation.h"

namespace MentalAI
{
	typedef unsigned int uint;
	typedef unsigned long ulong;
	typedef unsigned short ushort;

	//MLP will be a friend class for all MLP executors
	class MLP;

	namespace MLPExecutor
	{
		class IMLPExecutor abstract
		{		
			//Only MLP class can use this methods
			friend MLP;

			//Create copy
			virtual IMLPExecutor* createCopy() = 0;
			
			//Get output layer from NN. Input ptr to data container or buffer.		
			virtual std::vector<float> getOutputLayerFromInput(std::vector<float> &input) = 0;

			//Prepare device, data for executing
			virtual int prepareExecutor(std::vector<std::vector<std::vector<float>>> &wMatrix, 
										std::vector<std::vector<float>> &tMatrix, 
										std::vector<Activation::IActivation*> &activatonFunc) = 0;
		
			//Clear mem
			virtual void quitExecutor() = 0;
		};
	}
}