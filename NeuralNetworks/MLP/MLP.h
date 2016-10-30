#pragma once

#include <vector>

#include "../../Activation/IActivation.h"
#include "../../Executors/MLPExecutors/IMLPExecutor.h"

namespace MentalAI
{
	typedef unsigned int uint;
	typedef unsigned long ulong;
	typedef unsigned short ushort;

	//MultiLayer Perceptron
	class MLP
	{
	private:

		/*Full weight matrix. First is layer id (between layer i and j)
		than i-layer neuron index, and tran j-layer neuron index*/
		std::vector<std::vector<std::vector<float>>> wMatrix;

		//Weighting factors matrix
		std::vector<std::vector<float>> tMatrix;

		//Pointers to activation function classes (one per layer except input), use for Cpp Implementation
		std::vector<Activation::IActivation*> pActivators;

		//if true - network ready to execute and lock to change, and revers
		bool lockNetwork;

		//Pointer to executor
		MLPExecutor::IMLPExecutor *pExecutor;

	public:
		
		//2-layer NN (input and output)
		MLP();

		/*Create empty weight matrix with [size] layers and Relu activation
		in hidden layers and Sigmoid activation in output layer*/
		MLP(std::vector<std::vector<std::vector<float>>> wMatrix, 
			std::vector<std::vector<float>> tMatrix);

		//Copy
		MLP(const MLP& old);

		//Clear mem
		~MLP();

		//Return layers count
		uint getLayersCount();

		//Return size of layer[layerID]
		uint getLayerSize(uint layerId);

		//Set weight matrix. 
		bool setWeightMatrix(std::vector<std::vector<std::vector<float>>> wMatrix);

		//Return weight matrix copy
		std::vector<std::vector<std::vector<float>>> getWeightMatrix();

		//Set T const matrix
		bool setTMatrix(std::vector<std::vector<float>> tMatrix);

		//Return T matrix copy
		std::vector<std::vector<float>> getTMatrix();

		//set activators (copy). Clean after using.
		bool setActivators(std::vector<Activation::IActivation*> pActivators);

		//Set executor for NN
		bool setExecutor(MLPExecutor::IMLPExecutor *pExecutor);

		//Prepare data for executor, compil kernels, lock network
		int prepareExecutor();

		//Clear device memory, unluck network
		void quitExecutor();

		//Calculations on NN executor
		std::vector<float> getOutputLayer(std::vector<float> input);
	};	
}