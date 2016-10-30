#pragma once

#include <vector>

#include "../../Activation/IActivation.h"

namespace MentalAI
{
	namespace Teacher
	{
		class IMLPTeacher abstract
		{
		public:

			//load training set to divece mem
			virtual bool loadTrainingSet(std::vector<std::vector<float>> trainSet) = 0;

			//load reference value for training set
			virtual bool loadReferenceValues(std::vector<std::vector<float>> refvalue) = 0;

			//Compil kernel, check sizes
			virtual int prepareTeacher() = 0;

			//Clear mem
			virtual void quitTeacher() = 0;

			//Set config of network and start values
			virtual bool setWeightMatrix(std::vector<std::vector<std::vector<float>>> wMatrix) = 0;

			//set activator
			virtual bool setActivators(std::vector<Activation::IActivation*> pActivators) = 0;

			//Generate random WM and try train
			virtual int tryTrain() = 0;

			//After train you can get new WM
			virtual std::vector<std::vector<std::vector<float>>> getNewWeightMatrix() = 0;

			//After train you can get new TM
			virtual std::vector<std::vector<float>> getNewTMatrix() = 0;

			//Train step (0 ... 1)
			virtual bool setTrainStep(float a) = 0;

			//Set target error for MLP
			virtual bool setTargetError(float err) = 0;

			//After training the error may differ from the desired
			virtual float getCurrError() = 0;
		};
	}
}