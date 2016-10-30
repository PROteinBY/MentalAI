#pragma once

#include "IMLPTeacher.h"

namespace MentalAI
{
	typedef unsigned int uint;
	typedef unsigned long ulong;
	typedef unsigned short ushort;

	namespace Teacher
	{
		class MLPCPUSTT : public IMLPTeacher
		{
		private:

			std::vector<std::vector<float>> trainingSet;
			std::vector<std::vector<float>> refVal;
			std::vector<std::vector<std::vector<float>>> wMatrix;
			std::vector<std::vector<float>> tMatrix;
			std::vector<Activation::IActivation*> pActivators;
			std::vector<std::vector<float>> bpaLastSum;
			std::vector<std::vector<float>> bpaLasyY;				

			float a; //Training step
			float targetError;
			float currError;

			//Lock change network params.
			bool lock;

			uint check_step;

			//================================================================================

			float calculateError();
			void initRandomWM();			
			std::vector<float> getOutputLayerFromInput(std::vector<float>); //Modified for training
			bool Compare(std::vector<float> out, std::vector<float> refVal);
			std::vector<std::vector<float>> getLayersError(std::vector<float> out, uint index);

		public:

			MLPCPUSTT();
			MLPCPUSTT(const MLPCPUSTT& old);

			~MLPCPUSTT();

			bool loadTrainingSet(std::vector<std::vector<float>> trainSet);
			bool loadReferenceValues(std::vector<std::vector<float>> refvalue);
			int prepareTeacher();
			void quitTeacher();
			bool setWeightMatrix(std::vector<std::vector<std::vector<float>>> wMatrix);
			bool setActivators(std::vector<Activation::IActivation*> pActivators);
			int tryTrain();
			std::vector<std::vector<std::vector<float>>> getNewWeightMatrix();
			std::vector<std::vector<float>> getNewTMatrix();
			bool setTrainStep(float a);
			bool setTargetError(float err);
			float getCurrError();
		};
	}
}