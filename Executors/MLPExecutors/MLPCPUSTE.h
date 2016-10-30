#pragma once

#include "IMLPExecutor.h"

namespace MentalAI
{
	namespace MLPExecutor
	{
		class MLPCPUSTE : public IMLPExecutor
		{
		public:

			MLPCPUSTE();

		private:

			std::vector<std::vector<std::vector<float>>> *wMatrix;
			std::vector<std::vector<float>> *tMatrix;
			std::vector<Activation::IActivation*> *activatonFunc;

			//======================================================================

			IMLPExecutor* createCopy();

			std::vector<float> getOutputLayerFromInput(std::vector<float> &input);

			int prepareExecutor(std::vector<std::vector<std::vector<float>>> &wMatrix, 
								std::vector<std::vector<float>> &tMatrix, 
								std::vector<Activation::IActivation*> &activatonFunc);

			void quitExecutor();
		};
	}
}