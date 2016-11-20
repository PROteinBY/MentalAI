#include "MLPCPUSTE.h"
#include <fstream>

namespace MentalAI
{
	namespace MLPExecutor
	{
		MLPCPUSTE::MLPCPUSTE() 
		{
			wMatrix = nullptr;
			tMatrix = nullptr;
			activatonFunc = nullptr;
		}

		IMLPExecutor* MLPCPUSTE::createCopy()
		{
			MLPCPUSTE *newExecutor = new MLPCPUSTE();
			newExecutor->prepareExecutor(*wMatrix, *tMatrix, *activatonFunc);

			return newExecutor;
		}

		int MLPCPUSTE::prepareExecutor(std::vector<std::vector<std::vector<float>>> &wMatrix,
										std::vector<std::vector<float>> &tMatrix,
										std::vector<Activation::IActivation*> &activatonFunc)
		{
			this->wMatrix = &wMatrix;
			this->tMatrix = &tMatrix;
			this->activatonFunc = &activatonFunc;

			return 0;
		}

		void MLPCPUSTE::quitExecutor()
		{
			wMatrix = nullptr;
			tMatrix = nullptr;
			activatonFunc = nullptr;
		}

		std::vector<float> MLPCPUSTE::getOutputLayerFromInput(std::vector<float> &input)
		{
			if (input.size() != wMatrix->operator[](0).size()) 
				return std::vector<float>();
					
			std::vector<float> curLayer(input);

			for (uint i = 0; i < wMatrix->size(); i++)
			{
				std::vector<float> newLayer(wMatrix->operator[](i)[0].size(), 0);

				for (uint g = 0; g < newLayer.size(); g++)
				{
					float wSum = 0;

					for (uint z = 0; z < curLayer.size(); z++)
						wSum += wMatrix->operator[](i)[z][g] * curLayer[z];

					wSum -= tMatrix->operator[](i)[g];
					newLayer[g] = activatonFunc->operator[](i)->activate(wSum);
				}

				curLayer = newLayer;
			}

			return curLayer;
		}
	}
}