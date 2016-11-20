#include "MLPCPUSTT.h"
#include <time.h>

#include <iostream>

namespace MentalAI
{
	namespace Teacher
	{
		float MLPCPUSTT::calculateError()
		{
			float err = 0;

			for (uint i = 0; i < trainingSet.size(); i++)
			{
				std::vector<float> out = getOutputLayerFromInput(trainingSet[i]);

				for (uint g = 0; g < out.size(); g++)
					err += (out[g] - refVal[i][g]) * (out[g] - refVal[i][g]);
			}

			return err / 2;
		}

		//This is recommended random for MLP with BPA
		void MLPCPUSTT::initRandomWM()
		{
			srand(time(0));

			for (uint i = 0; i < wMatrix.size(); i++)
			{
				float LO = -1.f * (1.f / sqrtf((float)wMatrix[i][0].size()));
				float HI = 1.f / sqrtf((float)wMatrix[i][0].size());

				for (uint g = 0; g < wMatrix[i].size(); g++)
				{
					for (uint z = 0; z < wMatrix[i][g].size(); z++)
						wMatrix[i][g][z] = LO + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (HI - LO)));
				}
			}
		}

		std::vector<float> MLPCPUSTT::getOutputLayerFromInput(std::vector<float> curLayer)
		{
			for (uint i = 0; i < bpaLasyY[0].size(); i++)
				bpaLasyY[0][i] = curLayer[i];

			for (uint i = 0; i < wMatrix.size(); i++)
			{
				std::vector<float> newLayer(wMatrix.operator[](i)[0].size(), 0);

				for (uint g = 0; g < newLayer.size(); g++)
				{
					float wSum = 0;

					for (uint z = 0; z < curLayer.size(); z++)
						wSum += wMatrix[i][z][g] * curLayer[z];

					wSum -= tMatrix[i][g];
					newLayer[g] = pActivators[i]->activate(wSum);

					bpaLastSum[i][g] = wSum;
				}

				if (i != wMatrix.size() - 1)
					for (uint g = 0; g < bpaLasyY[i + 1].size(); g++)
						bpaLasyY[i + 1][g] = newLayer[g];

				curLayer = newLayer;
			}

			return curLayer;
		}

		bool MLPCPUSTT::Compare(std::vector<float> out, std::vector<float> refVal)
		{
			bool flag = true;

			for (uint i = 0; i < out.size(); i++)
			{
				if (abs(out[i] - refVal[i]) > targetError)
				{
					flag = false;
					break;
				}
			}

			return flag;
		}

		std::vector<std::vector<float>> MLPCPUSTT::getLayersError(std::vector<float> out, uint index)
		{
			std::vector<std::vector<float>> err(tMatrix.size());

			for (uint i = 0; i < err.size(); i++)
				err[i].resize(tMatrix[i].size(), 0);

			//Set error of output layer
			for (uint i = 0; i < err[err.size() - 1].size(); i++)
				err[err.size() - 1][i] = out[i] - refVal[index][i];

			//For 2-layer NN
			if (err.size() == 1) return err;

			//Set error of hidden layers
			for (uint i = err.size() - 2;; i--)
			{
				for (uint g = 0; g < err[i].size(); g++)
				{
					for (uint z = 0; z < err[i + 1].size(); z++)
						err[i][g] += err[i + 1][z] * pActivators[i + 1]->derivActivate(bpaLastSum[i + 1][z]) * wMatrix[i + 1][g][z];
				}

				if (i == 0) break;
			}

			return err;
		}

		//====================================================================================================

		MLPCPUSTT::MLPCPUSTT()
		{
			a = 0.2f;
			targetError = 0.01f;
			check_step = 1000;
			currError = 0;
			lock = false;
		}

		MLPCPUSTT::MLPCPUSTT(const MLPCPUSTT& old)
		{
			a = old.a;
			check_step = old.check_step;
			targetError = old.targetError;
			currError = old.currError;
			lock = old.lock;

			trainingSet = old.trainingSet;
			refVal = old.refVal;
			wMatrix = old.wMatrix;
			tMatrix = old.tMatrix;
			bpaLastSum = old.bpaLastSum;
			bpaLasyY = old.bpaLasyY;
			
			pActivators.resize(old.pActivators.size(), nullptr);

			for (uint i = 0; i < pActivators.size(); i++)
				pActivators[i] = old.pActivators[i]->createCopy();
		}

		MLPCPUSTT::~MLPCPUSTT()
		{
			trainingSet.clear();
			refVal.clear();
			wMatrix.clear();
			tMatrix.clear();
			bpaLastSum.clear();
			bpaLasyY.clear();

			for (uint i = 0; i < pActivators.size(); i++)
				if (pActivators[i] != nullptr) delete pActivators[i];

			pActivators.clear();
		}

		bool MLPCPUSTT::loadTrainingSet(std::vector<std::vector<float>> trainSet)
		{
			if (lock) return false;
			if (trainSet.empty()) return false;			
			if (trainSet[0].empty()) return false;
			if (wMatrix.empty()) return false;
			if (trainSet[0].size() != wMatrix[0].size()) return false;

			this->trainingSet = trainSet;
			return true;
		}

		bool MLPCPUSTT::loadReferenceValues(std::vector<std::vector<float>> refvalue)
		{
			if (lock) return false;
			if (refvalue.size() != trainingSet.size() || trainingSet.empty()) return false;
			if (wMatrix[wMatrix.size() - 1][0].size() != refvalue[0].size()) return false;

			refVal = refvalue;
			return true;
		}

		int MLPCPUSTT::prepareTeacher()
		{
			if (lock) return -1;
			if (wMatrix.empty()) return -2;
			if (trainingSet.empty()) return -4;
			if (pActivators.empty()) return -5;
			if (a <= 0 || a > 1) return -6;
			if (targetError < 0 || targetError > 1) return -7;
			if (trainingSet.size() != refVal.size()) return -8;

			tMatrix.resize(wMatrix.size());
			bpaLastSum.resize(wMatrix.size());

			for (uint i = 0; i < wMatrix.size(); i++)
			{
				tMatrix[i].resize(wMatrix[i][0].size(), 0);
				bpaLastSum[i].resize(wMatrix[i][0].size(), 0);
			}

			bpaLasyY.resize(wMatrix.size());
			for (uint i = 0; i < wMatrix.size(); i++)
				bpaLasyY[i].resize(wMatrix[i].size(), 0);

			lock = true;
			return 0;
		}

		void MLPCPUSTT::quitTeacher()
		{
			lock = false;
		}

		bool MLPCPUSTT::setWeightMatrix(std::vector<std::vector<std::vector<float>>> wMatrix)
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

		bool MLPCPUSTT::setActivators(std::vector<Activation::IActivation*> pActivators)
		{
			if (lock) return false;
			if (pActivators.size() != wMatrix.size()) return false;

			this->pActivators.resize(pActivators.size(), nullptr);

			for (uint i = 0; i < this->pActivators.size(); i++)
				this->pActivators[i] = pActivators[i];

			return true;

		}

		std::vector<std::vector<std::vector<float>>> MLPCPUSTT::getNewWeightMatrix()
		{
			return wMatrix;
		}

		std::vector<std::vector<float>> MLPCPUSTT::getNewTMatrix()
		{
			return tMatrix;
		}

		bool MLPCPUSTT::setTrainStep(float a)
		{
			if (lock) return false;

			this->a = a;
			return true;
		}

		bool MLPCPUSTT::setTargetError(float err)
		{
			if (lock) return false;

			this->targetError = err;
			return true;
		}

		float MLPCPUSTT::getCurrError()
		{
			return currError;
		}

		int MLPCPUSTT::tryTrain(uint maxIter)
		{
			if (!lock) return -1;

			try 
			{
				//initRandomWM();

				bool exitFlag = false;
				uint iter_num = 0;
				uint glob_iter = 0;
				float netError = 0;

				while (!exitFlag)
				{
					for (uint i = 0; i < trainingSet.size(); i++)
					{
						std::vector<float> out = getOutputLayerFromInput(trainingSet[i]);

						if (Compare(out, refVal[i]))
							continue;

						std::vector<std::vector<float>> err = getLayersError(out, i);

						for (uint g = wMatrix.size() - 1;; g--)
						{
							for (uint z = 0; z < wMatrix[g].size(); z++)
								for (uint w = 0; w < err[g].size(); w++)
									wMatrix[g][z][w] -= a * err[g][w] * pActivators[g]->derivActivate(bpaLastSum[g][w]) * bpaLasyY[g][z];

							for (uint z = 0; z < err[g].size(); z++)
								tMatrix[g][z] += a * err[g][z] * pActivators[g]->derivActivate(bpaLastSum[g][z]);

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