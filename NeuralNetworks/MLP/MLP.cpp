#include "MLP.h"

namespace MentalAI
{
	MLP::MLP()
	{
		wMatrix.resize(1);
		tMatrix.resize(1);
		pActivators.resize(1);
		lockNetwork = false;
		pExecutor = nullptr;
	}

	MLP::MLP(std::vector<std::vector<std::vector<float>>> wMatrix, std::vector<std::vector<float>> tMatrix)
	{
		this->wMatrix = wMatrix;
		this->tMatrix = tMatrix;
		this->pActivators.resize(pActivators.size(), nullptr);

		lockNetwork = false;
	}
	
	MLP::MLP(const MLP& old)
	{
		wMatrix = old.wMatrix;
		tMatrix = old.tMatrix;
		lockNetwork = old.lockNetwork;

		pActivators.resize(old.pActivators.size());

		for (uint i = 0; i < pActivators.size(); i++)
			pActivators[i] = old.pActivators[i]->createCopy();

		pExecutor = old.pExecutor->createCopy();
	}

	MLP::~MLP()
	{
		wMatrix.clear();
		tMatrix.clear();

		for (uint i = 0; i < pActivators.size(); i++)
			if (pActivators[i] != nullptr) delete pActivators[i];

		pActivators.clear();

		if (pExecutor != nullptr)
			delete pExecutor;
	}

	uint MLP::getLayersCount()
	{
		return wMatrix.size() + 1;
	}

	uint MLP::getLayerSize(uint layerId)
	{
		if (wMatrix.size() < layerId) return 0;
		if (!wMatrix[layerId].size()) return 0;

		if (layerId != wMatrix.size()) return wMatrix[layerId].size();
		else return wMatrix[layerId][0].size();
	}

	bool MLP::setWeightMatrix(std::vector<std::vector<std::vector<float>>> wMatrix)
	{
		if (!lockNetwork)
		{
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
		else return false;
	}

	std::vector<std::vector<std::vector<float>>> MLP::getWeightMatrix()
	{
		return wMatrix;
	}

	bool MLP::setTMatrix(std::vector<std::vector<float>> tMatrix)
	{
		if (lockNetwork) return false;

		if (wMatrix.size() == tMatrix.size())
		{
			this->tMatrix = tMatrix;
			return true;
		}
		else return false;
	}

	std::vector<std::vector<float>> MLP::getTMatrix()
	{
		return tMatrix;
	}

	bool MLP::setActivators(std::vector<Activation::IActivation*> pActivators)
	{
		if (lockNetwork) return false;
		if (pActivators.size() != wMatrix.size()) return false;

		this->pActivators.resize(pActivators.size(), nullptr);

		for (uint i = 0; i < this->pActivators.size(); i++)
			this->pActivators[i] = pActivators[i];

		return true;
	}

	bool MLP::setExecutor(MLPExecutor::IMLPExecutor *pExecutor)
	{
		if (lockNetwork) return false;

		if (this->pExecutor != nullptr)	delete pExecutor;

		this->pExecutor = pExecutor;
		return true;
	}

	int MLP::prepareExecutor()
	{
		if (lockNetwork) return -1;

		if (wMatrix.size() != tMatrix.size() || wMatrix.size() != pActivators.size()
			|| wMatrix.size() < 1) return -2;
			
		for (uint i = 0; i < wMatrix.size(); i++)
			if (wMatrix[i][0].size() != tMatrix[i].size()) return -3;

		int prepRet = pExecutor->prepareExecutor(wMatrix, tMatrix, pActivators);
		
		if (prepRet == 0)
		{
			lockNetwork = true;
			return 0;
		}
		else return 0;

	}

	void MLP::quitExecutor()
	{
		if (!lockNetwork) return;

		pExecutor->quitExecutor();
		lockNetwork = false;
	}

	std::vector<float> MLP::getOutputLayer(std::vector<float> input)
	{
		if (!lockNetwork) return std::vector<float>();

		return pExecutor->getOutputLayerFromInput(input);
	}
}