#include "Relu.h"

namespace MentalAI
{
	namespace Activation
	{
		Relu::Relu()
		{
			k = (float)0.01;
		}

		Relu::Relu(float k)
		{
			this->k = k;
		}

		float Relu::activate(float wSum)
		{
			return wSum > 0 ? wSum : k * wSum;
		}

		float Relu::derivActivate(float y)
		{
			return y > 0 ? 1 : k;
		}

		int Relu::getKernelActivatorId()
		{
			return 2; //2 is id of Relu activation
		}

		IActivation* Relu::createCopy()
		{
			return new Relu(k);
		}
	}
}