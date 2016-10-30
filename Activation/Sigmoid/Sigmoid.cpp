#include "Sigmoid.h"
#include <math.h>

namespace MentalAI
{
	namespace Activation
	{
		Sigmoid::Sigmoid() {}

		float Sigmoid::activate(float wSum)
		{
			return 1.f / (1.f + (float)exp(-wSum));
		}

		float Sigmoid::derivActivate(float wSum)
		{
			float y = activate(wSum);
			return y * (1 - y);
		}

		int Sigmoid::getKernelActivatorId()
		{
			return 1; //1 is id of sigmoid function
		}

		IActivation* Sigmoid::createCopy()
		{
			return new Sigmoid();
		}
	}
}