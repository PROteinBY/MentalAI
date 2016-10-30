#include "Linear.h"

namespace MentalAI
{
	namespace Activation
	{
		Linear::Linear() {}

		float Linear::activate(float wSum)
		{
			return wSum;
		}

		float Linear::derivActivate(float y)
		{
			return 1;
		}

		int Linear::getKernelActivatorId()
		{
			return 4; //2 is id of Relu activation
		}

		IActivation* Linear::createCopy()
		{
			return new Linear();
		}
	}
}