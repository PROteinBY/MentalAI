#include "Threshold.h"

namespace MentalAI
{
	namespace Activation
	{
		Threshold::Threshold() {}

		float Threshold::activate(float wSum)
		{
			return wSum >= 0 ? (float)1 : (float)0;
		}

		float Threshold::derivActivate(float y)
		{
			return 0;
		}

		int Threshold::getKernelActivatorId()
		{
			return 3; //3 is id of Threshold activation
		}

		IActivation* Threshold::createCopy()
		{
			return new Threshold();
		}
	}
}