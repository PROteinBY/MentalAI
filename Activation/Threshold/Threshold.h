#pragma once

#include "../IActivation.h"

namespace MentalAI
{
	namespace Activation
	{
		class Threshold : public IActivation
		{
		public:

			Threshold();

			float activate(float wSum);
			float derivActivate(float wSum);
			int getKernelActivatorId();
			
			IActivation* createCopy();
		};
	}
}
