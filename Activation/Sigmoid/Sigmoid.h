#pragma once

#include "../IActivation.h"

namespace MentalAI
{
	namespace Activation
	{
		class Sigmoid : public IActivation
		{
		public:

			Sigmoid();

			float activate(float wSum);
			float derivActivate(float wSum);
			int getKernelActivatorId();
			
			IActivation* createCopy();
		};
	}
}