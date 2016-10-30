#pragma once

#include "../IActivation.h"

namespace MentalAI
{
	namespace Activation
	{
		class Relu : public IActivation
		{
		private:

			float k;

		public:

			Relu();
			Relu(float k);

			float activate(float wSum);
			float derivActivate(float wSum);
			int getKernelActivatorId();
			
			IActivation* createCopy();
		};
	}
}
