#pragma once

#include "../IActivation.h"

namespace MentalAI
{
	namespace Activation
	{
		class Linear : public IActivation
		{
		private:

			float k;

		public:

			Linear();

			float activate(float wSum);
			float derivActivate(float wSum);
			int getKernelActivatorId();

			IActivation* createCopy();
		};
	}
}
#pragma once
