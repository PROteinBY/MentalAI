#pragma once

namespace MentalAI
{
	namespace Activation
	{
		//Interface for all activation classes
		class IActivation abstract
		{
		public:

			//Create copy
			virtual IActivation* createCopy() = 0;

			/*
			y = F(S)
			*/
			virtual float activate(float wSum) = 0;

			/*
			F'(S)
			*/
			virtual float derivActivate(float y) = 0;

			/*
			Get activation function id in kernel source
			return -1 if not implemented in kernel source
			*/
			virtual int getKernelActivatorId() = 0;
		};
	}
}