#ifndef LANCZOS_TYPEDEFS
#define LANCZOS_TYPEDEFS

struct LANCZOS
{
	struct REORTHO    {enum OPTION {NO, FULL, SIMON, GRCAR};};
	struct EFFICIENCY {enum OPTION {TIME, MEMORY};};
	struct EDGE       {enum OPTION {GROUND, ROOF};};
	
	struct STAT
	{
		int N_reortho;
		int N_mvm;
		int last_invSubspace;
		int N_restarts;
		int last_N_iter;
		double last_memory;
		bool BREAK;
		
		void reset()
		{
			N_reortho = N_mvm = last_invSubspace = N_restarts = last_N_iter = last_memory = 0;
			BREAK = false;
		}
	};
};

#endif
