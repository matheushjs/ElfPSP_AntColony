#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

const char *Config::sFilename = "configuration.yml";

Config::Config(){
	FILE *fp = fopen(sFilename, "r");
	if(!fp){
		std::cerr << "Configuration file does not exist!\n";
		std::exit(1);
	}

	char *hpChain;
	int counter = fscanf(fp,
		// configuration.yml format
		" HP_CHAIN: %ms "
		" CYCLES: %d "
		" N_ANTS: %d "
		" ACO_ALPHA: %lf "
		" ACO_BETA: %lf "
		" RANDOM_SEED: %d ",
		// Destination variables
		&hpChain,
		&dCycles,
		&dNAnts,
		&dAcoAlpha,
		&dAcoBeta,
		&dRandSeed
	);

	dHPChain = std::string(hpChain);
	free(hpChain);

	fclose(fp);

	if(counter != 6){
		std::cerr << "Could not parse configuration file correctly.\n";
		std::exit(1);
	}
}
