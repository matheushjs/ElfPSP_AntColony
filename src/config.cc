#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

const char *Config::sFilename = "configuration.yml";

/* We simply "parse" the file using fscanf, which takes care of white space for us. */
Config::Config(){
	FILE *fp = fopen(sFilename, "r");
	if(!fp){
		std::cerr << "Configuration file does not exist!\n";
		std::exit(1);
	}

	char *hpChain;
	char *filename;
	int counter = fscanf(fp,
		// configuration.yml format
		" HP_CHAIN: %ms "
		" CYCLES: %d "
		" N_ANTS: %d "
		" ACO_ALPHA: %lf "
		" ACO_BETA: %lf "
		" ACO_EVAPORATION: %lf "
		" LS_FREQUENCY: %d "
		" EXCHANGED_ANTS: %d"
		" RANDOM_SEED: %d "
		" STRUCTURE_FILENAME: %ms ",
		// Destination variables
		&hpChain,
		&dCycles,
		&dNAnts,
		&dAcoAlpha,
		&dAcoBeta,
		&dAcoEvaporation,
		&dLSFrequency,
		&dExchangedAnts,
		&dRandSeed,
		&filename
	);

	dHPChain = std::string(hpChain);
	dFilename = std::string(filename);
	free(hpChain);
	free(filename);

	fclose(fp);

	if(counter != 10){
		std::cerr << "Could not parse configuration file correctly.\n";
		std::exit(1);
	}
}
