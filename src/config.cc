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
	int counter = 0;

	counter += fscanf(fp, " HP_CHAIN: %ms", &hpChain);
	dHPChain = std::string(hpChain);
	free(hpChain);
	counter += fscanf(fp, " RANDOM_SEED: %d", &dRandSeed);

	fclose(fp);

	if(counter != sParamCount){
		std::cerr << "Could not parse configuration file correctly.\n";
		std::exit(1);
	}
}
