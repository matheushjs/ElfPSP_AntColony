#include "acopredictor.h"

ACOPredictor::ACOPredictor(const HPChain &chain)
: hpchain(chain){
	
}

MovChain ACOPredictor::predict(){
	return MovChain();
}
