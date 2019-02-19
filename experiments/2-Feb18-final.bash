
TIMEFORMAT="%E";

#prog_bt
#prog_bt_omp
#prog_nbt
#prog_nbt_cuda
#prog_nbt_omp

PROTSIZE=$(seq 64 64 1024);
N_ITERATIONS=10;

BINARY=prog_bt
{
	echo "protSize,wallTime";
	for protSize in $PROTSIZE; do
		protSequence=$( ./tools/genProt $protSize );
		
		for i in $(seq $N_ITERATIONS); do
			echo -n "$protSize"",";
			{ time ./$BINARY -s $protSequence &> /dev/null; } 2>&1 | xargs echo -n;
			echo;
		done
	done
} > ${BINARY}_time.out

BINARY=prog_bt_omp
{
	echo "protSize,wallTime";
	for protSize in $PROTSIZE; do
		protSequence=$( ./tools/genProt $protSize );
		
		for i in $(seq $N_ITERATIONS); do
			echo -n "$protSize"",";
			{ time ./$BINARY -s $protSequence &> /dev/null; } 2>&1 | xargs echo -n;
			echo;
		done
	done
} > ${BINARY}_time.out

BINARY=prog_nbt
{
	echo "protSize,wallTime";
	for protSize in $PROTSIZE; do
		protSequence=$( ./tools/genProt $protSize );
		
		for i in $(seq $N_ITERATIONS); do
			echo -n "$protSize"",";
			{ time ./$BINARY -s $protSequence &> /dev/null; } 2>&1 | xargs echo -n;
			echo;
		done
	done
} > ${BINARY}_time.out

BINARY=prog_nbt_cuda
{
	echo "protSize,wallTime";
	for protSize in $PROTSIZE; do
		protSequence=$( ./tools/genProt $protSize );
		
		for i in $(seq $N_ITERATIONS); do
			echo -n "$protSize"",";
			{ time ./$BINARY -s $protSequence &> /dev/null; } 2>&1 | xargs echo -n;
			echo;
		done
	done
} > ${BINARY}_time.out

BINARY=prog_nbt_omp
{
	echo "protSize,wallTime";
	for protSize in $PROTSIZE; do
		protSequence=$( ./tools/genProt $protSize );
		
		for i in $(seq $N_ITERATIONS); do
			echo -n "$protSize"",";
			{ time ./$BINARY -s $protSequence &> /dev/null; } 2>&1 | xargs echo -n;
			echo;
		done
	done
} > ${BINARY}_time.out
