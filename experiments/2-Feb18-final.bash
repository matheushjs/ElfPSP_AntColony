
TIMEFORMAT="%E";

echo "protSize,wallTime";

for protSize in $( seq 64 64 1024 ); do
	protSequence=$( ./tools/genProt $protSize );
	
	for i in $(seq 100); do
		echo -n "$protSize"",";
		{ time ./prog_nbt &> /dev/null; } 2>&1 | xargs echo -n;
		echo;
	done
done
