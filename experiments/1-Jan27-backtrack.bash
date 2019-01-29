binary=$1;

declare TIMEFORMAT="%E";

echo "contacts,time";

for hpChain in PHPHPHHHPPHPHPPPPPPPPPPPHHP HHHHPPPPHPHHPPPHHPPPPPPPPPP PPPPPHHPHPHPHPHPPHHPHHPHPPP; do
	for execId in $(seq 100); do
		output=$( { time ./$binary -s $hpChain | cut -f 2 -d' '; } 2>&1 );
		echo $output | sed -e "s/ /,/g";
	done
done;
