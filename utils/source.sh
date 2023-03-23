#!/bin/bash
user="$USER"

if [ $user == "b***7" ]; then		# on server
        activatePath="/afs/math.uni-hamburg.de/users/stud/b***7/.../firedrake/bin/activate"		# ~ doesnt work somehow with variable in source
elif [ $user == "b***r" ]; then		# local
        activatePath="/home/b***r/local/simulation/firedrake/bin/activate"		# ~ doesnt work somehow with variable in source
        echo -n mpiexec -n 8 python3 rb.py | xclip -selection clipboard
else
	echo "not the right user"
	exit 1
fi
echo "sourcing "$activatePath
source $activatePath
$SHELL
