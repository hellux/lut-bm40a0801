# run Matlab scripts from command line
# usage: ./matlab.sh FILE...

for prgm in $@; do
    dir="$(dirname $prgm)"
    name="$(basename $prgm)"
    init_code="addpath('$dir');run('$name');exit;"

    matlab -nodisplay -nosplash -nodesktop -r "$init_code"
done
