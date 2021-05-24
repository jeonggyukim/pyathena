#!/usr/bin/env bash

# Wrapper for join_vtk program
# Author: Jeong-Gyu Kim
# Date: March 2017

# set -e stops the execution of a script if a command or pipeline has an error
set -e

# Set join_vtk.c path
ATHENAVTK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
join_vtk_c="$ATHENAVTK_DIR/join_vtk.c"
join_vtk=${join_vtk_c%.*}

usage () {
  cat <<EOM
  $0 -r <START:END[:STRIDE]> [-b BASENAME] [-i INDIR] [-o OUTDIR] [-s SUFFIX] [-C] [-V]
  -b <BASENAME>: Basename (problem_id) of joined vtk files, e.g., BASENAME.xxxx[.SUFFIX].vtk. Default: Same as original.
  -i <INDIR>: Directory where vtk files are located (INDIR/id*/). Default: Current directory.
              Takes both relative and absolute paths.
  -o <OUTDIR>: Directory where joined vtk files will be stored. Default: INDIR
               Takes both relative and absolute paths.
  -s <SUFFIX>: Suffix (for 2d vtk)
  -C Compile join_vtk.c file
  -V Produce verbose messages."
EOM
  exit 0
}

indir=""
outdir=""
outbasename=""
suffix=""
compile=0
verbose=0
while getopts hb:i:o:r:s:CV opt
do
    case "$opt" in
	h) usage;;
	b) outbasename=$OPTARG;;
	i) indir=$OPTARG;;
	o) outdir=$OPTARG;;
	r) range=$OPTARG;;
        s) suffix=$OPTARG;;
	C) compile=1;;
	V) verbose=1;;
	\?) usage;;
    esac
done

if [ -n "$indir" ]; then 	# if indir is not null
    indir=$(cd ${indir} && pwd)
else
    indir=$(pwd)
fi

if [ -n "$outdir" ]; then
    [[ -d "$outdir" ]] || mkdir -p "$outdir"
    outdir=$(cd ${outdir} && pwd)
else
    outdir=$indir
fi

# get nproc, basename
nproc=$(find ${indir} -maxdepth 1 -type d -name 'id*[0-9]' | wc -l)
f=( $(find ${indir}/id0 -name "*.vtk") ) ; f=${f[0]} ; f=${f##*/}
basename=${f%%.*}

if [ -z "$outbasename" ]; then
    outbasename=$basename
fi

if [ ! -f "$join_vtk" ]; then
    compile=1
fi

# compile join_vtk.c
if [ $compile == 1 ]; then
    if [ $verbose == 1 ];  then
	gcc -o $join_vtk $join_vtk_c -lm -D VERBOSE
    else
	gcc -o $join_vtk $join_vtk_c -lm
    fi
    if [ $? -ne 0 ]; then
	echo "Complie failed!"
	exit 1
    fi
fi

# Parse the range variable
if [[ ${range##*:*} != "" ]]; then
    usage
else
    start=${range%%:*}
    rest=${range#*:}
    end=${rest%%:*}
    stride=${rest#*:}
    if [[ ${rest##*:*} != "" ]]; then
	stride="1"
    else
	stride=${rest#*:}
    fi
fi

dotsuffix=""
if [[ $suffix != "" ]]; then
  dotsuffix=.${suffix}
  #[[ -d ${outdir}/${suffix} ]] || mkdir -p ${outdir}/${suffix}
  #outdir=${outdir}/${suffix}
fi

#echo indir: $indir
#echo outdir: $outdir
echo steps: $(seq -s" " $start $stride $end)
#echo nproc: $nproc
#echo suffix: $suffix
#echo basename: $basename

for c in $(seq $start $stride $end)
do
    num=$(printf "%04d" $c)
    filepattern=${basename}*.${num}${dotsuffix}.vtk
    vtkfiles=( $(find ${indir}/id* -mindepth 1 -name "$filepattern") )
    # echo $num
    # echo $filepattern
    #echo ${vtkfiles[@]}
    #echo ${outdir}/${outbasename}.${num}${dotsuffix}.vtk
    #echo $join_vtk -o ${outdir}/${outbasename}.${num}${dotsuffix}.vtk ${vtkfiles[@]}
    $join_vtk -o ${outdir}/${outbasename}.${num}${dotsuffix}.vtk ${vtkfiles[@]}
done
