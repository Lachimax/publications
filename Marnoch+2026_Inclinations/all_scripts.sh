if [ -z "${1}" ]; then
    do_table=0
else
    do_table=${1}
fi
if [ -z "${2}" ]; then
    do_imaging=0
else
    do_imaging=${2}
fi

cd ~/Projects/PyCRAFT/publications/Marnoch2024_PhD-Thesis/frb_populations/

scripts=( "1?-*.py" )
for script in ${scripts[@]}; do
  echo $script
  python $script
done