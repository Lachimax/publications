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


if [[ ${do_imaging} == 1 ]]; then
  python 00-imaging.py
fi
if [[ ${do_table} == 1 ]]; then
  python 01-frb_table.py
fi

scripts=( "1?-*.py" )
for script in ${scripts[@]}; do
  echo $script
  python $script
done