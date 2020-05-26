#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

resultpath="YOU PATH FOR SAVEING RESULTS"

if [ ! -d $resultpath ];then
	mkdir -p $resultpath
fi

data_type="night" # or "day"

python -u test.py\
	"YOU TEST PATH" \
	"YOU CHECKPOINTS PATH" \
	--data-type $data_type \
	--results $resultpath \
	--score-path $resultpath'score_syn.txt'