#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

tag='release_test'
resultpath='/data1/output/rl_exposure_fusion_0919/'$tag'/'
if [ ! -d $resultpath ];then
	mkdir -p $resultpath
fi

python -u test.py\
	/data1/data/NightSight/testset/ \
	checkpoints/ \
	--data-type 'night' \
	--results $resultpath \
	--score-path $resultpath'score_syn.txt' \
	2>&1 | tee $resultpath'/test.log'