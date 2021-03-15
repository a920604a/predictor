#! /bin/bash
python3 ./image_predictor_db.py  \
		--model-info-file ./model_info_db.json \
		--mode db \
		--top1 \
		--draw-top1 \
        --pool-size 2 \
		--gpus 0
