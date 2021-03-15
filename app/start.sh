#! /bin/bash
python3 ./image_predictor.py  \
		--model-info-file ./model_info.json \
		--input-folder-path  /data/data/data-test/100 \
		--result-folder /output/data/predict_folder \
		--draw-top1 \
		--mode csv \
		--auto-labelme \
		--top1 \
		--total \
        --pool-size 1 \
		--gpus 0
        