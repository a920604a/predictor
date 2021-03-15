# yuan-predictor

1. Build docker image depend on your situation. If you must build image ,  ```./build.sh```
2. ```./run.sh``` to start docker container , you would configure if you need 
3. the below , having three main programs
   - execute ```./start.sh``` : run image_predictor.py
   - execute ```./start_db.sh``` : run image_predictor_db.py
   - ```python send_mysql.py``` : send local sqlite db data to remote mysql server  


- file structure 
    - app--> main function
    - algorithm --> the same as yuan-backend algorithm
    - build.sh :  Build docker script
    - Dockerfile :  Build docker image
    - run.sh : Start docker container
    - Shanghai 

- image_predictor.py 
  - feature 
    - draw visual image [--draw-top1 --draw-total]
    - write csv record [--top1 --total]
    - write auto-labelme image [--auto-labelme]
    - multi-gpus [--gpus] to accelerate
    - multi-threading to accelerate 
  - 
- image_predictor_db.py
  - feature 
    - draw visual image [--draw-top1 --draw-total]
    - write csv record [--top1 --total]
    - write auto-labelme image [--auto-labelme]
    - multi-gpus [--gpus] to accelerate , but bind to product
    - multi-threading to accelerate 
- send_mysql.py

- Other details about config of image_predictor and image_predictor_db , you could see model_info.json  **and** model_info_db.json 
  