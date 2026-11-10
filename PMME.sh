bash train_one_OT-pems.sh pems-bay 5 0  # thread gpu_id
wait
bash train_one_OT-chengdu.sh chengdu_m 5 0
wait
bash train_one_OT-metr.sh metr-la 5 0
wait
bash train_one_OT-shenzhen.sh shenzhen 5 0

