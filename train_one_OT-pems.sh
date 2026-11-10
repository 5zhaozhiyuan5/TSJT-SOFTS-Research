test_dataset=$1
threads=$2
gpu=$3

train_epochs=3000 #1000
finetune_epochs=0 #6000 ##500 #500


if [ ${test_dataset} == 'chengdu_m' ]; then
    data_list='shenzhen_pems_metr'
fi

if [ ${test_dataset} == "shenzhen" ]; then
    data_list='chengdu_metr_pems'
fi

if [ ${test_dataset} == 'metr-la' ]; then
    data_list='chengdu_shenzhen_pems'
fi

if [ ${test_dataset} == 'pems-bay' ]; then
    data_list='chengdu_shenzhen_metr'
fi

mkdir -p ./out/${data_list}
mkdir -p ./out_COT/${data_list}


finetune_epochs=6000
for model_name in 'SOFTS'; do  #  
    for projection in 1; do 
        for DAcoef in 1; do  #
            for classifier in 0.5; do  
                for seed in 1 3 5; do  
                    for DAepoch in 1500; do  
                        for xySquaredRatio in 100; do  #  
                            for MemSize in 10000 ; do  
                                for DA in 'OT' ; do
                                    OMP_NUM_THREADS=${threads} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u train_DA_OT.py  \
                                        --finetune_epochs ${finetune_epochs} \
                                        --train_epochs ${train_epochs} \
                                        --data_list ${data_list} \
                                        --test_dataset ${test_dataset} \
                                        --config_file ./configs/config.yaml  \
                                        --seed ${seed} \
                                        --classifier ${classifier} \
                                        --xySquaredRatio ${xySquaredRatio} \
                                        --DAcoef ${DAcoef} \
                                        --DAepoch ${DAepoch} \
                                        --MemSize ${MemSize} \
                                        --projection ${projection} \
                                        --DA ${DA} \
                                        --model_name ${model_name} > ./out_COT/${data_list}/${test_dataset}_${DA}_${model_name}_daE${DAepoch}_daC${DAcoef}_cls${classifier}_seed${seed}_Proj${projection}_MemSize${MemSize}_xyR${xySquaredRatio}.out 2>&1
                                done
                            done
                        done 
                    done 
                done 
            done 
        done 
    done 
done


for model_name in 'iTransformer'; do  #  
    for projection in 1; do  
        for DAcoef in 0.01; do  #
            for classifier in 0.2; do  
                for seed in 1 3 5; do 
                    for DAepoch in 1500; do  
                        for xySquaredRatio in 100; do  #  
                            for MemSize in 10000 ; do 
                                for DA in 'OT' ; do
                                    OMP_NUM_THREADS=${threads} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u train_DA_OT.py  \
                                        --finetune_epochs ${finetune_epochs} \
                                        --train_epochs ${train_epochs} \
                                        --data_list ${data_list} \
                                        --test_dataset ${test_dataset} \
                                        --config_file ./configs/config.yaml  \
                                        --seed ${seed} \
                                        --classifier ${classifier} \
                                        --xySquaredRatio ${xySquaredRatio} \
                                        --DAcoef ${DAcoef} \
                                        --DAepoch ${DAepoch} \
                                        --MemSize ${MemSize} \
                                        --projection ${projection} \
                                        --DA ${DA} \
                                        --model_name ${model_name} > ./out_COT/${data_list}/${test_dataset}_${DA}_${model_name}_daE${DAepoch}_daC${DAcoef}_cls${classifier}_seed${seed}_Proj${projection}_MemSize${MemSize}_xyR${xySquaredRatio}.out 2>&1
                                done
                            done
                        done 
                    done 
                done 
            done 
        done 
    done 
done


