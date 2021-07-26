# 0.1 pruning, change prune.c, than make 
LOG=prune_res/5_images_0.1_pruning.log
for name in "cat" "dog" "eagle" "giraffe" "horses"
do
    for net in "alexnet" "mobilenetv2" "densenet201" "resnet50"
    do
        echo ${net}>>${LOG}
        if [ ${net} == "mobilenetv2" ]; then
            ./run.sh mpredict ${net} data/${name}.jpg |grep "%" >>${LOG} 2>/dev/null
        else
            ./run.sh predict ${net} data/${name}.jpg |grep "%" >>${LOG} 2>/dev/null
        fi
    done
done