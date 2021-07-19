cat ./prune_res/alexnet_0717.log |grep zeepres |cut -d',' -f2- |xargs -I {} echo "AlexNet, "{} >>analyse.csv

cat ./prune_res/mobilenetv2_0717.log |grep zeepres |cut -d',' -f2- |xargs -I {} echo "MobileNetv2, "{} >>analyse.csv

cat ./prune_res/densenet201_0717.log |grep zeepres |cut -d',' -f2- |xargs -I {} echo "DenseNet201, "{} >>analyse.csv

cat ./prune_res/resnet50_0717.log |grep zeepres |cut -d',' -f2- |xargs -I {} echo "ResNet50, "{} >>analyse.csv