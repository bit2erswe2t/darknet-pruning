## Dynamic Runtime Feature Map Pruning

```sh
# update data/imagenet.data
# update weights/network.weights (download weights from Google)

make -4j PRUNE=1 SAT_FEATURE=1

bash prune_test.sh

bash prune_save.sh
```

