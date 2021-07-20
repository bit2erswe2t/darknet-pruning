## Dynamic Runtime Feature Map Pruning

```sh
# update data/imagenet.data and data/synset_imagenet.data (ensure the valid dir is your data list)

# update weights/network.weights (you need to download weights from Google)

make -4j PRUNE=1 SAT_FEATURE=1

bash prune_test.sh

bash prune_save.sh
```

