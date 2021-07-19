Dynamic Runtime Feature Map Pruning

'''sh
  update data/imagenet.data

  make -4j PRUNE=1 SAT_FEATURE=1

  bash prune_test.sh

  bash prune_save.sh
'''
