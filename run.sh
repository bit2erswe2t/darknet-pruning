#!/bin/bash

function="classifier"
if (echo $* | grep -E 'yolo' -q); then function="detector" ; fi
if (echo $* | grep -E 'quantize' -q); then function="quantize" ; fi

usage="predict"
if [[ $function == "detector" ]]; then usage="test"; fi
if [[ $function == "quantize" ]]; then usage=""; fi
if (echo $* | grep -E 'predict|valid|test' -q); then usage=`echo $* | sed 's/ /\n/g' | grep -E 'predict|valid|test'`; fi

net="caffe_densenet"
if (echo $* | grep -E 'net|tiny|vgg|yolo|dense' -q); then net=`echo $* | sed 's/ /\n/g' | grep -E "net|tiny|vgg|yolo|dense"` ; fi
netcfg="cfg/$net.cfg"
netweights="weights/$net.weights"

dataset="data/imagenet.data"
if (echo $net | grep -E 'yolo' -q); then dataset="cfg/coco.data" ; fi
if (echo $net | grep -E 'mobile|squeeze|caffe_densenet|caffe' -q )
then
    dataset="data/synset.imagenet.data"
fi
if (echo $net | grep -E 'caffe_googlenet' -q )
then
    dataset="data/imagenet.data"
fi


image="data/eagle.jpg"
if (echo $* | grep -E 'jpg|JPEG' -q); then image=`echo $* | sed 's/ /\n/g' | grep -E 'jpg|JPEG'`; fi


if (echo $* | grep nogpu -q); then nogpu="-nogpu"; else nogpu=""; fi
if (echo $* | grep noquant -q); then noquant="-noquant"; else noquant=""; fi
if (echo $* | grep nodla -q); then nodla="-nodla"; else nodla=""; fi
if (echo $* | grep gdb -q); then gdb="gdb --args "; else gdb=""; fi
echo "----------------------------------- executing -----------------------------------"
echo "$gdb./darknet $function $usage $dataset $netcfg $netweights $image $nogpu $nodla $noquant"
echo "---------------------------------------------------------------------------------"

$gdb./darknet $function $usage $dataset $netcfg $netweights $image $nogpu $nodla $noquant
