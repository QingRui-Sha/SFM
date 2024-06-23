# **Detail-preserving image warping by enforcing smooth image sampling**



## Training

If you would like to train your own model, you will likely need to customize some of the data-loading code in `voxelmorph/datasets.py` for your own datasets and data formats. However, it is possible to run the example scripts out-of-the-box, assuming that you provide a list of filenames in the training dataset. It's also assumed that the shape of all training image data is consistent, but this, of course, can be handled in a customized generator if desired.

For a given image list file `./data/list.txt` and output directory `./data/exp2`, the following script will train an image-to-image registration network with an unsupervised loss. Model weights will be saved to a path specified by the `--model-dir` flag.

```
python ./train_sfim.py --img-list ./data/list.txt --model-dir ./data/exp2 --img-prefix ./data/demo/ --data-inshape 160 192 224
```

The `--img-prefix` and `--img-suffix` flags can be used to provide a consistent prefix or suffix to each path specified in the image list. The `--data-inshape` flags can be used to specify the shape of the image for components designed in the model.


## Registration

If you simply want to register two images, you can use the `register.py` script with the desired model file. For example, if we have a model `./data/exp2/0400.pt`, we can run the following command:

```
python ./register.py --data-inshape 160 192 224 --img-list ./data/list.txt --img-prefix ./data/demo/ --model-dir ./data/exp2 --atlas-path ./data/demo/1.nii.gz --load-model ./data/exp2/0400.pt
```




## Rationality Inspection

To assess potential areas of errors after performing registration operations, we provide a rationality registration inspection tool that can be used offline by non-computer professionals, doctors, and experts. Run the following script:

```
Rationality Inspection.ipynb
```

In addition, the Sampling Frequency Map (SFM) has the potential to assist with segmentation or classification tasks as a pseudo-attention map. You can also use the above script to obtain it.


## Citation:
If you find this code is useful in your research, please consider to cite:
    

	@article{sha2024detail,
    title={Detail-preserving image warping by enforcing smooth image sampling},
    author={Sha, Qingrui and Sun, Kaicong and Jiang, Caiwen and Xu, Mingze and Xue, Zhong and Cao, Xiaohuan and Shen, Dinggang},
    journal={Neural Networks},
    pages={106426},
    year={2024},
    publisher={Elsevier}
}
## Reference:
<a href="https://github.com/voxelmorph/voxelmorph">VoxelMorph</a>