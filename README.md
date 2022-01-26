# model_interpretability_captum
- Using Captum library and interpreting the models to understand their decisions better. 
- the folowing results are on CIFAR10 and most of the code is from the captum.ai site.
- Captum can be easily used to integrate with python but it requires high amount of RAM.
- I tried using it on a resnet18 with (224 & 128) resolution and my 16GB machine was not enough. 
- Resolution of (60 * 60) does work though. 
- You can try the code (Main-cats&dogs.ipynb) if you have enough RAM.
- I will be uloading results once I have access to a capable Machine.
- I've tested with PyTorch 1.5 but I couldn't run it on PyTorch 1.1.0(It doesn't support)

## Installaion
- `conda install pytorch torchvision cpuonly -c pytorch`
- `conda install -c captum pytorch`

## Results
![result](https://github.com/Santosh7vasa/Model_Interpretability_Captum/blob/master/results/captum1.jpg?raw=true)
![result](https://github.com/Santosh7vasa/Model_Interpretability_Captum/blob/master/results/captum2.jpg?raw=true)
![result](https://github.com/Santosh7vasa/Model_Interpretability_Captum/blob/master/results/captum3.jpg?raw=true)
![result](https://github.com/Santosh7vasa/Model_Interpretability_Captum/blob/master/results/captum4.jpg?raw=true)
