## Dataset

We use the **ATLASv2** dataset for our experiments.  
Please refer to [ATLASv2 on Bitbucket](https://bitbucket.org/sts-lab/atlasv2/src/master/) for detailed download and usage instructions.

In our setup, we only utilize logs collected from **Carbon Black Cloud**, which feature long event sequences and complex temporal dependencies. Although the dataset size is relatively small, it effectively captures realistic endpoint behaviors.


## ⚙️ Experimental Configuration
- **Training:** Conducted on data from host 1.  
- **Testing:** Performed on host 2 to avoid any data leakage.  
- **Data Augmentation:** We apply the over-sampling strategy provided in ATLASv2 to enhance log sequence diversity.  
- **Model Settings:** All experiments are run using our model’s default configuration.