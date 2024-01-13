# SmCCNet

Sparse multiple canonical correlation network analysis (SmCCNet) is a machine learning technique for integrating omics data along with a variable of interest (e.g., phenotype of complex disease), and reconstructing multiomics networks that are specific to this variable. We present the second-generation SmCCNet (SmCCNet 2.0) that adeptly integrates single or multiple omics data types along with a quantitative or binary phenotype of interest. In addition, this new package offers a streamlined setup process that can be configured manually or automatically, ensuring a flexible and user-friendly experience.

To install and use the package, you may download the directory or follow the instructions below.
```{r, install-and-example}
# Install package
if (!require("devtools")) install.packages("devtools")
devtools::install_github("KechrisLab/SmCCNet")

# Load package
library(SmCCNet)
```

In the **vignettes** folder, users can find a documentation that illustrates how to implement SmCCNet with an example data set. The data file is included under the **data** folder. Details on all the functions included in the package are documented in the package manual under the **package** folder. Users may directly download the package tarball for all functions and example data, which is also under the **package** folder.

Please report any issues at https://github.com/KechrisLab/SmCCNet/issues.
