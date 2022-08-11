# DeepPi
End to end reconstruction of neutral pions produced in hadronic tau decays for analysis of the CP structure of the Higgs boson. 

Draft README document. 
- Production of RHTree ROOT tuples in `Production` (run with `cmsenv`)
- Tools to convert to dataframes/images in `Analysis` (run with `cmsenv` or `tau-ml`)
- NN Training and DataLoading in `Training` (run with `tau-ml`)
- NN Evaluation in `Evaluation` (run with `tau-ml`)

When using `cmsenv` for tuple production, it is recommended to only compile from the `Production` folder as python files in other areas may not be compatible with Python 2 and can cause compilation errors.

To install `tau-ml` conda environment run `source env.sh conda`.
