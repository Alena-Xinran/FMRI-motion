from nilearn import datasets

abide = datasets.fetch_abide_pcp(
    n_subjects=20,         
    pipeline="cpac",        
    derivatives=["func_preproc"],  
)

print("共下载功能影像个数:", len(abide.func))
print("第一个文件路径:", abide.func[0])
print("前 5 个被试的诊断标签 (DX_GROUP):", [p["DX_GROUP"] for p in abide.phenotypic[:5]])