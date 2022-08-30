目录结构展示
```text
在 **hardwares/rk** 目录下有两个子目录 *cpp* 和 *python* 分别存放了c++代码和python代码。

.
├── README.md  -> 目录结构文件
├── cpp  -> 存放c++文件夹
│   ├── README.md  -> c++ 代码部署文档
│   ├── detection  -> detection 相关模型
│   │   └── picodet  -> picodet 模型
│   ├── lib  -> 存放公共使用的第三方库
│   ├── ocr  -> ocr相关模型
│   │   └── PP_OCR_V2  -> PP_OCR_V2 模型(包含了三个)
│   └── segmentation  -> segmentation模型
│       ├── BisenetV2  -> BisenetV2模型
│       └── PP_HumanSeg  -> PP_HumanSeg模型
└── python  -> 存放python文件夹
    ├── README.md  -> python 代码部署文档
    ├── detection  -> detection 相关模型
    │   └── picodet  -> picodet 模型
    ├── ocr  -> ocr相关模型
    │   └── PP_OCR_V2 -> PP_OCR_V2 模型(包含了三个)
    ├── segmentation -> segmentation模型
    │   ├── BisenetV2  -> BisenetV2模型
    │   └── PP_HumanSeg  -> PP_HumanSeg模型
    └── utils -> 存放公共使用的第三方库
```