# 骨干替换范围核查

基于旧版 `tsjt_source_interaction_v2` 逐字节比对：

- `pgrad/source_interaction.py`：**一致**；SHA256 `d6d1da0a902fd8db7f0ca142cfa78714a6008eb59102c675732e4081380999ce`
- `pgrad/pgrad.py`：**一致**；SHA256 `bcf4f668d2e410d0a31644cdf567aa731bdc95bc7b6aca83a8e45a39f7ab856a`
- `pgrad/rotation.py`：**一致**；SHA256 `a9fde7bdcc94767435b7b0db1d7e6a4ff7408967e12719c1d35b37a6c8a302db`
- `pgrad/fusion.py`：**一致**；SHA256 `1ad1eee31548b8bd7b6f8c9c7b7cefa8240b193647f07aad739c68778716506d`
- `training/meta.py`：**一致**；SHA256 `164bbf1bb94725d18f833062f3622064c78d585ef0a0fa9fc7e41717748bc273`
- `training/trainer.py`：**一致**；SHA256 `34346b1d85c434eacc14af4ffc647096293fdd48a3e022812faa62aea270cc87`

替换的文件：`models/pgrad_model.py`。新增：`models/softs_backbone.py`。

训练日志/真实指标：**当前没有重新运行用户的四城训练数据，因此暂不报告模型 MAE/RMSE。**
已有本地合成测试涵盖真实 SOFTS + 交互 v2 + 原旋转 + 元更新 + 原融合。
