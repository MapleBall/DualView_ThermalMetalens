# DualView_ThermalMetalens

2025/04/11 : 訓練 epoch=100/200 batch_size=16 兩種
2025/04/12 : write predict model (done) 
2025/04/12 04:44 已經寫好 train/predict 但泛用性很差，只能辨識訓練集的資料，下一步可能往擴增資料集方向試試看
2025/04/13 更改預處理，新增資料擴增:旋轉平移，加入應用率(讓訓練集資料有一定比例被旋轉一部分不會)
           更改模型結構，新增 BatchNormalize / Dropout 層
           (小) 把預處理與訓練程式分開
2525/04/14 把影像切小一點(128*128)，盡量避免noise的干擾
Do list : check training model : 已用訓練集資料檢查，100%正確，代表模型確實有訓練到，但已經overfitting
try optimize training process (Add ReduceCallBack in train) (done)
update training dataset
優化創建csv功能


