## 一、摘要

FinAnnounceChat是基于InternLM2-7B-Chat模型进行微调，该模型使用金融财经领域的公告数据进行微调，旨在探索模型在解读公司公告内容方面的应用潜力，为信息检索、风险监控、市场分析等提供价值。
![Alt text](image.png)
## 二、使用方法

### 2.1快速开始
1、模型下载
模型已上传到https://openxlab.org.cn/models，下载方法详见：https://openxlab.org.cn/docs/models/%E4%B8%8A%E4%BC%A0%E6%A8%A1%E5%9E%8B.html
```python
git clone https://lituotuo:50cd91c574ba3730894f85a55aa90b942358c4d1@code.openxlab.org.cn/lituotuo/FinAnnounceChat.git
```
2、本地部署
```python
git clone https://github.com/Hilly0420/finance_assistant.git
python start.py #修改web_demo.py中模型的路径
```
### 2.2 模型微调
#### XTuner安装
1、conda虚拟环境
``` python
conda create --name xtuner-env python=3.10 -y
conda activate xtuner-env
```
2、集成 DeepSpeed 版本的 XTuner
``` python
pip install -U 'xtuner[deepspeed]>=0.1.13'
```
3、微调
1）准备配置文件
``` python
xtuner list-cfg -p internlm2
```
复制并修改内容
``` python
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}
vi ${SAVE_PATH}/${CONFIG_NAME}_copy.py
```
2）数据集
微调准备的数据集格式如下：
``` python
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
            "input": "xxx",
            "output": "xxx"
        },
        xxx
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
            "input": "xxx",
            "output": "xxx"
        },
        xxx
    ]
}]
``` 
3）开始微调
``` python
xtuner train ${CONFIG_NAME_OR_PATH}
```
参考：
``` python
# 单卡
xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
# 多卡
(DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
(SLURM) srun ${SRUN_ARGS} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2
```
3）转为HF模型
``` python
xtuner convert pth_to_hf ./internlm2_chat_7b_qlora_oasst1_e3_copy.py \
                         ./work_dirs/internlm2_chat_7b_qlora_oasst1_e3_copy/iter1800.pth \
                         ./hf
```
4）HF模型合并
``` python
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

# 原始模型参数存放的位置
export NAME_OR_PATH_TO_LLM=/root/model/FinAnnounceChat

# Hugging Face格式参数存放的位置
export NAME_OR_PATH_TO_ADAPTER=/root/math/config/hf

# 最终Merge后的参数存放的位置
mkdir /root/math/config/work_dirs/hf_merge
export SAVE_PATH=/root/finance/config/work_dirs/hf_merge

# 执行参数Merge
xtuner convert merge \
    $NAME_OR_PATH_TO_LLM \
    $NAME_OR_PATH_TO_ADAPTER \
    $SAVE_PATH \
    --max-shard-size 2GB
```
4、Demo

``` python
streamlit run web_demo.py --server.address=0.0.0.0 --server.port 9060
```
