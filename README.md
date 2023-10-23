# pdf_chatbot_llama2_pinecone
### 1. 放置 PDF 於data 資料夾
```
mkdir data
cp Medical_Chatbot.pdf data/
```
### 2. 利用singularity啟動程式
```
ml libs/singularity/3.10.2
singularity exec -B /work /work/u00cjz00/nvidia/transformers-pytorch-gpu_latest.sif pip3 install -r requirements.txt
singularity exec -B /work /work/u00cjz00/nvidia/transformers-pytorch-gpu_latest.sif python3 python_script.py 
```

### 3. 問題範例
```
# 針灸
ware is Acupuncture?
Give me some BOOKS about Acupuncture.
# 過敏
What are Allergies?

# 穴位
請列出跟肺臟有關的穴位
```
