更新到th1.9_cuda11.1后，没有那些cuda error了


Dataset incident

测试数据选 icdar15_incident

F:\scripts\blurOCR\icdar15_incident

训练数据选 

syth90k

C:\Users\Administrator\Downloads\mjsynth.tar\mjsynth\mnt\ramdisk\max\90kDICT32px\1\1  

api:

https://github.com/andreasveit/coco-text



== experiment 1:

icdar15 --> train

icdar15_incident --> test


== experiment 2:

super vision preprocessing

icdar15_incident --> test


== experiment 3:

super vision preprocessing train with crnn

icdar15_incident --> test


== experiment 4:

validate in Chinese Dataset

 
# ------------
task0  

--blur用最简单的resize
--sr和crrn分开训练
--cv只用crrn，不用sr

# ------------

task1

--添加feature loss

img --> crnn --> out / feature

blur img --> sr net --> out_img --> crnn --> out_blur / feature_blur

feature + feature_blur --> loss feature  ======== DRRN + CRNN

out_img + blur_img --> loss sr  ================= DRRN

out + target --> loss cls  ====================== CRNN

# -------------- 思路

因分为两部分： 

1. drrn是否能在crnn训练中起到正向作用，如使用drrn来finetune 训好的crnn网络，让crnn具备模糊图片识别能力，最好有与普通的模糊增强的效果对比； —— 还没想好
2. crrn利用feature loss辅助drrn的训练,在validation中使用drrn + crnn的方式推理， 然后添加原告内容，作为汉字方面的应用推广实例。