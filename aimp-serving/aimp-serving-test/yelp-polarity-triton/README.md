# 介绍
This bert-base-uncased model was fine-tuned for sequence classification using TextAttack and the yelp_polarity dataset loaded using the nlp library. 

## 算法介绍

1. 算法名称：bert-base-uncased（利用bert-base-uncased进行文本分类）

2. 输入：英文句子文本

3. 输出：2维向量，代表是否是负向的文本。[prob_negative, prob_positive]

   ```
   inputs text:  ['Although this movie is fantastic,and my daughter likes it, I still hate it.']
   prediction probs:   [0.3615776598453522, -0.5517081022262573]
   prediction label:  negative
```
   
   

# 参考
https://huggingface.co/ydshieh/bert-base-uncased-yelp-polarity
