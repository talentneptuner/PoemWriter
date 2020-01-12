## 基于attention的自动对诗

### 数据来源
数据量总共73130条唐诗数据，数据来源自
[chinese-poetry: 最全中文诗歌古典文集数据库](https://github.com/chinese-poetry/chinese-poetry)


### 程序运行方式
运行main.py     
超参数在main.py指定    

### 训练示例
Epoch : 1, batch : 0, loss : 1.09474, sample : 人世几回伤往事, result : 风一<eos><eos><eos><eos><eos><eos>     
Epoch : 1, batch : 100, loss : 0.77611, sample : 人世几回伤往事, result : 不不不不不人<eos><eos>      
Epoch : 1, batch : 200, loss : 0.79168, sample : 人世几回伤往事, result : 一不不人人<eos><eos><eos>     
Epoch : 1, batch : 300, loss : 0.75662, sample : 人世几回伤往事, result : 不人不人不人<eos><eos>     
Epoch : 1, batch : 400, loss : 0.74512, sample : 人世几回伤往事, result : 一人无是不人<eos><eos>     
Epoch : 1, batch : 500, loss : 0.70768, sample : 人世几回伤往事, result : 不知不是不知人<eos>     
Epoch : 1, batch : 600, loss : 0.71916, sample : 人世几回伤往事, result : 一里春风落月来<eos>    
Epoch : 1, batch : 700, loss : 0.72391, sample : 人世几回伤往事, result : 一枝花下一枝秋<eos>     
Epoch : 1, batch : 800, loss : 0.70459, sample : 人世几回伤往事, result : 不知何事是君家<eos>     
Epoch : 1, batch : 900, loss : 0.66287, sample : 人世几回伤往事, result : 不知人事不知年<eos>     
Epoch : 1, batch : 1000, loss : 0.66619, sample : 人世几回伤往事, result : 一夜风流一夜风<eos>      
Epoch : 1, batch : 1100, loss : 0.67818, sample : 人世几回伤往事, result : 不知何事不知君<eos>      
Epoch : 1, loss : 0.72902, sample : 人世几回伤往事, result : 不知何处是何人<eos>     
Epoch : 2, batch : 0, loss : 0.67126, sample : 人世几回伤往事, result : 不知何处是何人<eos>      
Epoch : 2, batch : 100, loss : 0.61783, sample : 人世几回伤往事, result : 不知何事更无情<eos>     
Epoch : 2, batch : 200, loss : 0.65150, sample : 人世几回伤往事, result : 一身相见一枝枝<eos>      
Epoch : 2, batch : 300, loss : 0.62480, sample : 人世几回伤往事, result : 不知何处不知人<eos>      
Epoch : 2, batch : 400, loss : 0.64624, sample : 人世几回伤往事, result : 不知何处不相逢<eos>     
Epoch : 2, batch : 500, loss : 0.59355, sample : 人世几回伤往事, result : 不知何事不如来<eos>     
Epoch : 2, batch : 600, loss : 0.62644, sample : 人世几回伤往事, result : 一时无事不知年<eos>      
Epoch : 2, batch : 700, loss : 0.63523, sample : 人世几回伤往事, result : 一生何事更何如<eos>     
Epoch : 2, batch : 800, loss : 0.61616, sample : 人世几回伤往事, result : 一时相伴两三年<eos>     
Epoch : 2, batch : 900, loss : 0.58849, sample : 人世几回伤往事, result : 不知何处是何人<eos>     
Epoch : 2, batch : 1000, loss : 0.60686, sample : 人世几回伤往事, result : 不知何处更相关<eos>     
Epoch : 2, batch : 1100, loss : 0.59648, sample : 人世几回伤往事, result : 不知何事不堪伤<eos>     
Epoch : 2, loss : 0.62207, sample : 人世几回伤往事, result : 不知何事不堪伤<eos>      
Epoch : 3, batch : 0, loss : 0.58105, sample : 人世几回伤往事, result : 不知何事不堪伤<eos>      
Epoch : 3, batch : 100, loss : 0.56460, sample : 人世几回伤往事, result : 不知何处更相思<eos>      
Epoch : 3, batch : 200, loss : 0.58459, sample : 人世几回伤往事, result : 一身犹是旧山人<eos>     
Epoch : 3, batch : 300, loss : 0.56172, sample : 人世几回伤往事, result : 不知谁是此中人<eos>    
Epoch : 3, batch : 400, loss : 0.55547, sample : 人世几回伤往事, result : 故人何必更相违<eos>   
Epoch : 3, batch : 500, loss : 0.55261, sample : 人世几回伤往事, result : 人间无事可堪悲<eos>   
Epoch : 3, batch : 600, loss : 0.56802, sample : 人世几回伤往事, result : 故乡何处是今年<eos>    
Epoch : 3, batch : 700, loss : 0.56088, sample : 人世几回伤往事, result : 山中无事亦无穷<eos>    
*epoch 9 batch 100 出现了 result:山形依旧枕寒流*
