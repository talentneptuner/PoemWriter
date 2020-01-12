## 基于attention的自动对诗

### 数据来源
数据量总共73130条唐诗数据，数据来源自
[chinese-poetry: 最全中文诗歌古典文集数据库](https://github.com/chinese-poetry/chinese-poetry)


### 程序运行方式
运行main.py     
超参数在main.py指定    

### 训练示例
Epoch : 1, batch : 0, loss : 1.09479, sample : 窗含西岭千秋雪, result : 郑<eos><eos><eos>卢<eos><eos>嗣                  
Epoch : 1, batch : 100, loss : 0.80019, sample : 窗含西岭千秋雪, result : 不不不不人人<eos><eos>                   
Epoch : 1, batch : 200, loss : 0.78623, sample : 窗含西岭千秋雪, result : 一一不不不不人<eos>                   
Epoch : 1, batch : 300, loss : 0.76617, sample : 窗含西岭千秋雪, result : 不人人不不不人<eos>                     
Epoch : 1, batch : 400, loss : 0.74086, sample : 窗含西岭千秋雪, result : 一人无是一人<eos><eos>                   
Epoch : 1, batch : 500, loss : 0.73344, sample : 窗含西岭千秋雪, result : 不知何处不知人<eos>               
Epoch : 1, batch : 600, loss : 0.71875, sample : 窗含西岭千秋雪, result : 一里无人不得知<eos>                
Epoch : 1, batch : 700, loss : 0.70334, sample : 窗含西岭千秋雪, result : 一声声雨满云台<eos>                  
Epoch : 1, batch : 800, loss : 0.67214, sample : 窗含西岭千秋雪, result : 一声风雨雨声开<eos>                 
Epoch : 1, batch : 900, loss : 0.66058, sample : 窗含西岭千秋雪, result : 月下风流月下风<eos>                        
Epoch : 1, batch : 1000, loss : 0.64403, sample : 窗含西岭千秋雪, result : 玉辇初开一夜风<eos>                      
Epoch : 1, batch : 1100, loss : 0.66809, sample : 窗含西岭千秋雪, result : 水上连天一夜寒<eos>       
Epoch : 1, loss : 0.72170, sample : 窗含西岭千秋雪, result : 水上山中有一声<eos>                               
Epoch : 2, batch : 0, loss : 0.65959, sample : 窗含西岭千秋雪, result : 水上山中有一声<eos>              
Epoch : 2, batch : 100, loss : 0.66355, sample : 窗含西岭千秋雪, result : 红叶红枝满眼前<eos>              
Epoch : 2, batch : 200, loss : 0.66177, sample : 窗含西岭千秋雪, result : 水水云山水汽寒<eos>             
Epoch : 2, batch : 300, loss : 0.62909, sample : 窗含西岭千秋雪, result : 水上山河水上天<eos>                      
Epoch : 2, batch : 400, loss : 0.61864, sample : 窗含西岭千秋雪, result : 水上青山白日斜<eos>                             
Epoch : 2, batch : 500, loss : 0.63404, sample : 窗含西岭千秋雪, result : 白日斜阳一夜歌<eos>                       
Epoch : 2, batch : 600, loss : 0.59852, sample : 窗含西岭千秋雪, result : 月照寒云带雨中<eos>                                            
Epoch : 2, batch : 700, loss : 0.63141, sample : 窗含西岭千秋雪, result : 月下寒云夜夜深<eos>             
Epoch : 2, batch : 800, loss : 0.59108, sample : 窗含西岭千秋雪, result : 水上山花一夜风<eos>                 
Epoch : 2, batch : 900, loss : 0.57881, sample : 窗含西岭千秋雪, result : 水阔山深水自流<eos>                 
Epoch : 2, batch : 1000, loss : 0.59872, sample : 窗含西岭千秋雪, result : 月照江南一夜风<eos>                
Epoch : 2, batch : 1100, loss : 0.57872, sample : 窗含西岭千秋雪, result : 月照松窗一夜风<eos>           
Epoch : 2, loss : 0.61708, sample : 窗含西岭千秋雪, result : 雨湿云山一片云<eos>           
Epoch : 3, batch : 0, loss : 0.57037, sample : 窗含西岭千秋雪, result : 雨湿云山一片云<eos>       
Epoch : 3, batch : 100, loss : 0.56317, sample : 窗含西岭千秋雪, result : 水上楼台月下楼<eos>                  
Epoch : 3, batch : 200, loss : 0.56406, sample : 窗含西岭千秋雪, result : 月照湖山万里空<eos>     
Epoch : 3, batch : 300, loss : 0.57911, sample : 窗含西岭千秋雪, result : 月照湖山万里秋<eos>               
Epoch : 3, batch : 400, loss : 0.56596, sample : 窗含西岭千秋雪, result : 月照西江一树蝉<eos>                          
Epoch : 3, batch : 500, loss : 0.55013, sample : 窗含西岭千秋雪, result : 雨湿寒江万里春<eos>             
Epoch : 3, batch : 600, loss : 0.55550, sample : 窗含西岭千秋雪, result : 风过三湘一片帆<eos>               
Epoch : 3, batch : 700, loss : 0.55023, sample : 窗含西岭千秋雪, result : 月照江南万里春<eos>                
Epoch : 3, batch : 800, loss : 0.54731, sample : 窗含西岭千秋雪, result : 月照江南万里愁<eos>          
Epoch : 3, batch : 900, loss : 0.52199, sample : 窗含西岭千秋雪, result : 树接岩花一两枝<eos>        
Epoch : 3, batch : 1000, loss : 0.53518, sample : 窗含西岭千秋雪, result : 树下寒江万里流<eos>     
Epoch : 3, batch : 1100, loss : 0.51949, sample : 窗含西岭千秋雪, result : 水浸山根一片冰<eos>    
Epoch : 3, loss : 0.55605, sample : 窗含西岭千秋雪, result : 树绕渔舟万里愁<eos>    
Epoch : 4, batch : 0, loss : 0.51344, sample : 窗含西岭千秋雪, result : 树绕渔舟万里愁<eos>   
Epoch : 4, batch : 100, loss : 0.49657, sample : 窗含西岭千秋雪, result : 树映寒鸦一片蝉<eos>   
Epoch : 4, batch : 200, loss : 0.52207, sample : 窗含西岭千秋雪, result : 树映寒潭一片云<eos>    
Epoch : 4, batch : 300, loss : 0.50866, sample : 窗含西岭千秋雪, result : 树接东林一夜风<eos>    

