# 纸片人模仿视频人脸
> 本项目基于openpose的脸部追踪，然后只提供二次元人物的图片，即可实现让图片的人物模仿视频里的人说话
> 
> 因为项目主要还是以娱乐为主，所以效果非常差，这后面其实还是有非常多的优化空间的，理论上可以做到非常还原，我懒得优化，感兴趣的可以自行优化算法
## 视频文章
[B站]()<br>
[YouTube]()<br>
[掘金]()
## 效果展示
![](./images/out.gif)

![](./images/out2.jpg)
## 项目运行
```shell
pip install -r requirements.txt
```
简单使用openpose来生成人脸信息
```shell
bin\OpenPoseDemo.exe --video test.mp4 --face --write_json _json --write_video _out.avi
```


## 相关问题
