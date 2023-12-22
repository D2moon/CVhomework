

## 最终结果：

**bear:**

<div style="display: flex; justify-content: space-between;">
	<figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\bearNorm.png" alt="bearNorm" style="zoom: 50%; display: block;" />
		<figcaption>法线示意图</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\bearAlbedo.png" alt="bearAlbedo" style="zoom: 50%; display: block;" />
		<figcaption>表面反射率</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\bearRender.png" alt="bearRender" style="zoom: 50%; display: block;" />
		<figcaption>重渲染图片</figcaption>
	</figure>
</div>

**cat:**

<div style="display: flex; justify-content: space-between;">
	<figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\catNorm.png" alt="catNorm" style="zoom: 50%; display: block;" />
		<figcaption>法线</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\catAlbedo.png" alt="catAlbedo" style="zoom: 50%; display: block;" />
		<figcaption>表面反射率</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\catRender.png" alt="catRender" style="zoom: 50%; display: block;" />
		<figcaption>重渲染图片</figcaption>
	</figure>
</div>



**buddha:**

<div style="display: flex; justify-content: space-between;">
	<figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\buddhaNorm.png" alt="buddhaNorm" style="zoom: 50%; display: block;" />
		<figcaption>法线</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\buddhaAlbedo.png" alt="buddhaAlbedo" style="zoom: 50%; display: block;" />
		<figcaption>表面反射率</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\buddhaRender.png" alt="buddhaRender" style="zoom: 50%; display: block;" />
		<figcaption>重渲染图片</figcaption>
	</figure>
</div>

**pot:**

<div style="display: flex; justify-content: space-between;">
	<figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\potNorm.png" alt="potNorm" style="zoom: 50%; display: block;" />
		<figcaption>法线</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\potAlbedo.png" alt="potAlbedo" style="zoom: 50%; display: block;" />
		<figcaption>表面反射率</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\potRender.png" alt="potRender" style="zoom: 50%; display: block;" />
		<figcaption>重渲染图片</figcaption>
	</figure>
</div>


## 结论：

上述图片是每个像素选择亮度在中间的50张照片使用Photometric Stereo得到的结果，选取不同数量照片得到的法线结果分别如下：

<div style="display: flex; justify-content: space-between;">
	<figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\bearNorm3.png" alt="bearNorm3" style="zoom: 50%; display: block;" />
		<figcaption>3</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\bearNorm20.png" alt="bearNorm20" style="zoom: 50%; display: block;" />
		<figcaption>20</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\bearNorm.png" alt="bearNorm50" style="zoom: 50%; display: block;" />
		<figcaption>50</figcaption>
	</figure>
    <figure style="text-align:center;">
		<img src="E:\CVhomework\Homework1\build\bearNorm96.png" alt="bearNorm96" style="zoom: 50%; display: block;" />
		<figcaption>96</figcaption>
	</figure>
</div>

物体表面颜色的不同主要体现在反射率的不同上，在猫的例子上法线和重渲染图像几乎看不到身上的纹理，甚至原本的照片上也很难看清楚，但在表面反射率上就可以清晰的看到。