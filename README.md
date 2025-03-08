# EVIT-UNET

## 原理

### efficinet

确定了每个维度最合适的调整系数，然后将它们一同应用到基线网络中，对每个维度都进行适当的缩放，并且确保其符合目标模型的大小和计算预算。分别找到宽度、深度和分辨率的最佳系数，然后将它们组合起来一起放入原本的网络模型中，对每一个维度都有所调整。从整体的角度缩放模型。与传统方法相比，这种复合缩放法可以持续提高模型的准确性和效率。

EfﬁcientNet网络提出是有以下设想：设计一个标准化的卷积网络扩展方法，既可以实现较高的准确率 ，又可以充分的节省算力资源。即如何平衡分辨率 、深度和宽度这三个维度，来实现网络在效率和准确率上的优化。

平衡网络**宽度/深度/分辨率**的所有维度是至关重要的，令人惊讶的是，这种平衡可以通过简单地以恒定的比例缩放每个维度来实现。**第一个对网络宽度、深度和分辨率这三个维度之间的关系进行实证量化**

**我们限制所有层必须以恒定比例均匀缩放**

使用了移动倒置瓶颈卷积（MBConv），类似于 MobileNetV2 

![img](https://i-blog.csdnimg.cn/blog_migrate/5139903c3c4646a947b03e42a0929a21.png)



![img](https://i-blog.csdnimg.cn/blog_migrate/1b861d609b2df2fb78abd6f745875d5c.png#pic_center)

2~8个stage：这几个stage在重复堆叠`MBConv`结构（最后一列的`Layers`表示该Stage重复MBConv结构多少次）

每个MBConv后会跟一个数字1或6，这里的1或6就是倍率因子n，即MBConv中第一个1x1的卷积层会将输入特征矩阵的channels扩充为n倍，其中k3x3或k5x5表示MBConv中Depthwise Conv(深度可分离卷积)所采用的卷积核大小。Channels表示通过该Stage后输出特征矩阵的Channels。Resolution表示输入的特征图尺寸(使用长、宽表示)



![img](https://i-blog.csdnimg.cn/blog_migrate/5cdf5d2a52a789b193251fbfcc4813af.png#pic_center)

MBConv结构主要由一个1x1的普通卷积（升维作用，包含BN和Swish），一个k ∗ k k*kk∗k的Depthwise Conv卷积（包含BN和Swish深度可分离卷积），k的具体值可看EfficientNet-B0的网络框架主要有3x3和5x5两种情况，一个SE模块，一个1x1的普通卷积（降维作用，包含BN），一个Droupout层构成。

- 第一个升维的1x1卷积层，它的卷积核个数是输入特征矩阵channel的n倍，我们看见网络中的`MBConv1`和`MBConv6`后面跟着的1、6就是倍率因子。
- **当n=1时，不要第一个升维的1x1卷积层**，即Stage2中的MBConv结构一样，都没有第一个升维的1x1卷积层(这和MobileNetV3网络类似)，在源码中我们就舍弃MBConv中的第一层卷积构建。
- 经过1x1卷积**升维**之后的特征，再经过深度可分离层之后，特征尺寸需要保持不变(只有保证分组数和输入通道数保持一致才能确保输入和输入的channel保持不变)。
- shortcut连接(类似于残差连接)，只有当输入MBConv结构的特征矩阵与输出的特征矩阵shape相同时才能进行。
- **MBconv中的Droupout和全连接中的Droupout是不相同的**，在源码中只有使用到shortcut连接的MBConv模块才有Dropout层。



![img](https://i-blog.csdnimg.cn/blog_migrate/82b0d697e82626a562fe4cc217d3d846.png#pic_center)

经过Depthwise Conv层之后的特征图在传入SE模块之后分成两条分支，上面一条将当前的feature map保留下来，第二天分支首先将特征图进行全局平均池化，然后经过第一个全连接层FC1进行降维操作，再经过第二全连接层FC2进行升维操作，最后将两条分支的特征矩阵进行相乘。

- 第一个全连接层的节点个数是输入该MBConv特征矩阵channels的1/4，且使用的是swish激活函数；
- 第二个全连接层的节点个数等于经过Depthwise Conv层之后的特征图channels，且使用的是sigmoid激活函数。



![QQ_1733820534779](../../../Typora/image/QQ_1733820534779.png)

![QQ_1733820548530](../../../Typora/image/QQ_1733820548530.png)

![img](https://i-blog.csdnimg.cn/blog_migrate/1776c6b2c10c4935aecf939fee881e38.png#pic_center)

![QQ_1733820577693](../../../Typora/image/QQ_1733820577693.png)

![img](https://i-blog.csdnimg.cn/blog_migrate/f6fc1bb30c976eaf6666f1ff5c71d36e.png#pic_center)

#### 逐深度卷积（Depthwise Convolution）

逐深度卷积就是深度(channel)维度不变，改变特征图尺寸H/W；

深度卷积的卷积核为单通道模式，需要对输入的每一个通道进行卷积，这样就会得到和输入特征图通道数一致的输出特征图。即有输入特征图通道数=卷积核个数=输出特征图个数。

假设，一个大小为N×N像素、3通道彩色图片，3个单通道卷积核分别进行卷积计算，输出3个单通道的特征图。所以，一个3通道的图像经过运算后生成了3个Feature map

#### 逐点卷积（Pointwise Convolution）

逐点卷积就是W/H维度不变(不改变特征图的尺寸)，改变特征图的通道数channel。

根据深度卷积可知，输入特征图通道数=卷积核个数=输出特征图个数，这样会导致输出的特征图个数过少（或者说输出特征图的通道数过少，可看成是输出特征图个数为1，通道数为3），从而可能影响信息的有效性。此时，就需要进行逐点卷积。

逐点卷积（Pointwise Convolution，PWConv）实质上是用1x1的卷积核进行升维。例如在GoogleNet的三个版本中都大量使用1 ∗ 1 1*11∗1的卷积核，那里主要是用来降维。1 ∗ 1 1*11∗1的卷积核主要作用是对特征图进行升维和降维，但是不改变特征图的尺寸大小。

#### 深度可分离卷积（Depthwise Separable Convolution）

由深度卷积（depthwise convolution）加逐点卷积（pointwise convolution）组成。

深度卷积用于提取空间特征，逐点卷积用于提取通道特征。深度可分离卷积在特征维度上**分组卷积**，对每个channel进行独立的**逐深度卷积**，并在输出前使用一个1x1卷积（**逐点卷积**）将所有通道(channels)进行**聚合**

![img](https://i-blog.csdnimg.cn/blog_migrate/73e4f49443fc86d485380a234b2dfc0e.png#pic_center)

#### V2小改

![img](https://i-blog.csdnimg.cn/blog_migrate/77229e872a26e1576f632522ffde636f.png#pic_center)

- 除了使用到`MBConv`模块外，还使用了`Fused-MBConv`模块（主要是在网络浅层中使用）。
- 使用较小的`expansion ratio`（`MBConv`中第一个`expand conv1x1`或者`Fused-MBConv`中第一个`expand conv3x3`）比如`4`，在EfficientNetV1中基本都是`6`. 这样的好处是能够减少内存访问开销。
- 偏向使用更小(`3x3`)的kernel_size
- 移除了EfficientNetV1中最后一个步距为1的stage（就是EfficientNetV1中的stage8)
- ![img](https://i-blog.csdnimg.cn/blog_migrate/d96d3e601296a960f8db8d6907a69c0b.png#pic_center)

EfficientNetV2-S分为Stage0到Stage7（EfficientNetV1中是Stage1到Stage9）

**在源码中Stage6的输出Channels是等于256并不是表格中的272，Stage7的输出Channels是1280并不是表格中的1792**



`Fused-MBConv`模块

注意当expansion ratio等于1时是没有expand conv的，还有这里是没有使用到SE结构的（原论文图中有SE）。注意当stride=1且输入输出Channels相等时才有shortcut连接。还需要注意的是，当有shortcut连接时才有Dropout层，而且这里的Dropout层是Stochastic Depth，即会随机丢掉整个block的主分支（只剩捷径分支，相当于直接跳过了这个block）也可以理解为减少了网络的深度。

![img](https://i-blog.csdnimg.cn/blog_migrate/62922ddc1f727308c0b71163ebf6aa5c.jpeg#pic_center)







### EVIT-UNET

使用深度卷积（DW）卷积[13]构建 FFN （FFN）以提取局部特征,与标准卷积相比，DW卷积对每个输入通道应用一个滤波器，显著降低了计算复杂性并增强了局部特征。

![QQ_1733833245357](../../../Typora/image/QQ_1733833245357.png)

## 代码

**编码器（Encoder）**

- **初始层（Stem）**：这是网络的初始层，负责从输入图像中提取基本特征。
- **局部层（Local Layers）**：这些层使用卷积运算处理局部信息，通过下采样（如最大池化）减少空间维度并提取更抽象的特征。
- **全局层（Global Layers）**：这些层捕获图像的全局上下文。这些可能涉及全局平均池化或类似的方法。
- **跳跃连接（Skip Connections）**：这些连接跳过几层，允许来自早期阶段（具有更丰富的空间细节）的信息直接传递到后期阶段（具有更抽象的特征），改善梯度流并防止梯度消失。

**解码器（Decoder）**

- **上采样（Upsampling）**：该过程增加特征的空间分辨率，朝着原始图像大小的方向发展。它帮助从编码器中提取的更抽象的特征中重建详细信息。
- **局部层（上采样）**：这些层使用转置卷积或上采样运算增加特征的空间分辨率。
- **全局层（上采样）**：这些层将上采样的特征与编码器中学习的全局上下文信息相结合。
- **跳跃连接（解码器）**：这些连接合并上采样的特征与来自编码器的相!应跳跃连接，融合来自局部和全局视角的信息。

**注意力模块（Attention Module）**

- **下采样（Downsample）**：该层在应用注意力机制之前减少特征的空间维度。
- **注意力（Attention）**：该模块允许网络关注图像的相关部分，通过计算注意力权重来实现。这通常涉及计算查询、键和值矩阵，并执行缩放点积注意力操作。
- **上采样（Upsample）**：该层增加注意力权重特征的空间分辨率，使其与输出特征图对齐。

**输出**

- **Conv1x1-BN**：该层应用1x1卷积和批量归一化将特征投影到期望的输出类别数量。



1. `forward_features(x)` 方法使用的类和方法：

类：

1. stem: 初始卷积层，将输入图像转换为初始特征表示
2. Embedding: 特征嵌入，将输入特征映射到指定维度，支持多种下采样策略
3. eformer_block: 构建下采样网络块，包含多个FFN或AttnFFN层，控制网络深度和特征提取
4. FFN: 前馈神经网络，通过MLP对特征进行非线性变换和特征重新表达
5. AttnFFN: 结合注意力机制和前馈神经网络，增强特征表示能力
6. Attention4D: 四维注意力机制，通过计算查询、键、值增强特征表示
7. Attention4DDownsample: 4D注意力下采样模块，在下采样过程中应用注意力机制
8. LGQuery: 局部全局查询模块，结合局部和全局特征信息

方法：

- forward_tokens()
- forward()

1. `forward_up_features(x, x_downsample)` 方法使用的类和方法：

类：

1. Expanding: 上采样模块，将特征图恢复到更高分辨率，支持多种上采样策略
2. eformer_block_up: 构建上采样网络块，包含多个FFN或AttnFFN层，控制网络上采样和特征重建过程

3. FFN: 前馈神经网络，通过MLP对特征进行非线性变换和特征重新表达

4. AttnFFN: 结合注意力机制和前馈神经网络，增强特征表示能力

5. Attention4DUpsample: 4D注意力上采样模块，在上采样过程中应用注意力机制

6. CCA: 跨通道注意力模块，通过通道间的交互增强特征表示
7. nn.Conv2d

方法：

- forward()
- concat()

`AttnFFN` 块包含了注意力机制，而 `FFN` 块不包含注意力机制。



- **Local（局部特征）**：

  - `LGQuery` 类：通过 `local` 卷积层实现局部特征提取，在注意力模块中对q使用

  ```python
  self.local = nn.Sequential(
      nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim),
  )
  ```

- **Downsampling（下采样）**：

  - `Embedding` 类：实现下采样功能

  ```python
  self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                        stride=stride, padding=padding)
  ```

  - `Attention4DDownsample` 类：带注意力机制的下采样

- **Global + Local（全局 + 局部）**：

  - `LGQuery` 类：通过 `local_q + pool_q` 结合局部和全局特征

  ```python
  def forward(self, x):
      local_q = self.local(x)  # 局部查询
      pool_q = self.pool(x)    # 池化查询
      q = local_q + pool_q     # 合并查询
  ```

- **Skip Connection（跳跃连接）**：

  - `concat_back_dim` 模块列表：实现特征图的跳跃连接

  ```python
  skip_down_conv = nn.Conv2d(in_channels=2 * embed_dim_reverse[i], 
                              out_channels=embed_dim_reverse[i], 
                              kernel_size=3, padding=1)
  ```

- **Bottleneck（瓶颈层）**：

  - `eformer_block` 函数：通过 `AttnFFN` 和 `FFN` 实现瓶颈层变换

  ```python
  blocks.append(AttnFFN(
      dim, mlp_ratio=mlp_ratio,
      act_layer=act_layer, norm_layer=norm_layer,
      drop=drop_rate, drop_path=block_dpr,
      use_layer_scale=use_layer_scale,
      layer_scale_init_value=layer_scale_init_value,
      resolution=resolution,
      stride=stride,
  ))
  ```

每一层都是下采样，真正实现全局与局部是靠q来实现的，Downsampling与下采样中都可能会有注意力机制，只是FFN在三层之后，下采样是在两层过后，他会改变sbsd得值使其让其判断为使用增强的下采样即含全局与局部的q

进行解码时，先会判断当前层会不会进行拼接，如果是不是指定层则只进行上采样，如果是指定层且是多分类则会进行通道注意力CCA，每一层进行一次，相当于是预处理，后进行cat拼接，由于是对于初始层的倍数通道，所以会使用skip_down_conv进行通道转换，相当于一步格式处理









## **PS**

1. 卷积核大小 → 感受野
2. 步长 → 下采样
3. 填充 → 保持空间信息



points = list(itertools.product(range(3), range(3)))会生成所有可能的 `(x, y)`

- `range(3)` 生成的序列为 `[0, 1, 2]`。

- 所以，生成的 `(x, y)` 组合为：

- 结果为：

  ```python
  points = [(0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2)]
  ```

“嵌入”（Embedding）是机器学习和深度学习中的一个重要概念，主要用于将高维数据映射到低维空间，以便于进行计算和分析。嵌入的主要目的是捕捉数据的语义特征，同时减少数据的维度，使得后续的学习任务（如分类、聚类等）更加高效和准确。

- 低维空间

  ：通常指的是维度较少的空间，常见的低维空间包括一维（线）、二维（平面）和三维（立体）。例如：

  - 一维：只有一个特征（如温度）。
  - 二维：有两个特征（如身高和体重）。
  - 三维：有三个特征（如长、宽、高）。

在数据分析中，低维通常指特征数量相对较少的情况。

- **高维空间**：高维空间通常指的是维度数量较多的空间，特征数量可能达到数十、数百甚至数千个。例如，在图像处理任务中，一张 28x28 像素的灰度图像可以被视为一个 784 维的特征向量（每个像素对应一个特征）。



to_2tuple

无论输入是单个值（如 16）还是已经是二元组（如 (16, 16)），都能得到一致的格式。



下采样（Downsampling）**定义**：下采样是指通过**减少**数据的分辨率或样本数量来降低数据的维度。这通常涉及到从原始数据中选择部分数据点或通过某种方式合并数据点。下采样可以通过减少图像的像素数量来降低图像的分辨率。例如，将 256x256 像素的图像下采样到 128x128 像素。

上采样（Upsampling）**定义**：上采样是指通过**增加**数据的分辨率或样本数量来提高数据的维度。这通常涉及到插值或其他方法来生成新的数据点。



双MLP通道注意力

1. **输入**：输入特征图（Tensor）具有形状（B, C, H, W），其中B是批量大小，C是通道数，H是高度，W是宽度。
2. **第一个MLP**：第一个MLP接受输入特征图的平均值（或最大值）作为输入，输出一个通道注意力权重向量（Tensor）具有形状（B, C）。
3. **第二个MLP**：第二个MLP接受第一个MLP的输出作为输入，输出一个通道注意力权重向量（Tensor）具有形状（B, C）。
4. **通道注意力权重**：第二个MLP的输出是最终的通道注意力权重向量，用于计算通道注意力。



`CCA` 是一个类，代表了跨通道注意力，帮助模型关注到输入特征中不同通道之间的关系。

使用这个通道注意力模块来选择目标特征的重要通道，从而提高目标检测准确率。

可以根据输入特征的通道维度来调整权重，从而实现特征选择和加权的功能。



MLP是最基本的前馈网络（所以FFN里面会包含他，相当于一个更强大的前馈网络），MLP 的核心是通过深度学习从大量数据中学习特征和模式，并训练参数。通过参数与激活函数的组合，MLP 能够拟合特征与目标之间的真实函数关系。

简单来说，MLP 能够将信息逐层重新组合，每层重组的信息经过激活函数的放大或抑制后进入下一层的数据重组，从而实现特征提取和知识获取。



在卷积层后通常会使用激活函数，以引入非线性特性，但在某些情况下（如下采样层），可能会省略激活函数。



层级缩放的存在确实是为了减小相加后值的差距，并且通过可学习的缩放因子来提高模型的稳定性和学习能力。这种设计在深层网络中尤其重要，有助于提高模型的性能和收敛速度。

如MLP,注意力，与原图像进行相加

**在本程序得FFN中使用**
