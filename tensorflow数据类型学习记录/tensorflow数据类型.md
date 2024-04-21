# tensorflow数据类型

- **tensor张量：表示任意维度的数据（支持的int,float,double,bool,string）**

- 实例如下

  `In [3] : tf.constant(1)`//int 类型
  `out[3] : <tf. Tensor; id=2,shape=(), dtype=int32,numpy=1>`//shape:每一维的大小 ；dtype：数据类型
  `In[4]: tf.constant(1 . )`//float 类型
  `Out[4] : <tf. Tensor : id=4,shape=()， dtype=float32,numpy=1 .O>`
  `In[5] : tf.constant(2.2，dtype=tf.int32)`//给定value为浮点型,指定dtype为整形会报错

  `In [6]: tf.constant(2., dtype=tf.double)`
  `Out[6]: <tf.Tensor: id=7，shape=(), dtype=float64,numpy=2.0>`
  `In[7] : tf.constant( [True,False])`
  `out[7]: <tf. Tensor: id=9,shape=(2,)，dtype=bool,numpy=array([ True，False])>`
  `In[11]: tf.constant( ' hello, world.' )`
  `out[11]: <tf.Tensor: id=14,shape=(),dtype=string,numpy=b'hello,world.'>`

  - ### tensor的基本用法：

    1. device：表示当前tensor设备环境

       `with tf.device("cpu"):a=tf.constant(1)`

       `with tf.device("gpu"):b=tf.range(4)`

       `a.device`		//返回cpu

       `b.device`		//返回gpu

       `aa=a.gpu()`		//转换a的设备环境

       `aa.device`		//返回gpu

    2. numpy:将tensor转换为numpy

       `In : b.numpy()`
       `out: array([0，1，2，3]，dtype=int32)`

    3. ndim:返回tensor维度

       `in : b.ndim`

       `out : 1`

    4. rank:返回tensor类型

       `In : tf.rank(b)J`
       `out:<tf.Tensor: id=20，shape=()，dtype=int32，numpy=1>`

       `In : tf.rank(tf.ones([3,4,2])`
       `out:<tf.Tensor: id=25，shape=()，dtype=int32，numpy=3>`

    5. is_tensor:判断是否tensor类型

       `In : tf.is_tensor(b)`

       `out: True`

    6. dtype:返回数据类型

       `In : a.dtype,b.dtype,c.dtype`
       `out :(tf.float32,tf.bool, tf.string)`

       `in :a.dtypetf.float32`

       `out :True`

    7. convert_to_tensor:numpy转换为tensor

    8. cast:tensor类型转换

       `In :tf.cast(aa，dtype=tf.float32)`
       `out : <tf.Tensor: id=23，shape=(5,)，dtype=float32，numpy=array([0.，1.，2.，3.，4.]，dtype=float32)>`
    
  - ### tf.optimizer优化器
  
    1. ###### 梯度下降法：
  
       - 定义:
  
         解决局部最优问题。
  
       - 用法：
  
         ```python
         tf.train.GradientDescentOptimizer(learning_rate, use_locking=False, name='GradientDescent')
         ```
  
       
  
    2. ###### Adagrad下降法:
  
       - 定义:
  
         使用Adagrad算法的Optimizer，独立地适应所有模型参数的学习率，缩放每个参数反比于其所有梯度历史平均值总和的平方根。具有代价函数最大梯度的参数相应地有个快速下降的学习率，而具有小梯度的参数在学习率上有相对较小的下降。
         Adagrad 的主要优势在于不需要人为的调节学习率，它可以自动调节；缺点在于，随着迭代次数增多，学习率会越来越小，最终会趋近于0
  
       - 用法：
  
         ```python
         tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1, use_locking=False,name='Adagrad')
         ```
  
       
  
    3. ###### 动量优化法:
  
       - 定义：
  
         使用Momentum算法的Optimiter使用动量的随机下降法，主要思想是积累历史梯度信息动量来加速梯度下降。
  
         动量优化法的优点是收敛快，不容易陷入局部最优解，但是缺点是有时候会冲过头了，使得结果不够精确。
  
       - 用法：
  
         ```python
         tf.train.MomentumOptimizer.__init__(learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False)
         ```
  
       
  
    4. ###### RMSProp算法:
  
       - 定义：
  
         RMSProp算法修改了AdaGrad的梯度积累为指数加权的移动平均，使得其在非凸设定下效果更好。
         RMSProp算法在经验上已经被证明是一种有效且实用的深度神经网络优化算法。目前它是深度学习从业者经常采用的优化方法之一。
  
       - 用法
  
         ```python
         tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')
         ```
  
    5. ###### Adam算法
  
       - 定义:
  
         Adam中动量直接并入了梯度一阶矩（指数加权）的估计。相比于缺少修正因子导致二阶矩估计可能在训练初期具有很高偏置的RMSProp，Adam包括偏置修正，修正从原点初始化的一阶矩（动量项）和（非中心的）二阶矩估计。
         Adam通常被认为对超参数的选择相当鲁棒，尽管学习率有时需要从建议的默认修改。
         在实际运用中Adam效果非常优秀。
  
       - 用法：
  
         ```python
         tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
         ```
  
         
